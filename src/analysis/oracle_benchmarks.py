"""**벤치마크·오라클 비교** — 외부 검증 표본에서 도달 가능한 Sharpe의 범위를 낸다.

`final_report_v3.md` 6.3절 표 6-2의 구현이다. 같은 외부 검증 표본·같은 실현 `XR`에
승인 규칙만 바꿔 다섯 가지를 계산한다. **모형은 하나이고 재학습은 없다** — 정렬과 컷만
바꾼다(#19).

| 기준 | 승인 규칙 | 쓰는 정보 |
| --- | --- | --- |
| `all_treasury` | 아무 건도 승인하지 않는다 | 없음 |
| `approve_all` | 전부 승인 (대조군, #17 ②) | 없음 |
| `model_tau` | `q_score ≥ τ*` (Validation에서 확정) | 승인 시점 정보만 |
| `oracle_status` | 정상상환(Fully Paid) 건만 승인 | **부도 여부** (사후) |
| `oracle_xr_pos` | 실현 `XR > 0`인 건만 승인 | **실현 초과수익 부호** (사후) |

`oracle_tau_current`는 표 6-1의 `사후 최적 τ`와 같은 값이다 — 같은 `q_score` 랭킹 안에서
컷만 사후에 고른 것이므로 **완전예지 오라클이 아니다.** 두 개념이 "oracle"이라는 한 단어로
섞여 있던 것을 갈라 놓기 위해 함께 낸다.

## 왜 필요한가

절대 Sharpe 0.2068이 높은지 낮은지는 그 자체로 판단할 수 없다. 전부 승인(0.1175)과
완전예지 오라클(1.4294) 사이의 어디인지가 판단 근거이며, 이 구간에서 모형이 회수한
몫(6.8%)의 상한을 결정하는 것이 판별력(AUC 0.71)이라는 정보 한계다.

⚠️ `all_treasury`는 승인 건이 없어 평균도 표준편차도 0이라 **Sharpe가 0/0으로 정의되지
않는다**(`NaN`으로 나온다). 0으로 바꿔 적지 않는다.

실행 (모형 재학습이 있어 약 10분):
    /opt/anaconda3/bin/python src/analysis/oracle_benchmarks.py [--scheme 8_2]
    → outputs/oracle_benchmarks_8_2.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from utils.config import repo_root
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import repo_root

from analysis.final_evaluation import CRITERION_KEYS
from analysis.realized_return import ReturnAssumptions, cash_reinvestment
from analysis.second_test_evaluation import (
    load_pipeline_data,
    rebuild_winner_for_second_test,
    select_best_model,
)
from analysis.sharpe_optimizer import (
    evaluate_threshold,
    find_optimal_threshold,
    portfolio_excess_returns,
    scores_for_assumptions,
    sharpe_ratio,
)

OUT_STEM = "oracle_benchmarks"
CRITERION = "q_score"


def _row(realized: pd.Series, approved: pd.Series) -> dict:
    """승인 마스크 하나를 평가한다. 분모는 거절 건을 포함한 전체 표본이다(#18)."""
    pxr = portfolio_excess_returns(realized, approved)
    return {
        "n_approved": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "mean_xr": float(pxr.mean()),
        "sd_xr": float(pxr.std(ddof=1)),
        "sharpe": sharpe_ratio(pxr),
    }


def benchmark_table(bundle: dict, best: pd.Series, criterion: str = CRITERION) -> pd.DataFrame:
    """재투자 가정 2종 × 승인 규칙 6종을 평가한다."""
    key = CRITERION_KEYS[criterion]
    y_second = bundle["y_second"]
    treasury = ReturnAssumptions()
    rows: list[dict] = []

    for assumptions in (treasury, cash_reinvestment(treasury)):
        scores, realized, audit = scores_for_assumptions(bundle, assumptions)
        score, lower = scores[key]
        y = y_second.reindex(realized.index)
        if y.isna().any():
            raise ValueError("부도 라벨이 실현 XR 인덱스에 정렬되지 않는다.")

        def cut(tau: float) -> pd.Series:
            return score <= tau if lower else score >= tau

        rules = {
            "all_treasury": pd.Series(False, index=realized.index),
            "approve_all": pd.Series(True, index=realized.index),
            "model_tau": cut(float(best["threshold"])),
            "oracle_status": y == 0,          # 부도 여부를 미리 안다
            "oracle_xr_pos": realized > 0,    # 실현 초과수익 부호를 미리 안다
        }
        for name, approved in rules.items():
            rows.append({
                "reinvest": assumptions.reinvest,
                "assumptions": assumptions.label(),
                "variant": name,
                "n_total": int(len(realized)),
                **_row(realized, approved),
            })

        orc = find_optimal_threshold(score, realized, lower)
        res = evaluate_threshold(score, realized, float(orc["threshold"]),
                                 lower_is_better=lower)
        rows.append({
            "reinvest": assumptions.reinvest,
            "assumptions": assumptions.label(),
            "variant": "oracle_tau_current",
            "n_total": int(len(realized)),
            **_row(realized, cut(float(orc["threshold"]))),
            "threshold": res["threshold"],
        })
        print(f"  [{assumptions.reinvest:8s}] 유효 {audit['n_usable']:,}건 "
              f"제외 {audit['n_dropped']:,}건  부도율 {float(y.mean()):.4f}")

    out = pd.DataFrame(rows)
    base = out[out["variant"] == "approve_all"].set_index("reinvest")["sharpe"]
    out["sharpe_approve_all"] = out["reinvest"].map(base)
    out["delta_sharpe"] = out["sharpe"] - out["sharpe_approve_all"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", default="8_2", help="분할 체계 (기본 8_2)")
    parser.add_argument("--min-k", type=int, default=50, help="이 K에 못 미치면 중단")
    args = parser.parse_args()

    best = select_best_model(CRITERION, "treasury", min_k=args.min_k, scheme=args.scheme)
    print(f"[선정 모형] seed {int(best['seed'])}  τ* {best['threshold']:.6f}")

    data = load_pipeline_data()
    bundle = rebuild_winner_for_second_test(int(best["seed"]), data, scheme=args.scheme)
    result = benchmark_table(bundle, best)
    result.insert(0, "scheme", args.scheme)

    out_path = repo_root() / "outputs" / f"{OUT_STEM}_{args.scheme}.csv"
    result.to_csv(out_path, index=False)

    show = result[["reinvest", "variant", "approval_rate", "mean_xr", "sd_xr",
                   "sharpe", "delta_sharpe"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
