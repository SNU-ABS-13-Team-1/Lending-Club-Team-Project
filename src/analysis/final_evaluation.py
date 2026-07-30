"""**Test 1회 평가 — 최종 Sharpe Ratio 확정.**

파이프라인의 마지막 단계다. 앞 단계에서 확정된 것만 받아서 적용한다.

## 무엇을 Test에 적용하는가 (2026-07-30 결정)

`sharpe_optimizer.py --repeat 0-49`이 낸 **50개 모델 중 Validation Sharpe가 가장 높은 모델**을
그대로 Test에 적용한다.

- 50개 모델은 **같은 스펙**이고 매니페스트 80% 풀을 6:2로 다시 가른 **분할만 다르다.**
  따라서 "최고 모델"은 `(seed, train 분할, 모형, 보정기, 분위경계, 칸별 통계표, τ*)` 한 묶음이며,
  이 스크립트는 그 seed를 **똑같이 재현**해 Test에 적용한다(`resplit_train_validation(seed)`가
  결정적이므로 재현된다).
- 최종 모형의 학습 풀은 그 seed의 **Train 60%** 다 — 이긴 모델을 그대로 쓰기 때문이다.
  Train+Validation 80%로 다시 학습하면 τ*를 만든 모형과 다른 모형이 Test에 가게 된다.
- **Test에서 threshold를 다시 찾지 않는다.** `find_optimal_threshold()`를 부르지 않고
  `evaluate_threshold()`에 이긴 seed의 τ*를 고정값으로 넣는다. Test에서 τ를 재탐색하면
  "Test set으로 모형을 재조정하지 않는다"(`src/analysis/AGENTS.md`) 위반이다.
- **랭킹 기준은 하나만 Test에 적용한다.** 세 기준(`pd`/`E[XR]`/`q_score`)을 Test에서 비교하면
  규칙 위반이므로, Validation K=50에서 승자를 확정한 뒤 그 하나만 넣는다.

## 보고 시 함께 낼 것 — 선택 규칙의 낙관 편의

최고값 선택은 **분할 운을 성과에 포함**한다(승자의 저주). 그래서 산출 CSV에는 헤드라인과 함께
`median_tau` 행을 남긴다 — 같은 Test 점수에 **K=50 τ* 중앙값**을 적용한 값이며, 추가 학습이
없다(정렬만 다르다). 두 값의 차이가 곧 선택 규칙이 만든 낙관분이다.

Validation K=50에서 절대 Sharpe는 평균 0.298 / 최대 0.304였다(제외 스펙 실측) — 최대와 평균의
간격 약 +0.006이 그 크기의 눈금이다. 리포트에서는 **Validation 최고값을 Test 성과로 인용하지
않는다** — Test 값이 최종 성과다.

실행:
    python src/analysis/final_evaluation.py                  # 기본: q_score
    python src/analysis/final_evaluation.py --criterion pd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.config import load_config, repo_root
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import load_config, repo_root

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.model import (  # noqa: E402
    apply_calibrator,
    assign_quantile_by_term,
    compute_oof,
    fit_calibrator,
    predict_default_probability,
    quantile_edges_by_term,
    train_model,
)
from analysis.realized_return import ReturnAssumptions, cash_reinvestment  # noqa: E402
from analysis.sharpe_optimizer import (  # noqa: E402
    approve_all_sharpe,
    evaluate_threshold,
    load_pipeline_data,
    scores_for_assumptions,
)
from preprocessing.preprocessor import (  # noqa: E402
    resplit_train_validation,
    split_from_manifest,
)

REPEAT_CSV = "sharpe_repeat_k50.csv"
OUT_CSV = "final_test_evaluation.csv"

# 랭킹 기준의 짧은 이름 → `scores_for_assumptions()`가 돌려주는 키
CRITERION_KEYS = {
    "pd": "pd (보정 전)",
    "E[XR]": "E[XR]",
    "q_score": "q_score",
}

# 헤드라인 재투자 가정 — #18 확정(잔존기간 매칭 국채). `cash`는 민감도.
HEADLINE_REINVEST = "treasury"


# ---------------------------------------------------------------------------
# 어느 모델이 이겼는가
# ---------------------------------------------------------------------------
def select_best_model(
    criterion: str,
    reinvest: str = HEADLINE_REINVEST,
    min_k: int = 50,
    scheme: str = "6_2_2",
) -> pd.Series:
    """K=50 중 **Validation Sharpe 최고** seed의 행을 돌려준다.

    `min_k`에 못 미치면 예외로 중단한다 — 반복이 덜 끝난 상태에서 "최고"를 뽑으면 아직
    돌지 않은 seed가 더 좋을 수 있고, **Test는 1회뿐이라 되돌릴 수 없다.**

    `scheme`은 어느 반복 결과를 읽을지 정한다 (`repeat_output_path()`).
    """
    from analysis.sharpe_optimizer import repeat_output_path

    path = repeat_output_path(scheme)
    if not path.exists():
        raise FileNotFoundError(
            f"K=50 결과가 없다: {path}\n"
            f"먼저 `python src/analysis/sharpe_optimizer.py --repeat 0-49 --scheme {scheme}`를 돌려라."
        )
    key = CRITERION_KEYS[criterion]
    df = pd.read_csv(path)
    sub = df[(df["criterion"] == key) & (df["reinvest"] == reinvest)]
    if sub.empty:
        raise ValueError(f"'{key}' × '{reinvest}' 결과가 없다 — 인자를 확인하라.")

    k = int(sub["seed"].nunique())
    if k < min_k:
        raise RuntimeError(
            f"K={k} < {min_k} — 반복이 아직 덜 끝났다. 남은 seed가 더 좋을 수 있고 "
            "Test는 1회뿐이므로 여기서 중단한다. 끝까지 돌린 뒤 다시 실행하거나 "
            "의도한 것이면 --min-k로 낮춰라."
        )

    best = sub.loc[sub["sharpe"].idxmax()].copy()
    best["K"] = k
    best["median_tau"] = float(sub["threshold"].median())
    best["val_sharpe_mean"] = float(sub["sharpe"].mean())
    best["val_sharpe_sd"] = float(sub["sharpe"].std(ddof=1))
    best["val_sharpe_rank"] = int((sub["sharpe"] > best["sharpe"]).sum()) + 1
    return best


# ---------------------------------------------------------------------------
# 이긴 모델 재현 → Test 예측
# ---------------------------------------------------------------------------
def rebuild_winner_for_test(
    seed: int, data: tuple, te_idx: pd.Index, verbose: bool = True
) -> dict:
    """이긴 seed의 모델을 **그대로 재현**하고 Test 점수 재료를 만든다.

    `scores_for_assumptions()`가 먹는 형태로 돌려준다 — `validation_index` 자리에 Test
    인덱스를 넣는다. 방법론(분위 경계는 Train OOF PD, 칸별 통계표는 Train에서만, `E[XR]`의
    `p̂`는 보정 후 PD)을 K=50 단계와 **한 줄도 다르게 하지 않기** 위한 것이다.
    """
    X, y, meta, outcome = data
    parts = resplit_train_validation(X, y, meta, seed=seed)
    X_tr, y_tr, _ = parts["train"]
    X_te = X.loc[te_idx]
    if verbose:
        print(f"\n[이긴 모델 재현] seed {seed}  학습 {len(X_tr):,}건  Test {len(X_te):,}건")

    pd_oof, fold_auc, _ = compute_oof(X_tr, y_tr, seed=seed, verbose=verbose)
    calibrator = fit_calibrator(pd_oof, y_tr)
    final = train_model(X_tr, y_tr, seed=seed)
    p_te_raw = predict_default_probability(final, X_te)
    p_te_cal = apply_calibrator(calibrator, p_te_raw)

    return {
        "pd_oof_raw": pd_oof,
        "p_va_raw": p_te_raw,      # `scores_for_assumptions()`의 "적용 대상" 자리 = Test
        "p_va_cal": p_te_cal,
        "train_index": X_tr.index,
        "validation_index": te_idx,
        "outcome": outcome,
        "fold_auc": fold_auc,
        "edges_fn": quantile_edges_by_term,
        "assign_fn": assign_quantile_by_term,
    }


def evaluate_on_test(
    bundle: dict, best: pd.Series, criterion: str, verbose: bool = True
) -> pd.DataFrame:
    """재투자 가정 2종 × (이긴 τ* · K=50 중앙값 τ*)를 Test에서 평가한다.

    추가 학습은 없다 — 모형은 하나고 정렬·컷만 바꾼다(#19).
    """
    key = CRITERION_KEYS[criterion]
    treasury = ReturnAssumptions()
    rows: list[dict] = []

    for assumptions in (treasury, cash_reinvestment(treasury)):
        scores, realized, audit = scores_for_assumptions(bundle, assumptions)
        score, lower = scores[key]
        base = approve_all_sharpe(realized)

        taus = {
            "winner_tau": float(best["threshold"]),      # 헤드라인
            "median_tau": float(best["median_tau"]),      # 선택 편의 진단용
        }
        for vname, tau in taus.items():
            res = evaluate_threshold(score, realized, tau, lower_is_better=lower)
            rows.append({
                "variant": vname,
                "reinvest": assumptions.reinvest,
                "assumptions": assumptions.label(),
                "criterion": criterion,
                "best_seed": int(best["seed"]),
                "K": int(best["K"]),
                "fold_auc_mean": float(np.mean(bundle["fold_auc"])),
                "n_test_usable": audit["n_usable"],
                "n_test_dropped": audit["n_dropped"],
                **res,
                "sharpe_approve_all": base,
                "delta_sharpe": res["sharpe"] - base,
                "val_sharpe_winner": float(best["sharpe"]),
                "val_sharpe_mean": float(best["val_sharpe_mean"]),
                "val_approval_rate": float(best["approval_rate"]),
            })
            if verbose:
                print(f"  [{assumptions.reinvest:8s}] {vname:11s} τ={tau:.6f}  "
                      f"승인율 {res['approval_rate']:.1%}  "
                      f"Test Sharpe {res['sharpe']:.4f}  Δ{res['sharpe'] - base:+.4f}  "
                      f"(approve-all {base:.4f})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Test 1회 평가 — 최종 Sharpe Ratio 확정")
    ap.add_argument("--criterion", default="q_score", choices=list(CRITERION_KEYS),
                    help="Validation K=50에서 확정한 승인선 랭킹 기준 (기본: q_score)")
    ap.add_argument("--reinvest", default=HEADLINE_REINVEST, choices=["treasury", "cash"],
                    help="최고 모델을 고를 기준이 되는 재투자 가정 (기본: treasury — #18)")
    ap.add_argument("--min-k", type=int, default=50,
                    help="이 K에 못 미치면 중단한다 (기본 50 — #18)")
    args = ap.parse_args()

    print("=" * 78)
    print("최종 Test 평가 — Test set을 여는 단 한 번의 실행")
    print("=" * 78)
    print(f"랭킹 기준 : {args.criterion}   선택 기준 가정 : {args.reinvest}")
    print("선택 규칙 : K=50 중 **Validation Sharpe 최고** 모델을 그대로 적용")
    print("⚠️ XR은 **잠정 가정**이다 — 국채 ⓒ발행시점 고정 · 수수료 0% · 조기상환 보정 0(#20).")

    best = select_best_model(args.criterion, args.reinvest, min_k=args.min_k)
    print(f"\n[이긴 모델] seed {int(best['seed'])}  "
          f"Validation Sharpe {best['sharpe']:.4f} "
          f"(K={int(best['K'])} 중 1위, 평균 {best['val_sharpe_mean']:.4f} "
          f"· sd {best['val_sharpe_sd']:.4f})")
    print(f"  τ* {best['threshold']:.6f}   승인율 {best['approval_rate']:.1%}   "
          f"Δ {best['delta_sharpe']:+.4f}")
    print(f"  참고 — K=50 τ* 중앙값 {best['median_tau']:.6f} (선택 편의 진단용으로 병기)")

    print("\n[원본 로딩]")
    data = load_pipeline_data()
    X, y, meta, _ = data

    print("\n[분할 — Test를 연다]")
    parts = split_from_manifest(X, y, meta, unlock_test=True)
    te_idx = parts["test"][0].index

    bundle = rebuild_winner_for_test(int(best["seed"]), data, te_idx)
    result = evaluate_on_test(bundle, best, args.criterion)

    out_path = repo_root() / "outputs" / OUT_CSV
    result.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("최종 결과")
    print("=" * 78)
    show = result[["variant", "reinvest", "threshold", "approval_rate", "sharpe",
                   "sharpe_approve_all", "delta_sharpe", "val_sharpe_winner"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    head = result[(result["variant"] == "winner_tau")
                  & (result["reinvest"] == HEADLINE_REINVEST)]
    if not head.empty:
        h = head.iloc[0]
        print(f"\n▶ 최종 Sharpe Ratio = **{h['sharpe']:.4f}**  "
              f"(approve-all {h['sharpe_approve_all']:.4f}, Δ {h['delta_sharpe']:+.4f}, "
              f"승인율 {h['approval_rate']:.1%}, seed {int(h['best_seed'])})")
        print(f"  Validation에서 본 값 {h['val_sharpe_winner']:.4f} → Test {h['sharpe']:.4f} "
              f"(차이 {h['sharpe'] - h['val_sharpe_winner']:+.4f})")
        print("  ※ Validation 값은 그 분할에서 최고였던 값이다 — 최종 성과는 Test 쪽이다.")

    print(f"\n산출물 → outputs/{OUT_CSV}")
    print("⚠️ 이 실행으로 Test는 소진됐다. 결과를 보고 모형·threshold를 바꾸면 규칙 위반이다.")


if __name__ == "__main__":
    main()
