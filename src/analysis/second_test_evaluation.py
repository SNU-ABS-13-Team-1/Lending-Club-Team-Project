"""**2nd Test 평가** — 외부 표본(`lending_club_2020_test_2nd.csv`)에서 Sharpe를 낸다.

`final_evaluation.py`가 매니페스트 6:2:2의 Test 20%(144,713건)에 적용한 것과 **완전히 같은
모델·같은 τ\\***를, train 파일과 `id`가 한 건도 겹치지 않는 **별도 파일**(481,833건)에 적용한다.

## 무엇이 `final_evaluation.py`와 같고 무엇이 다른가

| | `final_evaluation.py` | 이 스크립트 |
| --- | --- | --- |
| 모델 | K=50 승자 seed 재현 | **동일** (같은 seed·같은 Train 60%) |
| τ* | 승자 τ* 고정 | **동일** (재탐색 없음) |
| 칸별 통계표·분위 경계 | 승자 seed의 Train | **동일** |
| 적용 대상 | 매니페스트 Test 20% | **2nd Test 파일 전체** |

즉 **모형 쪽은 한 줄도 다르지 않고 적용 대상만 바꾼다.** 그래서 두 결과의 차이는 오롯이
"다른 표본에서도 재현되는가"에 대한 답이다.

## 두 가지 함정과 대응

1. **범주 정렬** — `coerce_dtypes()`는 각 프레임의 값에서 category를 만든다. 2nd Test를 그냥
   읽으면 category **코드가 train과 어긋나** XGBoost가 다른 범주로 해석한다(AUC가 조용히
   무너진다). `align_categories()`가 train 프레임의 categories를 그대로 씌운다 —
   train에 없던 값은 NaN이 되며, 이는 XGBoost가 native로 처리한다.
2. **인덱스 충돌** — 두 파일 모두 0부터 시작하는 행 인덱스라 그대로 합치면 겹친다.
   2nd Test 인덱스에 `INDEX_OFFSET`을 더해 `scores_for_assumptions()`가 쓰는 단일 프레임을
   만든다(그 함수를 그대로 재사용하기 위한 것 — 방법론을 복제하면 갈라진다).

## `oracle_tau` 행을 어떻게 읽는가

산출 CSV에는 **2nd Test에서 다시 탐색한 τ**(`oracle_tau`)가 진단용으로 들어 있다.
**이 값을 성과로 인용하지 않는다** — 적용 대상을 보고 고른 값이라 정의상 낙관적이다.
쓰임은 하나뿐이다: `winner_tau`와의 차이가 작으면 τ 이전이 잘 된 것이고, 크면 τ가
분할에 과적합됐다는 뜻이다.

실행:
    /opt/anaconda3/bin/python src/analysis/second_test_evaluation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.config import repo_root
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import repo_root

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.final_evaluation import CRITERION_KEYS, select_best_model  # noqa: E402
from analysis.model import (  # noqa: E402
    apply_calibrator,
    assign_quantile_by_term,
    compute_oof,
    fit_calibrator,
    predict_default_probability,
    quantile_edges_by_term,
    train_model,
)
from analysis.realized_return import (  # noqa: E402
    ReturnAssumptions,
    build_return_inputs,
    cash_reinvestment,
)
from analysis.sharpe_optimizer import (  # noqa: E402
    approve_all_sharpe,
    evaluate_threshold,
    find_optimal_threshold,
    load_pipeline_data,
    scores_for_assumptions,
)
from preprocessing.loader import second_test_path  # noqa: E402
from preprocessing.preprocessor import (  # noqa: E402
    build_feature_table,
    resplit_train_validation,
)

OUT_CSV = "second_test_evaluation.csv"
HEADLINE_REINVEST = "treasury"

# 2nd Test 인덱스에 더할 값. train 원본이 1,755,295행이므로 어떤 값과도 겹치지 않는다.
INDEX_OFFSET = 10_000_000


# ---------------------------------------------------------------------------
# 2nd Test 로딩
# ---------------------------------------------------------------------------
def align_categories(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`X_new`의 category 컬럼을 **`X_ref`의 categories로 다시 씌운다.**

    XGBoost는 category dtype을 **코드(정수)** 로 받으므로, 두 프레임이 각자 값에서 만든
    categories를 쓰면 같은 코드가 다른 범주를 가리킨다. 학습 때 본 적 없는 값은 NaN이 되며
    (native 결측 처리로 흘러간다) — 없는 범주에 대한 분기는 애초에 학습되지 않았다.
    """
    out = X_new.copy()
    unseen: dict[str, int] = {}
    for col in X_ref.columns:
        if not isinstance(X_ref[col].dtype, pd.CategoricalDtype):
            continue
        cats = X_ref[col].cat.categories
        values = out[col].astype("object")
        n_unseen = int(values.notna().sum() - values.isin(cats).sum())
        if n_unseen:
            unseen[col] = n_unseen
        out[col] = pd.Categorical(values, categories=cats, ordered=X_ref[col].cat.ordered)
    return out, unseen


def load_second_test(X_ref: pd.DataFrame, verbose: bool = True) -> tuple:
    """2nd Test를 **같은 파이프라인**으로 읽어 `(X2, y2, meta2, outcome2)`를 돌려준다.

    표본 필터는 동일하되 건수 검증만 끈다 — `EXPECTED_SAMPLE_SIZE`는 train 기준이다.
    인덱스에는 `INDEX_OFFSET`이 더해져 있다.
    """
    path = second_test_path()
    if verbose:
        print(f"  파일 {path.name}")

    X2, y2, meta2 = build_feature_table(verify_sample=False, csv_path=path)
    outcome2 = build_return_inputs(csv_path=path, verify_sample=False)

    if list(X2.columns) != list(X_ref.columns):
        raise ValueError(
            "2nd Test의 피처 구성이 train과 다르다 — 컬럼 목록을 확인하라.\n"
            f"  train에만: {sorted(set(X_ref.columns) - set(X2.columns))}\n"
            f"  test에만 : {sorted(set(X2.columns) - set(X_ref.columns))}"
        )
    if not X2.index.equals(outcome2.index):
        raise ValueError("피처 테이블과 수익률 입력의 인덱스가 어긋난다 — 필터가 다르게 걸렸다.")

    X2, unseen = align_categories(X_ref, X2)
    if verbose:
        print(f"  표본 {len(X2):,}건  부도율 {y2.mean():.4%}  피처 {X2.shape[1]}")
        if unseen:
            top = sorted(unseen.items(), key=lambda kv: -kv[1])[:5]
            total = sum(unseen.values())
            print(f"  ⚠️ train에 없던 범주값 {total:,}건 → NaN 처리 "
                  f"({', '.join(f'{c} {n:,}' for c, n in top)})")
        else:
            print("  범주값은 전부 train에서 관측된 것이다")

    idx = pd.Index(X2.index + INDEX_OFFSET, name=X2.index.name)
    for frame in (X2, meta2, outcome2):
        frame.index = idx
    y2.index = idx
    return X2, y2, meta2, outcome2


# ---------------------------------------------------------------------------
# 승자 재현 → 2nd Test 예측
# ---------------------------------------------------------------------------
def rebuild_winner_for_second_test(seed: int, data: tuple, verbose: bool = True) -> dict:
    """승자 seed의 모델을 재현하고, 2nd Test 점수 재료를 `bundle` 형태로 만든다.

    `scores_for_assumptions()`가 그대로 먹도록 train·2nd Test의 `outcome`을 **하나의 프레임으로
    이어붙인다**(인덱스가 어긋나 있으므로 안전하다). 방법론(분위 경계는 Train OOF PD, 칸별
    통계표는 Train에서만, `E[XR]`의 `p̂`는 보정 후 PD)을 K=50·`final_evaluation`과 한 줄도
    다르게 하지 않기 위한 것이다.
    """
    X, y, meta, outcome = data
    parts = resplit_train_validation(X, y, meta, seed=seed)
    X_tr, y_tr, _ = parts["train"]

    print("\n[2nd Test 로딩 — 같은 전처리 파이프라인]")
    X2, y2, _, outcome2 = load_second_test(X_tr, verbose=verbose)
    if outcome.index.intersection(outcome2.index).size:
        raise ValueError("인덱스 오프셋이 충분하지 않다 — INDEX_OFFSET을 키워라.")

    if verbose:
        print(f"\n[승자 재현] seed {seed}  학습 {len(X_tr):,}건  2nd Test {len(X2):,}건")

    pd_oof, fold_auc, _ = compute_oof(X_tr, y_tr, seed=seed, verbose=verbose)
    calibrator = fit_calibrator(pd_oof, y_tr)
    final = train_model(X_tr, y_tr, seed=seed)
    p2_raw = predict_default_probability(final, X2)
    p2_cal = apply_calibrator(calibrator, p2_raw)

    return {
        "pd_oof_raw": pd_oof,
        "p_va_raw": p2_raw,          # "적용 대상" 자리 = 2nd Test
        "p_va_cal": p2_cal,
        "train_index": X_tr.index,
        "validation_index": X2.index,
        "outcome": pd.concat([outcome, outcome2]),
        "fold_auc": fold_auc,
        "edges_fn": quantile_edges_by_term,
        "assign_fn": assign_quantile_by_term,
        "y_second": y2,
    }


def evaluate_on_second_test(
    bundle: dict, best: pd.Series, criterion: str, verbose: bool = True
) -> pd.DataFrame:
    """재투자 가정 2종 × (승자 τ* · K=50 중앙값 τ* · 진단용 oracle τ)를 평가한다.

    추가 학습은 없다 — 모형은 하나고 정렬·컷만 바꾼다(#19).
    """
    from sklearn.metrics import roc_auc_score

    key = CRITERION_KEYS[criterion]
    treasury = ReturnAssumptions()
    y2 = bundle["y_second"]
    auc2 = float(roc_auc_score(y2, bundle["p_va_raw"]))
    rows: list[dict] = []

    for assumptions in (treasury, cash_reinvestment(treasury)):
        scores, realized, audit = scores_for_assumptions(bundle, assumptions)
        score, lower = scores[key]
        base = approve_all_sharpe(realized)
        oracle = find_optimal_threshold(score, realized, lower)

        taus = {
            "winner_tau": float(best["threshold"]),    # 헤드라인 — Validation에서 확정된 값
            "median_tau": float(best["median_tau"]),   # 선택 편의 진단용
            "oracle_tau": float(oracle["threshold"]),  # ⚠️ 2nd Test를 보고 고른 값 — 인용 금지
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
                "auc_second_test": auc2,
                "n_second_usable": audit["n_usable"],
                "n_second_dropped": audit["n_dropped"],
                **res,
                "sharpe_approve_all": base,
                "delta_sharpe": res["sharpe"] - base,
                "val_sharpe_winner": float(best["sharpe"]),
                "val_sharpe_mean": float(best["val_sharpe_mean"]),
                "val_approval_rate": float(best["approval_rate"]),
            })
            if verbose:
                mark = "  ← 진단 전용" if vname == "oracle_tau" else ""
                print(f"  [{assumptions.reinvest:8s}] {vname:11s} τ={tau:.6f}  "
                      f"승인율 {res['approval_rate']:.1%}  "
                      f"Sharpe {res['sharpe']:.4f}  Δ{res['sharpe'] - base:+.4f}  "
                      f"(approve-all {base:.4f}){mark}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="2nd Test 평가 — 외부 표본 Sharpe")
    ap.add_argument("--criterion", default="q_score", choices=list(CRITERION_KEYS),
                    help="Validation K=50에서 확정한 승인선 랭킹 기준 (기본: q_score)")
    ap.add_argument("--reinvest", default=HEADLINE_REINVEST, choices=["treasury", "cash"],
                    help="승자를 고르는 기준이 되는 재투자 가정 (기본: treasury — #18)")
    ap.add_argument("--min-k", type=int, default=50, help="이 K에 못 미치면 중단 (기본 50)")
    args = ap.parse_args()

    print("=" * 78)
    print("2nd Test 평가 — lending_club_2020_test_2nd.csv (train과 id 무교집합)")
    print("=" * 78)
    print(f"랭킹 기준 : {args.criterion}   선택 기준 가정 : {args.reinvest}")
    print("모델·τ*는 final_evaluation.py와 **동일**하다 — 적용 대상만 바꾼다.")
    print("⚠️ XR은 **잠정 가정**이다 — 국채 ⓒ발행시점 고정 · 수수료 0% · 조기상환 보정 0(#20).")

    best = select_best_model(args.criterion, args.reinvest, min_k=args.min_k)
    print(f"\n[승자] seed {int(best['seed'])}  Validation Sharpe {best['sharpe']:.4f} "
          f"(K={int(best['K'])} 중 1위)   τ* {best['threshold']:.6f}")

    print("\n[train 원본 로딩]")
    data = load_pipeline_data()

    bundle = rebuild_winner_for_second_test(int(best["seed"]), data)
    result = evaluate_on_second_test(bundle, best, args.criterion)

    out_path = repo_root() / "outputs" / OUT_CSV
    result.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("2nd Test 결과")
    print("=" * 78)
    show = result[["variant", "reinvest", "threshold", "approval_rate", "sharpe",
                   "sharpe_approve_all", "delta_sharpe"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    head = result[(result["variant"] == "winner_tau")
                  & (result["reinvest"] == HEADLINE_REINVEST)]
    if not head.empty:
        h = head.iloc[0]
        print(f"\n▶ 2nd Test Sharpe Ratio = **{h['sharpe']:.4f}**  "
              f"(approve-all {h['sharpe_approve_all']:.4f}, Δ {h['delta_sharpe']:+.4f}, "
              f"승인율 {h['approval_rate']:.1%})")
        print(f"  2nd Test AUC {h['auc_second_test']:.5f}  "
              f"(승자 seed의 Train fold AUC {h['fold_auc_mean']:.5f})")
        print(f"  유효 {int(h['n_second_usable']):,}건 / 제외 {int(h['n_second_dropped']):,}건")

    print(f"\n산출물 → outputs/{OUT_CSV}")
    print("⚠️ `oracle_tau` 행은 2nd Test를 보고 고른 값이다 — 성과로 인용하지 않는다.")


if __name__ == "__main__":
    main()
