"""OOF 파이프라인 진단 3종 — 구조 A′의 설계 선택을 실측으로 검증한다.

`decision_log.md` #20의 구현 경로는 **PD 분위 × term 칸별 통계표**를 만들고 `q_score`로
승인선을 긋는 것을 상정한다. 그런데 그 얼개가 값을 하는지는 아직 아무도 재보지 않았다.
여기서 세 가지를 잰다.

| 진단 | 묻는 것 | 판정 |
| --- | --- | --- |
| **C-1** | `q_score` 랭킹이 `pd` 랭킹과 **다른가?** | `\\|ρ\\|`≈0.99 → 칸별 통계표가 값을 못 함, 단순 PD threshold로 복귀(#18 전제)<br>`\\|ρ\\|`<0.9 → 재배열 실재, `q_score` 채택 근거 |
| **C-2** | 칸 안에서 `int_rate`가 **흩어져 있는가?** | sd > 1%p → A′의 건별 계산이 실효 있음<br>sd ≈ 0.3%p → 그룹 평균으로 충분했다는 뜻 |
| **C-3** | Train 경계를 Validation에 적용하면 **10%씩 들어가는가?** | ±2%p 안이면 무시<br>심하면 fold를 5 → 10으로 올린다 |

## C-1의 메커니즘 — 왜 뒤집힐 수 있나

PD 분위가 오르면 두 힘이 반대로 작용한다.

- `mu_정상`은 **올라간다** — LC가 위험한 대출에 높은 `int_rate`를 매기기 때문이다.
- `mu_부도`는 내려간다.

그래서 `E[XR]`이 PD에 대해 **비단조**일 수 있다. 예를 들어 분위 3의 금리 프리미엄이 PD
상승분보다 크면 분위 3이 분위 1보다 앞서게 되고, 그러면 PD 순서와 `q_score` 순서가 갈린다.
이 재배열이 실재하는지가 곧 **칸별 통계표를 만들 이유가 있는지**다.

⚠️ `q_score`는 `XR`이 필요하므로 **잠정 정의**(ⓒ 발행시점 고정 · 수수료 0% · 조기상환 보정 0)로
계산한다(`realized_return.py`). 보정이 확정되면 이 진단을 다시 돌린다 — 모형 재학습은 필요 없다.

실행: `/opt/anaconda3/bin/python src/analysis/oof_diagnostics.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.model import (  # noqa: E402
    DEFAULT_N_FOLDS,
    assign_pd_quantile,
    compute_oof,
    make_quantile_edges,
    predict_default_probability,
    train_model,
)
from analysis.realized_return import (  # noqa: E402
    ReturnAssumptions,
    build_return_inputs,
    contract_return,
    default_cell_stats,
    expected_excess_return,
    issue_risk_free_rate,
    q_score,
    realized_return_defaulted,
    variance_excess_return,
)
from preprocessing.preprocessor import build_feature_table, split_6_2_2  # noqa: E402
from utils.config import repo_root  # noqa: E402

N_QUANTILES = 10


def hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def build_excess_returns(
    outcome: pd.DataFrame, assumptions: ReturnAssumptions
) -> pd.DataFrame:
    """건별 `rf`·`XR_정상`(계약)·`XR_부도`(실현)를 만든다.

    `XR_정상`은 부도 건에도 정의된다 — "계약대로 갚았다면 얼마였을까"라서 실현 여부와
    무관하게 계산되며, `E[XR] = (1−p̂)·XR_정상 + p̂·mu_부도`의 첫 항이 바로 그 값이다.
    """
    rf = issue_risk_free_rate(outcome["issue_month_ord"], outcome["term"])

    r_contract = contract_return(
        installment=outcome["installment"],
        funded_amnt=outcome["funded_amnt"],
        term_months=outcome["term"],
        reinvest_rate=rf,
        assumptions=assumptions,
    )
    r_default = realized_return_defaulted(outcome, reinvest_rate=rf, assumptions=assumptions)

    return pd.DataFrame(
        {
            "rf": rf,
            "xr_normal": r_contract - rf,
            "xr_default": (r_default - rf).where(outcome["is_default"] == 1),
            "term": outcome["term"],
            "int_rate": outcome["int_rate"],
            "is_default": outcome["is_default"],
        }
    )


def quantile_edges_by_term(
    pd_oof: pd.Series, term: pd.Series, n_quantiles: int = N_QUANTILES
) -> dict[float, np.ndarray]:
    """term별로 따로 PD 분위 경계를 만든다 (#20 권고).

    전체 공통 분위로 자르면 **60개월·저PD 칸이 거의 빈다** — 60m는 위험군이라 고PD 구간에
    쏠리기 때문이다. #16으로 60개월 비중이 25.1% → 13.9%로 줄어 이 쏠림이 더 심해졌다.
    """
    return {
        float(t): make_quantile_edges(pd_oof[term == t], n_quantiles)
        for t in sorted(term.dropna().unique())
    }


def assign_quantile_by_term(
    pd_values: pd.Series, term: pd.Series, edges_by_term: dict[float, np.ndarray]
) -> pd.Series:
    out = pd.Series(np.nan, index=pd_values.index, name="pd_quantile")
    for t, edges in edges_by_term.items():
        mask = term == t
        if mask.any():
            out.loc[mask] = assign_pd_quantile(pd_values.loc[mask], edges).to_numpy()
    return out.astype("Int8")


def main() -> None:
    out_dir = repo_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    assumptions = ReturnAssumptions()

    hdr("0. 데이터 구성")
    X, y, meta = build_feature_table()
    parts = split_6_2_2(X, y, meta)
    X_tr, y_tr, meta_tr = parts["train"]
    X_va, y_va, meta_va = parts["validation"]
    print(f"표본 {len(X):,}  피처 {X.shape[1]}  Train {len(X_tr):,}  Validation {len(X_va):,}")
    print(f"가정: {assumptions.label()}  (조기상환 보정 {assumptions.prepayment_adjustment})")

    outcome = build_return_inputs()
    xr = build_excess_returns(outcome, assumptions)
    print(f"수익률 입력 {len(outcome):,}건  "
          f"XR_정상 결측 {xr['xr_normal'].isna().sum():,}  "
          f"XR_부도 결측(부도건 중) {xr.loc[xr['is_default'] == 1, 'xr_default'].isna().sum():,}")

    hdr(f"1. Train {DEFAULT_N_FOLDS}-fold OOF PD")
    pd_oof, fold_auc, _ = compute_oof(X_tr, y_tr)
    print(f"  fold AUC 평균 {np.mean(fold_auc):.5f} (sd {np.std(fold_auc):.5f})")
    print(f"  OOF 전체 AUC {roc_auc_score(y_tr, pd_oof):.5f}")

    term_tr = xr["term"].loc[X_tr.index]
    edges = quantile_edges_by_term(pd_oof, term_tr)
    q_tr = assign_quantile_by_term(pd_oof, term_tr, edges)
    print("\n  [term별 PD 분위 경계]")
    for t, e in edges.items():
        print(f"    {int(t):>3}m  " + " ".join(f"{v:.4f}" for v in e))

    # ---------------------------------------------------------------- 칸별 통계표
    hdr("2. 칸별 부도 통계표 (mu_부도 · var_부도)")
    xr_tr = xr.loc[X_tr.index]
    stats = default_cell_stats(
        xr_tr["xr_default"], q_tr, term_tr
    )
    print(stats[["n", "mu", "var", "shrunk", "mu_unreliable"]].to_string(
        float_format=lambda v: f"{v: .5f}"))
    print(f"\n  pooled var = {stats['pooled_var'].iloc[0]:.5f}  "
          f"축소추정된 칸 {int(stats['shrunk'].sum())}/{len(stats)}")
    stats.reset_index().to_csv(
        out_dir / f"oof_default_cell_stats_{assumptions.label()}.csv", index=False)

    mu_map = stats["mu"].to_dict()
    var_map = stats["var"].to_dict()
    keys = list(zip(term_tr, q_tr))
    mu_default = pd.Series([mu_map.get(k, np.nan) for k in keys], index=X_tr.index)
    var_default = pd.Series([var_map.get(k, np.nan) for k in keys], index=X_tr.index)

    e_xr = expected_excess_return(pd_oof, xr_tr["xr_normal"], mu_default)
    v_xr = variance_excess_return(
        pd_oof, xr_tr["xr_normal"], mu_default, var_default, var_normal=0.0
    )
    q = q_score(e_xr, v_xr)

    # ---------------------------------------------------------------- C-1
    hdr("3. [C-1] Spearman ρ(pd_oof, q_score) — 설계 존폐")
    ok = q.notna() & pd_oof.notna() & e_xr.notna()
    rho_q, _ = spearmanr(pd_oof[ok], q[ok])
    rho_e, _ = spearmanr(pd_oof[ok], e_xr[ok])
    print(f"  유효 {ok.sum():,}건 / {len(ok):,}")
    print(f"  ρ(pd_oof, q_score) = {rho_q:+.5f}")
    print(f"  ρ(pd_oof, E[XR])   = {rho_e:+.5f}")
    print(f"  ρ(E[XR], q_score)  = {spearmanr(e_xr[ok], q[ok])[0]:+.5f}")

    verdict = (
        "|ρ|≥0.99 → 랭킹이 사실상 같다. 칸별 통계표·분산추정의 대가가 0이므로 "
        "#18이 전제한 단순 PD threshold로 돌아가는 것이 정답이다."
        if abs(rho_q) >= 0.99
        else (
            "|ρ|<0.9 → 재배열이 실재한다. q_score를 채택할 근거가 있다."
            if abs(rho_q) < 0.9
            else "0.9≤|ρ|<0.99 → 부분적 재배열. Validation Sharpe 비교로 결정해야 한다."
        )
    )
    print(f"\n  판정: {verdict}")

    print("\n  [PD 분위별 평균 — 비단조성 확인]")
    by_cell = pd.DataFrame(
        {"term": term_tr, "q": q_tr, "pd": pd_oof, "xr_normal": xr_tr["xr_normal"],
         "E_XR": e_xr, "q_score": q, "int_rate": xr_tr["int_rate"]}
    ).groupby(["term", "q"], observed=True).mean(numeric_only=True)
    print(by_cell.to_string(float_format=lambda v: f"{v: .5f}"))
    by_cell.reset_index().to_csv(
        out_dir / f"oof_c1_cell_means_{assumptions.label()}.csv", index=False)

    # ---------------------------------------------------------------- C-2
    hdr("4. [C-2] 칸별 int_rate 표준편차 — A′ 건별 계산의 실효성")
    c2 = (
        pd.DataFrame({"term": term_tr, "q": q_tr, "int_rate": xr_tr["int_rate"]})
        .groupby(["term", "q"], observed=True)["int_rate"]
        .agg(["size", "mean", "std", "min", "max"])
    )
    c2["range"] = c2["max"] - c2["min"]
    print(c2.to_string(float_format=lambda v: f"{v: .4f}"))
    med_sd = c2["std"].median()
    print(f"\n  칸별 sd 중앙값 = {med_sd:.4f}%p")
    print("  판정: " + (
        "sd > 1%p — 칸 안에서 금리가 충분히 흩어져 있다. A′의 건별 계산이 실효 있다."
        if med_sd > 1.0
        else "sd ≈ 0.3%p 수준 — 그룹 평균으로 충분했다는 뜻이다."
        if med_sd < 0.5
        else "sd 0.5~1%p — 중간. 건별 계산의 이득이 크지 않을 수 있다."
    ))
    c2.reset_index().to_csv(out_dir / "oof_c2_int_rate_dispersion.csv", index=False)

    # ---------------------------------------------------------------- C-3
    hdr("5. [C-3] Validation 분위별 인원 분포 — 경계 이전 가능성")
    final = train_model(X_tr, y_tr)
    p_va = predict_default_probability(final, X_va)
    print(f"  Validation AUC {roc_auc_score(y_va, p_va):.5f}   (확정 표본 기준 0.70대)")

    term_va = xr["term"].loc[X_va.index]
    q_va = assign_quantile_by_term(p_va, term_va, edges)

    rows = []
    for t in sorted(edges):
        share_tr = (q_tr[term_tr == t].value_counts(normalize=True).sort_index() * 100)
        share_va = (q_va[term_va == t].value_counts(normalize=True).sort_index() * 100)
        for qi in range(1, N_QUANTILES + 1):
            rows.append({
                "term": int(t), "quantile": qi,
                "train_pct": round(float(share_tr.get(qi, 0.0)), 3),
                "validation_pct": round(float(share_va.get(qi, 0.0)), 3),
                "diff_pp": round(float(share_va.get(qi, 0.0) - share_tr.get(qi, 0.0)), 3),
            })
    c3 = pd.DataFrame(rows)
    print(c3.to_string(index=False))
    worst = c3["diff_pp"].abs().max()
    print(f"\n  최대 이탈 {worst:.2f}%p")
    print("  판정: " + (
        "±2%p 이내 — 무시해도 된다. Train 경계를 그대로 쓴다."
        if worst <= 2.0
        else "±2%p 초과 — fold를 5 → 10으로 올려 fold 모델과 최종 모델의 학습량 차를 줄인다."
    ))
    c3.to_csv(out_dir / "oof_c3_quantile_share.csv", index=False)

    hdr("완료 — 산출물")
    for f in sorted(out_dir.glob("oof_*.csv")):
        print(f"  {f.relative_to(repo_root())}")
    print("\n⚠️ 모든 XR은 잠정값 기준이다 — 조기상환 보정 확정 시 다시 돌린다(모형 재학습 불필요).")


if __name__ == "__main__":
    main()
