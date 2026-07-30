"""OOF 파이프라인 진단 4종 — 구조 A′의 설계 선택을 실측으로 검증한다.

`decision_log.md` #20의 구현 경로는 **PD 분위 × term 칸별 통계표**를 만들고 `q_score`로
승인선을 긋는 것을 상정한다. 그런데 그 얼개가 값을 하는지는 아직 아무도 재보지 않았다.
여기서 네 가지를 잰다.

| 진단 | 묻는 것 | 판정 |
| --- | --- | --- |
| **C-1** | `q_score` 랭킹이 `pd` 랭킹과 **다른가?** | `\\|ρ\\|`≈0.99 → 칸별 통계표가 값을 못 함, 단순 PD threshold로 복귀(#18 전제)<br>`\\|ρ\\|`<0.9 → 재배열 실재, `q_score` 채택 근거 |
| **C-2** | 칸 안에서 `int_rate`가 **흩어져 있는가?** | sd > 1%p → A′의 건별 계산이 실효 있음<br>sd ≈ 0.3%p → 그룹 평균으로 충분했다는 뜻 |
| **C-3** | Train 경계를 Validation에 적용하면 **10%씩 들어가는가?** | ±2%p 안이면 무시<br>심하면 fold를 5 → 10으로 올린다 |
| **C-4** | XGBoost `p̂`가 **확률로서 맞는가?** (isotonic 보정) | ECE > 0.5%p → 보정 필수<br>ECE ≈ 0 → 보정해도 `E[XR]`이 안 바뀜 |

## C-4가 왜 다른 셋보다 앞서 실행되나

구조 A′는 `p̂`를 **순위가 아니라 확률 값**으로 쓴다(`E[XR] = (1−p̂)·XR_정상 + p̂·mu_부도`).
따라서 보정을 나중에 붙이면 `E[XR]`·`q_score`가 전부 바뀌어 **C-1을 다시 돌려야 한다.**
그래서 보정을 먼저 학습하고, C-1이 처음부터 보정된 `p̂`로 `E[XR]`을 만든다.

## PD의 두 역할을 나눈다 (C-4가 실측으로 정한 것)

| 쓰임 | 어느 PD | 왜 |
| --- | --- | --- |
| PD 분위 **경계·배정** (칸을 묶는 도구) | **보정 전** | isotonic 평탄구간이 경계를 삼켜 칸 인원이 10%에서 기운다 |
| `E[XR]`·`Var[XR]`·`q_score`의 `p̂` (**확률 값**) | **보정 후** | 보정 오차가 `E[XR]`에 직접 편향으로 들어간다 |
| 승인선 **점수**로서의 `pd` | **보정 전** | 보정은 순위를 안 바꾸면서 동순위만 만든다 — 점수로는 잃을 것만 있다 |

⚠️ isotonic은 비감소 **계단함수**다. "단조변환이니 분위 배정이 그대로"는 정확히는 틀리다 —
수십만 개 PD가 수백 개 값으로 뭉치고, C-4가 그 결과 칸이 얼마나 기우는지 직접 잰다.

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
    apply_calibrator,
    assign_quantile_by_term,
    calibration_metrics,
    calibration_noise_floor,
    calibration_table,
    compute_oof,
    fit_calibrator,
    predict_default_probability,
    quantile_edges_by_term,
    train_model,
)
from analysis.realized_return import (  # noqa: E402
    ReturnAssumptions,
    build_excess_returns,
    build_return_inputs,
    default_cell_stats,
    expected_excess_return,
    q_score,
    variance_excess_return,
)
from preprocessing.preprocessor import build_feature_table, split_from_manifest  # noqa: E402
from utils.config import repo_root  # noqa: E402

N_QUANTILES = 10


def hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# `build_excess_returns()`(→ `realized_return.py`)와 `quantile_edges_by_term()` ·
# `assign_quantile_by_term()`(→ `model.py`)은 본 파이프라인(`sharpe_optimizer.py`)도 쓰므로
# 각자의 정본 모듈로 옮겼다. 진단 스크립트가 정의를 들고 있으면 두 벌이 갈라진다.


def main() -> None:
    out_dir = repo_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    assumptions = ReturnAssumptions()

    hdr("0. 데이터 구성")
    X, y, meta = build_feature_table()
    parts = split_from_manifest(X, y, meta)
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

    # ---------------------------------------------------------------- C-4
    hdr("1-b. [C-4] 확률보정(isotonic) — p̂를 확률 값으로 쓸 수 있는가")
    calibrator = fit_calibrator(pd_oof, y_tr)
    pd_oof_cal = apply_calibrator(calibrator, pd_oof)

    floor = calibration_noise_floor(pd_oof_cal)
    print(f"  ECE 바닥값(완벽 보정 시) 평균 {floor['ece_floor_mean_pp']:.3f}%p  "
          f"최대 {floor['ece_floor_max_pp']:.3f}%p  ← 판정 기준선")

    print("\n  [Train OOF]")
    m_raw = calibration_metrics(y_tr, pd_oof)
    m_cal = calibration_metrics(y_tr, pd_oof_cal)
    print(f"    보정 전  AUC {m_raw['auc']:.5f}  Brier {m_raw['brier']:.5f}  "
          f"ECE {m_raw['ece_pp']:.3f}%p  MCE {m_raw['mce_pp']:.3f}%p  편향 {m_raw['bias_pp']:+.3f}%p")
    print(f"    보정 후  AUC {m_cal['auc']:.5f}  Brier {m_cal['brier']:.5f}  "
          f"ECE {m_cal['ece_pp']:.3f}%p  MCE {m_cal['mce_pp']:.3f}%p  편향 {m_cal['bias_pp']:+.3f}%p"
          "   ⚠️ in-sample(보정을 학습한 데이터) — 증거로 쓰지 않는다")
    print("    → 보정 효과의 증거는 아래 C-3의 **Validation** 수치다.")

    print("\n  [보정 전 구간별 예측 vs 실측 — 어긋난 방향]")
    print(calibration_table(y_tr, pd_oof).to_string(
        index=False, float_format=lambda v: f"{v: .5f}"))

    bias_bp = m_raw["ece_pp"] * 0.20 * 100
    print("  판정: " + (
        f"ECE {m_raw['ece_pp']:.3f}%p 가 바닥값 {floor['ece_floor_mean_pp']:.3f}%p의 "
        f"{m_raw['ece_pp'] / max(floor['ece_floor_mean_pp'], 1e-9):.1f}배 — 실제로 어긋나 있다. "
        f"(mu_정상−mu_부도)≈20%p를 곱하면 E[XR]에 약 {bias_bp:.0f}bp 편향으로 들어간다."
        if m_raw["ece_pp"] > max(0.5, 2 * floor["ece_floor_mean_pp"])
        else f"ECE {m_raw['ece_pp']:.3f}%p 가 바닥값 {floor['ece_floor_mean_pp']:.3f}%p 수준이다 — "
             f"보정할 어긋남이 사실상 없다. E[XR] 편향은 약 {bias_bp:.0f}bp."
    ))

    # ---- 평탄구간(plateau)이 분위 배정에 무슨 일을 하는가 — 역할 분리의 근거
    #
    # isotonic은 PAV 블록 단위 계단함수라 수십만 개의 PD가 수백 개 값으로 뭉친다. 그래서
    # **보정된 PD로 분위를 자르면 칸 인원이 10%에서 크게 어긋난다** — 경계가 평탄구간
    # 안쪽에 떨어지면 그 구간 전체가 한 칸으로 몰리기 때문이다.
    #
    # 해법은 **역할을 나누는 것**이다. 분위는 칸별 통계를 낼 때 쓰는 **묶는 도구**일 뿐이고,
    # 확률 값이 필요한 곳은 `E[XR] = (1−p̂)·… + p̂·…` 하나다.
    #   → 분위 경계·배정: **보정 전 PD**(순위가 촘촘해 칸이 정확히 10%씩 나뉜다)
    #   → `E[XR]`·`Var[XR]`·`q_score`의 `p̂`: **보정 후 PD**
    # isotonic이 단조라 두 PD의 순위가 (동순위를 빼면) 같으므로 이 분리는 모순이 아니다.
    term_tr = xr["term"].loc[X_tr.index]
    edges = quantile_edges_by_term(pd_oof, term_tr)
    q_tr = assign_quantile_by_term(pd_oof, term_tr, edges)

    edges_cal = quantile_edges_by_term(pd_oof_cal, term_tr)
    q_cal = assign_quantile_by_term(pd_oof_cal, term_tr, edges_cal)

    def worst_imbalance(q: pd.Series) -> float:
        share = (
            pd.DataFrame({"term": term_tr, "q": q})
            .groupby("term", observed=True)["q"]
            .value_counts(normalize=True) * 100
        )
        return float((share - 100 / N_QUANTILES).abs().max())

    moved = int((q_cal != q_tr).sum())
    print(f"\n  [해상도 손실] 고유 PD 값 {pd_oof.nunique():,} → {pd_oof_cal.nunique():,}개 "
          f"(isotonic PAV 블록)")
    print(f"  ρ(pd_raw, pd_cal) Spearman = {spearmanr(pd_oof, pd_oof_cal)[0]:+.6f} "
          f"— 1이 아닌 만큼이 동순위 생성분이다")
    print(f"  분위 배정이 달라지는 건수 {moved:,} / {len(q_tr):,} ({moved / len(q_tr):.3%})")
    print(f"  칸 인원 10%에서의 최대 이탈:  보정 전 PD로 자르면 {worst_imbalance(q_tr):.3f}%p  ·  "
          f"보정 후 PD로 자르면 {worst_imbalance(q_cal):.3f}%p")
    print("  → 판정: " + (
        "보정 후 PD로 자르면 칸이 눈에 띄게 기운다. **분위는 보정 전 PD로 자르고 "
        "확률 값만 보정된 것을 쓴다**(역할 분리) — 위 주석 참고."
        if worst_imbalance(q_cal) > worst_imbalance(q_tr) + 0.5
        else "두 방식의 칸 균형 차이가 작다. 그래도 역할 분리가 더 안전하다 "
             "(평탄구간 폭은 표본·seed에 따라 커질 수 있다)."
    ))
    print("  ⚠️ 따라서 `pd`를 승인선 **점수**로 쓸 때도 보정 전 PD를 쓴다 — 보정은 순위를 "
          "바꾸지 않으면서 동순위만 만들므로 점수로서는 잃을 게 있고 얻을 게 없다.")

    pd.DataFrame([
        {"stage": "train_oof_raw", **m_raw, **floor,
         "n_unique_pd": pd_oof.nunique(), "worst_cell_imbalance_pp": worst_imbalance(q_tr)},
        {"stage": "train_oof_calibrated_INSAMPLE", **m_cal, **floor,
         "n_unique_pd": pd_oof_cal.nunique(), "worst_cell_imbalance_pp": worst_imbalance(q_cal)},
    ]).to_csv(out_dir / "oof_c4_calibration.csv", index=False)

    print("\n  [term별 PD 분위 경계 — 보정 전 PD 기준]")
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

    e_xr = expected_excess_return(pd_oof_cal, xr_tr["xr_normal"], mu_default)
    v_xr = variance_excess_return(
        pd_oof_cal, xr_tr["xr_normal"], mu_default, var_default, var_normal=0.0
    )
    q = q_score(e_xr, v_xr)

    # ---------------------------------------------------------------- C-1
    hdr("3. [C-1] Spearman ρ(pd, q_score) — 설계 존폐")
    ok = q.notna() & pd_oof_cal.notna() & e_xr.notna()
    rho_q, _ = spearmanr(pd_oof_cal[ok], q[ok])
    rho_e, _ = spearmanr(pd_oof_cal[ok], e_xr[ok])
    print(f"  유효 {ok.sum():,}건 / {len(ok):,}    (PD는 **보정 후** 기준 — C-4)")
    print(f"  ρ(pd_cal, q_score) = {rho_q:+.5f}")
    print(f"  ρ(pd_cal, E[XR])   = {rho_e:+.5f}")
    print(f"  ρ(E[XR], q_score)  = {spearmanr(e_xr[ok], q[ok])[0]:+.5f}")
    print(f"\n  [보정 전 PD로 재보기 — 보정이 C-1 결론을 바꿨는지]")
    print(f"  ρ(pd_raw, q_score) = {spearmanr(pd_oof[ok], q[ok])[0]:+.5f}")
    print(f"  ρ(pd_raw, E[XR])   = {spearmanr(pd_oof[ok], e_xr[ok])[0]:+.5f}")

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
        {"term": term_tr, "q": q_tr, "pd": pd_oof_cal, "pd_raw": pd_oof,
         "xr_normal": xr_tr["xr_normal"],
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

    # Train OOF로 학습한 보정을 최종 모델(Train 100%) 예측에 적용한다 — C-4와 같은 이유로
    # 분위 배정 전에 보정해야 한다. 여기서 Validation 보정 품질도 함께 검증된다
    # (Train에서 좋아진 것이 Validation으로 이전되는지).
    p_va_cal = apply_calibrator(calibrator, p_va)
    print("  [Validation 보정 품질]")
    va_rows = []
    for name, p in (("보정 전", p_va), ("보정 후", p_va_cal)):
        m = calibration_metrics(y_va, p)
        print(f"    {name}  AUC {m['auc']:.5f}  Brier {m['brier']:.5f}  "
              f"ECE {m['ece_pp']:.3f}%p  MCE {m['mce_pp']:.3f}%p  편향 {m['bias_pp']:+.3f}%p")
        va_rows.append({"stage": f"validation_{'raw' if name == '보정 전' else 'calibrated'}", **m})

    c4 = pd.read_csv(out_dir / "oof_c4_calibration.csv")
    pd.concat([c4, pd.DataFrame(va_rows)], ignore_index=True).to_csv(
        out_dir / "oof_c4_calibration.csv", index=False)

    # 분위 배정은 **보정 전** PD로 한다 (C-4의 역할 분리). 보정된 `p_va_cal`은
    # threshold 단계에서 `E[XR]`을 만들 때 쓴다.
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
