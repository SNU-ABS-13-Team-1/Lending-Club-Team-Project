"""Validation set 기준 Sharpe Ratio 극대화 threshold 탐색.

정의는 `src/analysis/AGENTS.md`, 확정 근거는 `outputs/reports/decision_log.md` #18·#20.

## 이 모듈이 하지 않는 것

- **실현수익률을 다시 정의하지 않는다.** 구조 A′의 단일 구현은 `src/analysis/realized_return.py`다
  (`contract_return()` · `realized_return_defaulted()` · `expected_excess_return()` · `q_score()`).
  여기서는 거기서 나온 건별 `XR`과 랭킹 점수를 **받아서 정렬·탐색만** 한다.
- **무위험수익률을 스칼라로 받지 않는다.** `rf`는 발행시점(`issue_d`) × 만기(`term`) 매칭 국채라
  건별로 다르다(#18). `config.yaml`의 `risk_free_rate.value`는 계속 `null`이며 읽지 않는다.
  이미 `XR = R − rf`로 차감된 초과수익률을 입력으로 받는다.
- **`int_rate`를 수익률로 쓰지 않는다.** 폐기된 관례다(#18) — `int_rate`는 피처이자 계약
  현금흐름의 입력일 뿐이다.

## 무엇으로 Sharpe를 계산하는가 — 기대값이 아니라 실현값

점수(`E[XR]`·`q_score`)는 **승인선을 그을 때만** 쓴다. Sharpe 자체는 Validation에서 실제로
벌어진 **실현 `XR`**(`realized_return.build_excess_returns()`의 `xr_realized`)로 계산한다.
기대값으로 Sharpe를 재면 모형이 자기 예측으로 자기를 채점하는 셈이다.

## 승인선 랭킹 기준 — 세 후보를 한 번에 비교한다

`pd` / `E[XR]` / `q_score` 중 무엇으로 승인선을 그을지는 **미확정**이다(#5·#20).
세 기준은 같은 중간 테이블에서 **정렬만 바꾸면 나오므로 추가 학습이 필요 없다.**
→ Validation에서만 비교해 승자를 사전 확정한 뒤 **Test는 1회**다. 세 기준을 Test에서
비교하면 "Test set으로 모형을 재조정하지 않는다" 규칙 위반이다.

⚠️ `pd`를 점수로 쓸 때는 **보정 전** PD를 쓴다 — isotonic 보정은 순위를 바꾸지 않으면서
동순위만 만들어(고유값 42.8만 → 115개), 점수로서는 잃을 게 있고 얻을 게 없다(진단 C-4).
보정된 PD는 `E[XR]`·`Var[XR]`의 `p̂`로만 들어간다.

## 절대 Sharpe와 Δ Sharpe를 **함께** 보고한다

헤드라인은 **Δ Sharpe = (모형) − (approve-all 대조군)** 이다(#17 ②·#18) — 재투자 가정이
양쪽을 같이 밀어올리므로 절대값에는 가정 효과가 섞인다.

Δ를 헤드라인으로 쓰는 것이 **최적화 목표를 바꾸지는 않는다.** approve-all Sharpe는 `τ`에
대해 상수라 `argmax_τ [Sharpe(τ) − 상수] = argmax_τ Sharpe(τ)` 이기 때문이다. 프로젝트 목적
(Sharpe 극대화)은 그대로다.

그럼에도 **절대 Sharpe를 반드시 병기한다.** 이유가 둘이다.

1. 수업에서 제시된 기준선이 절대 수준이다(`README.md`: "샤프비율은 0.2~0.4 정도 나오면
   옳게 한 것"). Δ만 내면 그 기준과 대조할 숫자가 보고서에 없다.
2. #18은 "가정이 두 전략을 똑같이 밀어올린다"고 전제했지만 **+107.5bp는 균일한 평행이동이
   아니다** — `int_rate`·`term`별로 다르게 들어가 건별 `XR`의 순서·분산을 바꾸므로 `τ*`
   자체가 움직일 수 있다. 그래서 **재투자 가정 2종(국채/0%)을 모두 돌려** Δ가 가정에
   안정적인지 실측한다(#18이 남긴 "검증 과제").

따라서 산출 표는 **(랭킹 기준 3종 + approve-all) × (재투자 가정 2종)** 이며 열은
`τ*` · 승인율 · 절대 Sharpe · Δ Sharpe다.

실행: `/opt/anaconda3/bin/python src/analysis/sharpe_optimizer.py`
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from utils.config import load_config, repo_root
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import load_config, repo_root


# ---------------------------------------------------------------------------
# Sharpe 계산
# ---------------------------------------------------------------------------
def portfolio_excess_returns(excess_returns: pd.Series, approved: pd.Series) -> pd.Series:
    """포트폴리오 건별 초과수익률.

    승인 건은 실현 `XR`을 그대로, **거절 건은 무위험자산에 투자했다고 보아 `XR = 0`** 이다
    (`R = rf` → `XR = R − rf = 0`). 거절 건을 표본에서 빼지 않는다 —
    빼면 승인율이 낮을수록 분모가 줄어 Sharpe가 부풀려진다.
    """
    return excess_returns.where(approved.astype(bool), 0.0)


def sharpe_ratio(portfolio_excess_returns: pd.Series) -> float:
    """`평균(XR) / 표본표준편차(XR, ddof=1)`.

    입력이 **이미 초과수익률**이므로 여기서 다시 `rf`를 빼지 않는다(#18).
    """
    x = portfolio_excess_returns.to_numpy(dtype="float64")
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def evaluate_threshold(
    score: pd.Series,
    excess_returns: pd.Series,
    threshold: float,
    lower_is_better: bool = True,
) -> dict:
    """threshold 하나를 평가한다.

    `score`는 랭킹 기준(`pd` / `E[XR]` / `q_score`) 중 **하나**다 — 어느 것을 쓸지는
    미확정이므로 주입받는다(#5·#20). `lower_is_better`는 `pd`면 `True`,
    `E[XR]`·`q_score`면 `False`다.

    반환에는 **승인율을 반드시 포함한다** — Sharpe만 보고 조이면 승인율이 비현실적으로
    낮아질 수 있다(`src/analysis/AGENTS.md`).
    """
    approved = score <= threshold if lower_is_better else score >= threshold
    pxr = portfolio_excess_returns(excess_returns, approved)
    return {
        "threshold": float(threshold),
        "n_total": int(len(score)),
        "n_approved": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "sharpe": sharpe_ratio(pxr),
        "mean_xr": float(pxr.mean()),
        "sd_xr": float(pxr.std(ddof=1)),
        "mean_xr_approved": (
            float(excess_returns[approved].mean()) if approved.any() else float("nan")
        ),
    }


def sharpe_curve(
    score: pd.Series, excess_returns: pd.Series, lower_is_better: bool = True
) -> pd.DataFrame:
    """**모든** 승인 컷의 Sharpe를 한 번에 계산한다 (누적합, `O(n log n)`).

    격자를 임의로 찍지 않는다 — 점수로 정렬해 상위 `k`건을 승인하는 경우를 `k=1..n` 전부
    훑으므로 **격자 해상도 때문에 최적점을 놓치는 일이 없다.**

    거절 건의 `XR`이 0이라는 점을 쓰면 누적합만으로 닫힌 형태가 나온다. 승인 `k`건의 `XR`
    합을 `S₁`, 제곱합을 `S₂`라 하면 (분모는 거절 건을 포함한 전체 `n`건이다)

        평균 = S₁ / n
        Σ(xᵢ − 평균)² = S₂ − S₁²/n        ← 거절 건은 두 합에 0으로 기여한다
        표본분산 = (S₂ − S₁²/n) / (n − 1)

    ⚠️ 반환 `threshold`는 그 컷의 **경계 점수값**이다. 동순위가 있으면 `score <= threshold`로
    자른 실제 승인 건수가 `k`보다 많을 수 있어, `find_optimal_threshold()`는 최적 `k`를 찾은
    뒤 **경계값으로 다시 평가**해 실제 승인율을 보고한다.
    """
    s = score.to_numpy(dtype="float64")
    x = excess_returns.to_numpy(dtype="float64")
    n = len(s)
    if n < 2:
        raise ValueError(f"표본이 너무 작다: n={n}")

    order = np.argsort(s, kind="mergesort")
    if not lower_is_better:
        order = order[::-1]
    xs, ss = x[order], s[order]

    s1 = np.cumsum(xs)
    s2 = np.cumsum(xs**2)

    mean = s1 / n
    var = (s2 - s1**2 / n) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(var > 0, mean / np.sqrt(var), np.nan)

    return pd.DataFrame(
        {
            "k": np.arange(1, n + 1),
            "threshold": ss,
            "approval_rate": np.arange(1, n + 1) / n,
            "mean_xr": mean,
            "sd_xr": np.sqrt(np.maximum(var, 0.0)),
            "sharpe": sharpe,
        }
    )


def approve_all_sharpe(excess_returns: pd.Series) -> float:
    """대조군 — 전부 승인(#17 ②). threshold와 무관한 **상수**다.

    그래서 `argmax_τ [Sharpe(τ) − 이 값] = argmax_τ Sharpe(τ)` 이며, Δ Sharpe를 헤드라인으로
    쓰는 것은 **최적화 목표를 바꾸지 않는다** — 보고·해석의 문제일 뿐이다.
    """
    return sharpe_ratio(excess_returns)


def find_optimal_threshold(
    score: pd.Series,
    excess_returns: pd.Series,
    lower_is_better: bool = True,
) -> dict:
    """Validation 실현 `XR`로 계산한 **실제 Sharpe**를 최대화하는 `τ*`를 찾는다.

    점수의 이론값을 최대화하지 않는다 — 개별 대출 `q_score` 최대화는 포트폴리오 Sharpe
    최대화와 같은 문제가 아니다(`src/analysis/AGENTS.md`).

    절대 Sharpe와 **Δ Sharpe = (모형) − (approve-all)** 을 함께 돌려준다(#17 ②·#18).
    """
    curve = sharpe_curve(score, excess_returns, lower_is_better)
    if curve["sharpe"].isna().all():
        raise RuntimeError("Sharpe가 전부 NaN이다 — XR 분산이 0인지 확인하라.")

    best = curve.loc[curve["sharpe"].idxmax()]
    # 동순위를 포함한 실제 승인 집합으로 다시 평가한다
    out = evaluate_threshold(score, excess_returns, best["threshold"], lower_is_better)
    baseline = approve_all_sharpe(excess_returns)
    out["sharpe_approve_all"] = baseline
    out["delta_sharpe"] = out["sharpe"] - baseline
    out["k_at_optimum"] = int(best["k"])
    return out


def compare_ranking_criteria(
    scores: dict[str, tuple[pd.Series, bool]],
    excess_returns: pd.Series,
) -> pd.DataFrame:
    """랭킹 기준 여러 개를 **같은 표본·같은 실현 XR**로 비교한다 (#5·#20).

    `scores`는 `{이름: (점수 시리즈, lower_is_better)}`다. 추가 학습이 없다 — 정렬만 바꾼다.

    ⚠️ 비교가 성립하려면 **세 기준이 모두 유효한 행만** 써야 한다. `q_score`는 분산이 0인
    칸에서 NaN이 되므로, 기준마다 표본이 다르면 Sharpe 차이가 기준 차이인지 표본 차이인지
    구분되지 않는다. 공통 유효 마스크는 호출자가 맞춰서 넣는다
    (`scores_for_assumptions()`가 그 일을 한다).
    """
    rows = [
        {"criterion": name, **find_optimal_threshold(s, excess_returns, lower)}
        for name, (s, lower) in scores.items()
    ]
    baseline = approve_all_sharpe(excess_returns)
    rows.append(
        {
            "criterion": "approve_all(대조군)",
            "threshold": float("nan"),
            "n_total": len(excess_returns),
            "n_approved": len(excess_returns),
            "approval_rate": 1.0,
            "sharpe": baseline,
            "mean_xr": float(excess_returns.mean()),
            "sd_xr": float(excess_returns.std(ddof=1)),
            "mean_xr_approved": float(excess_returns.mean()),
            "sharpe_approve_all": baseline,
            "delta_sharpe": 0.0,
            "k_at_optimum": len(excess_returns),
        }
    )
    return pd.DataFrame(rows)


def repeat_threshold_search(
    score_fn: Callable[[int], tuple[pd.Series, pd.Series, bool]],
    n_repeats: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """랜덤 6:2:2 분할을 **K=50회** 반복해 `τ*`의 분포(평균·표준편차)를 본다 (#18).

    Validation 표본 하나로 잰 Sharpe는 "진짜 성과"가 아니라 표본에 따라 흔들리는 추정치다 —
    대출 수익률이 부도 여부(베르누이)로 결정되므로 누가 Validation에 뽑혔는지에 따라 평균·
    표준편차가 크게 달라진다.

    최종 threshold는 **반복에서 나온 `τ*` 값 자체의 평균/중앙값**을 쓴다. "평균 Sharpe와
    비슷한 결과를 낸 threshold 하나를 고르는" 방식보다 안정적이다 — `τ` → Sharpe 매핑이
    1:1이 아니기 때문이다(`README.md`). 분포가 비대칭이면 중앙값이 낫다.

    `score_fn(seed)`는 seed로 재분할·재학습해 `(점수, 실현 XR, lower_is_better)`를 돌려준다.
    ⚠️ **비용이 크다** — seed마다 5-fold OOF + 최종 모델이라 K=50이면 300회 학습이다.
    칸별 통계표를 seed마다 다시 만들지 seed 1개로 고정할지는 아직 미확정이다(#20 구현경로 2).
    """
    rows = []
    for seed in range(n_repeats):
        score, xr, lower = score_fn(seed)
        res = find_optimal_threshold(score, xr, lower)
        rows.append({"seed": seed, **res})
        if verbose:
            print(f"  seed {seed:>3}  τ*={res['threshold']:.6f}  "
                  f"승인율 {res['approval_rate']:.3%}  "
                  f"Sharpe {res['sharpe']:.4f}  Δ {res['delta_sharpe']:+.4f}")
    return pd.DataFrame(rows)


def summarize_repeats(repeats: pd.DataFrame) -> pd.DataFrame:
    """반복 결과 요약 — `τ*`는 **중앙값**을 우선 본다(분포가 비대칭이다)."""
    cols = ["threshold", "approval_rate", "sharpe", "sharpe_approve_all", "delta_sharpe"]
    return repeats[cols].agg(["mean", "std", "median", "min", "max"]).T


# ---------------------------------------------------------------------------
# 파이프라인 조립
# ---------------------------------------------------------------------------
def build_validation_scores(seed: int | None = None, verbose: bool = True) -> dict:
    """Train으로 학습·보정하고, Validation 점수를 만들 재료를 모은다.

    **모형은 재투자 가정과 무관하다** — 타깃이 이진 `loan_status`라 수익률 정의가 학습에
    개입하지 않는다(#19). 그래서 여기서 한 번만 학습하고, 가정별 계산은
    `scores_for_assumptions()`가 맡는다.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.model import (
        apply_calibrator,
        assign_quantile_by_term,
        compute_oof,
        fit_calibrator,
        predict_default_probability,
        quantile_edges_by_term,
        train_model,
    )
    from analysis.realized_return import build_return_inputs
    from preprocessing.preprocessor import (
        build_feature_table,
        resplit_train_validation,
        split_from_manifest,
    )

    cfg = load_config()

    X, y, meta = build_feature_table()
    if seed is None:
        # 기준 실행 — 매니페스트 분할을 그대로 쓴다. Test는 잠긴 채로 남는다.
        parts = split_from_manifest(X, y, meta)
        seed = cfg.random_seed.default
    else:
        # K=50 반복 — **Test를 고정한 채** Train+Validation 풀만 재분할한다(#18).
        parts = resplit_train_validation(X, y, meta, seed=seed)
    X_tr, y_tr, _ = parts["train"]
    X_va, y_va, _ = parts["validation"]
    if verbose:
        print(f"  표본 {len(X):,}  피처 {X.shape[1]}  "
              f"Train {len(X_tr):,}  Validation {len(X_va):,}")

    pd_oof, fold_auc, _ = compute_oof(X_tr, y_tr, seed=seed, verbose=verbose)
    calibrator = fit_calibrator(pd_oof, y_tr)

    final = train_model(X_tr, y_tr, seed=seed)
    p_va_raw = predict_default_probability(final, X_va)
    p_va_cal = apply_calibrator(calibrator, p_va_raw)

    return {
        "pd_oof_raw": pd_oof,
        "p_va_raw": p_va_raw,
        "p_va_cal": p_va_cal,
        "train_index": X_tr.index,
        "validation_index": X_va.index,
        "outcome": build_return_inputs(),
        "fold_auc": fold_auc,
        "edges_fn": quantile_edges_by_term,
        "assign_fn": assign_quantile_by_term,
    }


def scores_for_assumptions(bundle: dict, assumptions) -> tuple[dict, pd.Series, dict]:
    """가정 하나에 대해 Validation 점수 3종 + 실현 `XR`을 만든다.

    반환 `(scores, realized_xr, audit)`. `scores`는 `compare_ranking_criteria()`에 그대로
    넣는 형태이며, 세 기준 모두 유효한 **공통 마스크**가 이미 적용돼 있다.

    ⚠️ 칸별 통계표(`mu_부도`·`var_부도`)는 **Train에서만** 만들고 Validation에 적용한다.
    Validation 결과를 보고 만들면 누수다. 분위 경계도 재분위하지 않는다(#20).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.realized_return import (
        build_excess_returns,
        default_cell_stats,
        expected_excess_return,
        q_score,
        variance_excess_return,
    )

    xr = build_excess_returns(bundle["outcome"], assumptions)
    tr_idx, va_idx = bundle["train_index"], bundle["validation_index"]
    term_tr, term_va = xr["term"].loc[tr_idx], xr["term"].loc[va_idx]

    # 분위 경계·배정은 **보정 전** PD로 (진단 C-4의 역할 분리)
    edges = bundle["edges_fn"](bundle["pd_oof_raw"], term_tr)
    q_tr = bundle["assign_fn"](bundle["pd_oof_raw"], term_tr, edges)
    q_va = bundle["assign_fn"](bundle["p_va_raw"], term_va, edges)

    stats = default_cell_stats(xr["xr_default"].loc[tr_idx], q_tr, term_tr)
    mu_map, var_map = stats["mu"].to_dict(), stats["var"].to_dict()
    keys = list(zip(term_va, q_va))
    mu_default = pd.Series([mu_map.get(k, np.nan) for k in keys], index=va_idx)
    var_default = pd.Series([var_map.get(k, np.nan) for k in keys], index=va_idx)

    # E[XR]·Var[XR]의 p̂는 **보정 후** PD (진단 C-4)
    xr_va = xr.loc[va_idx]
    e_xr = expected_excess_return(bundle["p_va_cal"], xr_va["xr_normal"], mu_default)
    v_xr = variance_excess_return(
        bundle["p_va_cal"], xr_va["xr_normal"], mu_default, var_default, var_normal=0.0
    )
    qs = q_score(e_xr, v_xr)
    realized = xr_va["xr_realized"]

    ok = realized.notna() & e_xr.notna() & qs.notna() & bundle["p_va_raw"].notna()
    audit = {
        "n_validation": int(len(va_idx)),
        "n_usable": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "nan_realized_xr": int(realized.isna().sum()),
        "nan_e_xr": int(e_xr.isna().sum()),
        "nan_q_score": int(qs.isna().sum()),
    }

    scores = {
        "pd (보정 전)": (bundle["p_va_raw"][ok], True),
        "E[XR]": (e_xr[ok], False),
        "q_score": (qs[ok], False),
    }
    return scores, realized[ok], audit


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.realized_return import ReturnAssumptions, cash_reinvestment

    out_dir = repo_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Sharpe threshold 탐색 — 랭킹 기준 3종 × 재투자 가정 2종")
    print("=" * 78)
    print("⚠️ 모든 XR은 **잠정값** 기준이다 — 국채 ⓒ발행시점 고정 · 수수료 0% · 조기상환 보정 0.")
    print("   조기상환 보정은 정상상환분 R을 낮추므로 절대 Sharpe는 과대추정이다(#20 B팀 1순위).")

    print("\n[모형 학습 — 재투자 가정과 무관하므로 1회만 (#19)]")
    bundle = build_validation_scores()
    print(f"  fold AUC 평균 {np.mean(bundle['fold_auc']):.5f}")

    treasury = ReturnAssumptions()
    frames = []
    for assumptions in (treasury, cash_reinvestment(treasury)):
        print("\n" + "=" * 78)
        print(f"[재투자 가정] {assumptions.label()}")
        print("=" * 78)

        scores, realized, audit = scores_for_assumptions(bundle, assumptions)
        print(f"  유효 {audit['n_usable']:,} / {audit['n_validation']:,}건  "
              f"(제외 {audit['n_dropped']:,} — 실현XR NaN {audit['nan_realized_xr']:,} · "
              f"E[XR] NaN {audit['nan_e_xr']:,} · q_score NaN {audit['nan_q_score']:,})")
        print(f"  실현 XR 평균 {realized.mean():+.4%}  sd {realized.std(ddof=1):.4%}")

        table = compare_ranking_criteria(scores, realized)
        table.insert(0, "assumptions", assumptions.label())
        table.insert(1, "reinvest", assumptions.reinvest)
        frames.append(table)

        show = table[["criterion", "threshold", "approval_rate", "sharpe",
                      "sharpe_approve_all", "delta_sharpe", "mean_xr", "sd_xr"]]
        print("\n" + show.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    result = pd.concat(frames, ignore_index=True)
    path = out_dir / "sharpe_threshold_comparison.csv"
    result.to_csv(path, index=False)

    print("\n" + "=" * 78)
    print("Δ Sharpe의 재투자 가정 안정성 (#18이 남긴 검증 과제)")
    print("=" * 78)
    pivot = result.pivot_table(
        index="criterion", columns="reinvest", values=["sharpe", "delta_sharpe", "approval_rate"]
    )
    print(pivot.to_string(float_format=lambda v: f"{v: .5f}"))
    print("\n  Δ가 두 가정에서 비슷하면 #18의 전제('가정이 양쪽을 똑같이 밀어올린다')가 성립한다.")
    print("  크게 다르면 Δ도 가정에 의존하므로 헤드라인 근거를 다시 세워야 한다.")

    print(f"\n산출물 → {path.relative_to(repo_root())}")
    print("⚠️ 랭킹 기준 확정은 **팀 결정**이다 — 이 표는 근거를 제공한다(#5·#20).")
    print("   Test는 승자를 확정한 뒤 1회만 적용한다.")


if __name__ == "__main__":
    main()
