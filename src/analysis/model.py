"""부도확률(PD) 예측 모형 — XGBoost 단일 · K-fold OOF.

**모형은 XGBoost 하나다** (`decision_log.md` #17 ①, 2026-07-29 회의 확정).
로지스틱 회귀는 폐기됐고, 그에 따라 결측 대체·표준화·더미화도 함께 폐기됐다(#13 ③·#17).

## 왜 OOF(out-of-fold)인가

구조 A′(#20)는 **PD 분위별 칸**에 실현수익률 통계를 채운다. 이때 분위를 만드는 PD가
**같은 데이터로 학습한 모형의 in-sample 예측**이면 과적합된 PD로 칸을 나누게 되어,
칸별 통계가 실제보다 잘 분리된 것처럼 보인다. 그래서 Train 안에서 K-fold를 돌려
**자기 fold를 학습에 쓰지 않은 예측(OOF)** 으로 분위 경계를 만든다.

- PD 분위 경계는 **Train OOF PD**로 만들고 Validation·Test에 **그대로 적용**한다.
  재분위 금지 — 재분위하면 threshold가 "PD 얼마 이하"가 아니라 "그 표본의 상위 몇 %"가
  되어 이전할 수 없다 (`src/analysis/AGENTS.md`).
- fold 모델은 Train의 `(K−1)/K`로 학습하고 최종 모델은 Train 전체로 학습하므로
  **PD 스케일이 미세하게 다르다.** 이 차이가 분위별 인원을 10%에서 밀어내는데,
  얼마나 밀리는지는 `oof_diagnostics.py`의 진단 C-3이 잰다.

## 확률보정(isotonic) — 왜 필요한가

구조 A′는 `p̂`를 **순위가 아니라 확률 값**으로 쓴다.

```
E[XR_i] = (1 − p̂_i) · XR_정상,i + p̂_i · mu_부도,d(i)
```

`p̂`가 0.10인데 실제 부도율이 0.13인 구간이 있으면, 그 오차 0.03이 `(mu_정상 − mu_부도)`
(대략 20%p)를 곱해 **`E[XR]`에 60bp의 편향**으로 그대로 들어간다. `E[XR]` 수준이 1~3%인 것을
감안하면 무시할 크기가 아니다. **AUC로는 이 오류가 전혀 안 잡힌다** — AUC는 순위만 본다.

- 보정은 **Train OOF PD로 학습한다**(`fit_calibrator`). in-sample 예측으로 학습하면 과적합된
  매핑을 배우게 되어 보정 자체가 무의미해진다 — OOF를 만드는 이유와 같다.
- 학습 비용이 **추가로 들지 않는다.** 이미 만든 `pd_oof`가 곧 isotonic의 학습 데이터다.
- ⚠️ **isotonic은 비감소(non-decreasing) 계단함수라 평탄구간(plateau)에서 동순위를 만든다.**
  "단조변환이니 분위 배정이 그대로"는 **정확히는 틀리다** — 수십만 개 PD가 수백 개 값으로
  뭉치고, 분위 경계가 평탄구간 안에 떨어지면 그 구간 전체가 한 칸으로 몰린다.
- 그래서 **PD의 두 역할을 나눈다** (근거는 진단 C-4):
  **분위 경계·배정과 승인선 점수는 보정 전 PD**, **`E[XR]`·`Var[XR]`의 `p̂`만 보정 후 PD**를 쓴다.
  분위는 칸을 묶는 도구라 순위만 필요하고 촘촘한 쪽이 정확히 10%씩 나뉜다. 확률 값이 필요한
  곳은 `E[XR]` 하나다.
- fold 모델(Train 80%)로 만든 보정을 최종 모델(Train 100%) 예측에 적용하므로 PD 스케일이 미세하게
  다르다. 그 차이는 C-3이 이미 쟀다(분위 인원 이탈 최대 0.46%p).

## AUC 사용 범위

성능 보고와 처리 방식 간 상대 비교에는 써도 되지만, **승인/거절 threshold를 정하는 데는
쓰지 않는다** (`src/analysis/AGENTS.md`). threshold는 오직 Sharpe로 정한다.
⚠️ **보정은 AUC를 (거의) 바꾸지 않는다** — 순위 지표라서 그렇다. 보정의 효과는 AUC가 아니라
**Brier·ECE**로 봐야 한다.
⚠️ 기준선은 **확정 표본(#16, 723,563건) 실측 0.70대**다. 문서에 남아 있는 "0.71대"는
#16 만기필터 확정 이전 값이라 인용하지 않는다 — 같은 피처·모델로 필터만 빼면 0.7283이 나와
**−2.1%p가 오롯이 필터 효과**임이 확인됐다(`outputs/reports/oof_diagnostics_kgj.md`).
Lean 0.68은 이미 확정 표본 기준이므로(#13 ⑤), 조건변수 효과는 **0.68 → 0.70**이다.
"""

from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

try:
    from utils.config import load_config
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import load_config


DEFAULT_N_FOLDS = 5


def default_params(seed: int | None = None) -> dict:
    """XGBoost 하이퍼파라미터.

    `enable_categorical=True` + `tree_method="hist"`로 범주형과 NaN을 native 처리한다 —
    원-핫도, 결측 대체도 하지 않는다는 확정(#13 ③·#17)을 코드 수준에서 지키는 부분이다.

    ⚠️ `scale_pos_weight`를 쓰지 않는다. 부도율 16.2%는 극단적 불균형이 아니고, 가중치를
    걸면 예측값이 확률이 아니게 되어 `E[XR] = (1−p̂)·… + p̂·…`에 그대로 쓸 수 없다(#20).
    `p̂`를 순위가 아니라 **확률 값**으로 쓰기 때문이다.
    """
    cfg = load_config()
    return {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "enable_categorical": True,
        "max_cat_to_onehot": 1,  # 항상 분할 기반 처리 — 고카디널리티에서 원-핫 폭발 방지
        "random_state": cfg.random_seed.default if seed is None else seed,
        "n_jobs": -1,
    }


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None, seed: int | None = None
) -> XGBClassifier:
    """PD 모형을 학습한다."""
    model = XGBClassifier(**(params or default_params(seed)))
    model.fit(X_train, y_train, verbose=False)
    return model


def predict_default_probability(model: XGBClassifier, X: pd.DataFrame) -> pd.Series:
    """부도확률 `p̂`를 반환한다 (양성 클래스 = `Charged Off`)."""
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="pd")


def compute_oof(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = DEFAULT_N_FOLDS,
    params: dict | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[pd.Series, list[float], pd.Series]:
    """Train 안에서 K-fold를 돌려 OOF PD를 만든다.

    Returns
    -------
    (pd_oof, fold_auc, fold_index)
        `pd_oof`는 Train과 같은 인덱스·길이를 갖는다.
    """
    cfg = load_config()
    seed = cfg.random_seed.default if seed is None else seed

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pd_oof = pd.Series(np.nan, index=X_train.index, name="pd_oof")
    fold_index = pd.Series(-1, index=X_train.index, name="fold", dtype="int8")
    fold_auc: list[float] = []

    for k, (tr, va) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr], y_train.iloc[tr]
        X_va, y_va = X_train.iloc[va], y_train.iloc[va]

        model = train_model(X_tr, y_tr, params=params, seed=seed)
        p = predict_default_probability(model, X_va)

        pd_oof.iloc[va] = p.to_numpy()
        fold_index.iloc[va] = k
        auc = roc_auc_score(y_va, p)
        fold_auc.append(auc)
        if verbose:
            print(f"  fold {k + 1}/{n_folds}  n_va={len(va):>7,}  AUC={auc:.5f}")

    if pd_oof.isna().any():
        raise RuntimeError("OOF 예측에 결측이 남았습니다 — fold 분할을 확인하세요.")

    return pd_oof, fold_auc, fold_index


# ---------------------------------------------------------------------------
# 확률보정 (isotonic)
# ---------------------------------------------------------------------------
def fit_calibrator(pd_oof: pd.Series, y_train: pd.Series) -> IsotonicRegression:
    """Train **OOF** PD로 isotonic 보정을 학습한다.

    `E[XR]`이 `p̂`를 확률 값으로 쓰므로 보정 오차가 `E[XR]`에 직접 들어간다(모듈 docstring).
    OOF를 쓰는 것이 요점이다 — in-sample 예측으로 맞추면 과적합된 매핑을 배운다.

    `out_of_bounds="clip"`은 Validation·Test에서 Train OOF 범위를 벗어난 PD가 나올 때
    양 끝값으로 자른다. 부도확률이므로 `[0, 1]`을 벗어나서는 안 된다.

    ⚠️ Platt(로지스틱) 대신 isotonic을 쓰는 이유: 표본이 43만 건(Train)이라 isotonic의
    과적합 위험이 낮고, Platt은 시그모이드 모양을 강제해 XGBoost의 왜곡을 다 못 펴기 때문이다.
    """
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(
        pd_oof.to_numpy(dtype="float64"), y_train.to_numpy(dtype="float64")
    )


def apply_calibrator(calibrator: IsotonicRegression, p_hat: pd.Series) -> pd.Series:
    """보정된 부도확률. 인덱스를 보존한다."""
    return pd.Series(
        calibrator.predict(p_hat.to_numpy(dtype="float64")),
        index=p_hat.index,
        name="pd_calibrated",
    )


def calibration_table(y: pd.Series, p_hat: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """구간별 예측확률 평균 vs 실제 부도율.

    **동일 폭이 아니라 동일 인원(분위) 구간**으로 자른다 — PD 분포가 오른쪽으로 길게
    치우쳐 있어 동일 폭으로 자르면 상위 구간이 거의 비고, 파이프라인이 쓰는 PD 10분위와도
    기준이 어긋난다.
    """
    edges = np.unique(np.quantile(p_hat.to_numpy(dtype="float64"), np.linspace(0, 1, n_bins + 1)))
    bins = pd.cut(p_hat, bins=edges, include_lowest=True, duplicates="drop")

    frame = pd.DataFrame({"y": y, "p": p_hat, "bin": bins})
    g = frame.groupby("bin", observed=True)
    out = pd.DataFrame(
        {"n": g.size(), "mean_pred": g["p"].mean(), "observed": g["y"].mean()}
    )
    out["gap_pp"] = (out["mean_pred"] - out["observed"]) * 100.0
    return out.reset_index(drop=True).assign(bin=lambda d: d.index + 1)


def calibration_metrics(y: pd.Series, p_hat: pd.Series, n_bins: int = 10) -> dict:
    """보정 품질 지표. **AUC가 아니라 이 값들로 보정 효과를 판단한다.**

    - `brier`: `mean((p̂ − y)²)`. 판별력과 보정을 함께 담는다(낮을수록 좋다).
    - `ece`: 기대보정오차 `Σ (n_b/N)·|예측평균_b − 실측률_b|`. 구간별 어긋남의 가중평균.
    - `mce`: 최대보정오차 — 최악 구간의 어긋남.
    - `bias_pp`: 전체 예측평균 − 실제 부도율. 방향(과대/과소)을 본다.
    """
    tbl = calibration_table(y, p_hat, n_bins=n_bins)
    w = tbl["n"] / tbl["n"].sum()
    gap = (tbl["mean_pred"] - tbl["observed"]).abs()
    return {
        "auc": float(roc_auc_score(y, p_hat)),
        "brier": float(np.mean((p_hat.to_numpy(dtype="float64") - y.to_numpy(dtype="float64")) ** 2)),
        "ece_pp": float((w * gap).sum() * 100.0),
        "mce_pp": float(gap.max() * 100.0),
        "bias_pp": float((p_hat.mean() - y.mean()) * 100.0),
    }


def calibration_noise_floor(
    p_hat: pd.Series, n_sim: int = 20, n_bins: int = 10, seed: int | None = None
) -> dict:
    """**완벽히 보정된 모형이라도 나오는 ECE** — 판정 기준선.

    ECE는 유한표본 잡음 때문에 0이 되지 않는다. `y ~ Bernoulli(p̂)`를 직접 생성해 재보면
    "이 표본 크기에서 보정이 완벽할 때의 ECE"가 나온다. 실측 ECE가 이 값 수준이면
    **보정할 것이 없다는 뜻**이고, 몇 배 크면 실제로 어긋난 것이다.

    합성 실험에서 n=50,000·10구간의 바닥값이 0.32%p였다 — 표본이 작으면 판정 기준
    0.5%p가 바닥값과 구분되지 않으므로 이 함수로 함께 보고한다.
    """
    rng = np.random.default_rng(load_config().random_seed.default if seed is None else seed)
    p = p_hat.to_numpy(dtype="float64")
    eces = [
        calibration_metrics(pd.Series(rng.binomial(1, p), index=p_hat.index), p_hat, n_bins)["ece_pp"]
        for _ in range(n_sim)
    ]
    return {"ece_floor_mean_pp": float(np.mean(eces)), "ece_floor_max_pp": float(np.max(eces))}


def assign_pd_quantile(
    pd_values: pd.Series,
    edges: np.ndarray,
) -> pd.Series:
    """PD 값을 **주어진 경계**로 분위에 배정한다 (1 = 최저 PD).

    경계를 인자로 받는 것이 요점이다 — Validation·Test에서 다시 분위를 만들지 않고
    Train OOF 경계를 그대로 적용한다(`src/analysis/AGENTS.md` 재분위 금지).
    """
    q = np.digitize(pd_values.to_numpy(), edges, right=True) + 1
    q = np.clip(q, 1, len(edges) + 1)
    return pd.Series(q, index=pd_values.index, name="pd_quantile", dtype="int8")


def make_quantile_edges(pd_train_oof: pd.Series, n_quantiles: int = 10) -> np.ndarray:
    """Train OOF PD로 분위 경계를 만든다. 내부 경계만 돌려준다(길이 `n_quantiles − 1`)."""
    qs = np.linspace(0, 1, n_quantiles + 1)[1:-1]
    return np.quantile(pd_train_oof.to_numpy(), qs)


def quantile_edges_by_term(
    pd_oof: pd.Series, term: pd.Series, n_quantiles: int = 10
) -> dict[float, np.ndarray]:
    """term별로 **따로** PD 분위 경계를 만든다 (`decision_log.md` #20 권고).

    전체 공통 분위로 자르면 **60개월·저PD 칸이 거의 빈다** — 60m는 위험군이라 고PD 구간에
    쏠리기 때문이다. #16으로 60개월 비중이 25.1% → 13.9%로 줄어 이 쏠림이 더 심해졌다.

    ⚠️ 입력은 **보정 전** OOF PD다 — isotonic 평탄구간이 경계를 삼켜 칸 인원이 기운다
    (진단 C-4). 보정된 PD는 `E[XR]`의 `p̂`로만 쓴다.
    """
    return {
        float(t): make_quantile_edges(pd_oof[term == t], n_quantiles)
        for t in sorted(term.dropna().unique())
    }


def assign_quantile_by_term(
    pd_values: pd.Series, term: pd.Series, edges_by_term: dict[float, np.ndarray]
) -> pd.Series:
    """term별 경계로 분위를 배정한다. Validation·Test에서 **재분위하지 않는다**."""
    out = pd.Series(np.nan, index=pd_values.index, name="pd_quantile")
    for t, edges in edges_by_term.items():
        mask = term == t
        if mask.any():
            out.loc[mask] = assign_pd_quantile(pd_values.loc[mask], edges).to_numpy()
    return out.astype("Int8")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.preprocessor import build_feature_table, split_from_manifest

    print("피처 테이블 구성 중...")
    X, y, meta = build_feature_table()
    parts = split_from_manifest(X, y, meta)
    X_tr, y_tr, _ = parts["train"]
    X_va, y_va, _ = parts["validation"]
    print(f"  Train {len(X_tr):,} / Validation {len(X_va):,}  피처 {X.shape[1]}개")

    print(f"\n[Train {DEFAULT_N_FOLDS}-fold OOF]")
    pd_oof, fold_auc, _ = compute_oof(X_tr, y_tr)
    print(f"  fold AUC 평균 {np.mean(fold_auc):.5f} (표준편차 {np.std(fold_auc):.5f})")
    print(f"  OOF 전체 AUC {roc_auc_score(y_tr, pd_oof):.5f}")

    print("\n[최종 모델 — Train 전체 학습 → Validation 평가]")
    final = train_model(X_tr, y_tr)
    p_va = predict_default_probability(final, X_va)
    print(f"  Validation AUC {roc_auc_score(y_va, p_va):.5f}   (확정 표본 기준 0.70대 — oof_diagnostics_kgj.md)")

    print("\n[확률보정 — isotonic, Train OOF로 학습 → Validation 적용]")
    calibrator = fit_calibrator(pd_oof, y_tr)
    p_va_cal = apply_calibrator(calibrator, p_va)
    for name, p in (("보정 전", p_va), ("보정 후", p_va_cal)):
        m = calibration_metrics(y_va, p)
        print(f"  {name}  AUC {m['auc']:.5f}  Brier {m['brier']:.5f}  "
              f"ECE {m['ece_pp']:.3f}%p  MCE {m['mce_pp']:.3f}%p  편향 {m['bias_pp']:+.3f}%p")
