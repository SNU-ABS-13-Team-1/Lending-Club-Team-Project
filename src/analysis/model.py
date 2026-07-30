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

## AUC 사용 범위

성능 보고와 처리 방식 간 상대 비교에는 써도 되지만, **승인/거절 threshold를 정하는 데는
쓰지 않는다** (`src/analysis/AGENTS.md`). threshold는 오직 Sharpe로 정한다.
⚠️ 기준선은 **확정 표본(#16, 723,563건) 실측 0.70대**다. 문서에 남아 있는 "0.71대"는
#16 만기필터 확정 이전 값이라 인용하지 않는다 — 같은 피처·모델로 필터만 빼면 0.7283이 나와
**−2.1%p가 오롯이 필터 효과**임이 확인됐다(`outputs/reports/oof_diagnostics_kgj.md`).
Lean 0.68은 이미 확정 표본 기준이므로(#13 ⑤), 조건변수 효과는 **0.68 → 0.70**이다.
"""

from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
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


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.preprocessor import build_feature_table, split_6_2_2

    print("피처 테이블 구성 중...")
    X, y, meta = build_feature_table()
    parts = split_6_2_2(X, y, meta)
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
