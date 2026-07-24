"""부도확률 예측 모형 학습/추론 뼈대.

후보 모델은 outputs/reports/decision_log.md 기준 아직 확정 전(로지스틱 + GBM 병행 검토 중,
SVM 등 다른 후보도 배제되지 않음)이므로 특정 모델을 하드코딩하지 않는다.
"""

from typing import Protocol

import pandas as pd


class DefaultProbabilityModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DefaultProbabilityModel": ...

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """부도확률 p를 반환한다."""
        ...


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DefaultProbabilityModel:
    raise NotImplementedError


def predict_default_probability(model: DefaultProbabilityModel, X: pd.DataFrame) -> pd.Series:
    raise NotImplementedError
