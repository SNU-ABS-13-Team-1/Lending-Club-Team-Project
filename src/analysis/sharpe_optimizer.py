"""Validation set 기준 Sharpe Ratio 극대화 threshold 탐색 뼈대 (src/analysis/AGENTS.md 정의 참고)."""

import pandas as pd


def realized_return(
    loan_status: pd.Series,
    int_rate: pd.Series,
    approved: pd.Series,
    risk_free_rate: float,
) -> pd.Series:
    """승인/거절·부도 여부에 따른 실현수익률을 계산한다."""
    raise NotImplementedError


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """(평균 - Rf) / 표본표준편차(ddof=1)."""
    raise NotImplementedError


def find_optimal_threshold(
    default_proba: pd.Series,
    loan_status: pd.Series,
    int_rate: pd.Series,
    risk_free_rate: float,
) -> float:
    """Sharpe Ratio를 극대화하는 threshold tau*를 탐색한다."""
    raise NotImplementedError
