"""차트 생성 뼈대 — outputs/figures/에 저장.

색상/스타일/파일명/해상도 등 세부 규칙은 src/viz/AGENTS.md 미확정.
"""

from pathlib import Path

import pandas as pd


def plot_sharpe_by_threshold(thresholds: pd.Series, sharpe_values: pd.Series, save_path: Path) -> None:
    raise NotImplementedError


def plot_return_distribution(returns: pd.Series, save_path: Path) -> None:
    """포트폴리오 수익률 분포 히스토그램 (좌측 꼬리 확인용, README.md 참고)."""
    raise NotImplementedError
