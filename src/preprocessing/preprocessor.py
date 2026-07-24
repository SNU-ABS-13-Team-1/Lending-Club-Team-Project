"""전처리 파이프라인 뼈대: 행 필터링, 결측치 더미화, Train/Validation/Test 분할.

기존 src/preprocessing/label_pre_post_by_rule.py, unify_variable_labels.py와의 관계
(대체/래핑/병행)는 아직 팀 확정 전이다 — 이 파일을 실제로 구현하기 전에 팀에서 정할 것.
"""

from dataclasses import dataclass

import pandas as pd


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Current/Late 등 분석 대상 제외 행을 필터링한다 (src/preprocessing/AGENTS.md 규칙 참고)."""
    raise NotImplementedError


def add_missing_dummies(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """변수 X마다 결측 더미 D_X와 X_masked = X * D_X를 추가한다."""
    raise NotImplementedError


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> SplitResult:
    """train_test_split을 두 번 적용해 Train/Validation/Test로 분할한다 (src/analysis/AGENTS.md 참고)."""
    raise NotImplementedError
