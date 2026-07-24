"""Raw CSV/데이터 사전 로딩 및 기본 검증 뼈대.

기존 src/preprocessing/label_pre_post_by_rule.py, unify_variable_labels.py와의 관계
(대체/래핑/병행)는 아직 팀 확정 전이다 — 이 파일을 실제로 구현하기 전에 팀에서 정할 것.
"""

from pathlib import Path

import pandas as pd


def load_raw_loans(csv_path: Path) -> pd.DataFrame:
    """data/raw의 원본 대출 CSV를 로드한다."""
    raise NotImplementedError


def load_data_dictionary(xlsx_path: Path) -> pd.DataFrame:
    """공식 데이터 사전(Data Dictionary) Excel을 로드한다."""
    raise NotImplementedError


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """필수 컬럼 존재 여부 등 기본 스키마를 검증한다."""
    raise NotImplementedError
