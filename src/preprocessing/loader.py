"""원본 대출 CSV 로딩 · 분석 표본 필터 · 피처 컬럼 선택.

본 파이프라인(`src/preprocessing/AGENTS.md`)의 입구다. 여기서 결정되는 것은 셋이다.

1. **어떤 행을 쓰는가** — `filter_analysis_sample()` → **723,563건** (`decision_log.md` #16)
2. **어떤 열을 쓰는가** — `select_feature_columns()` → 사전(pre-approval) 변수 중 식별자·자유서술 제외
3. **원본을 건드리지 않는가** — 모든 열기는 읽기 전용이다 (`AGENTS.md` 「원본 데이터 취급」)

⚠️ `data/raw/`에 쓰기 모드로 접근하는 함수를 이 모듈에 추가하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # 스크립트 직접 실행과 패키지 임포트 양쪽 지원
    from utils.config import load_config, repo_root
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import load_config, repo_root


RAW_FILENAME = "lending_club_2020_train.csv"
SECOND_TEST_FILENAME = "lending_club_2020_test_2nd.csv"
DICT_FILENAME = "variable_dictionary_byGJ.xlsx"

# 표본 필터 (decision_log.md #16, 2026-07-29 회의 확정)
KEEP_STATUS = ("Fully Paid", "Charged Off")
CREDIT_POLICY_PREFIX = "Does not meet the credit policy. Status:"
MATURITY_CUTOFF = "2020-04"  # issue_d + term ≤ 2020-04 (만기 + 버퍼 6개월)
EXPECTED_SAMPLE_SIZE = 723_563

# ---------------------------------------------------------------------------
# 변수 사전 오버라이드
# ---------------------------------------------------------------------------
# ⚠️ `variable_dictionary_byGJ.xlsx`의 `is_pre_approval` 컬럼이 확정 사항과 어긋나 있다.
# 시트는 아래 8개를 사후(0) 또는 미라벨(NaN)로 두고 있는데, `decision_log.md` #1은 이들을
# **사전 변수로 재분류**했고 #17 ③은 `grade`·`sub_grade`·`int_rate`를 **피처로 투입**하기로
# 확정했다. 시트를 그대로 믿으면 LC 조건변수가 전부 빠져 Lean 스펙(AUC 0.68대)이 되어버린다.
#
# 시트 개정은 `decision_log.md`의 열린 실행 항목이다("변수분류 시트 개정",
# `preprocessing_crosscheck_kgj.md` 10절). 개정되면 이 표는 비워도 된다 —
# `verify_dictionary_overrides()`가 불필요해진 항목을 알려준다.
PRE_APPROVAL_OVERRIDES: dict[str, str] = {
    "grade": "#1 재분류 (시트 미라벨) · #17 ③ 투입 확정",
    "sub_grade": "#1 재분류 (시트 미라벨) · #17 ③ 투입 확정",
    "int_rate": "#1 재분류 (시트 사후) · #17 ③ 투입 확정",
    "installment": "#1 재분류 (시트 사후)",
    "funded_amnt": "#1 재분류 (시트 사후)",
    "funded_amnt_inv": "#1 재분류 (시트 사후)",
    "issue_d": "#1 재분류 (시트 사후)",
    "initial_list_status": "#1 재분류 (시트 사후)",
}

# 사전 변수이지만 피처로 쓰지 않는 것. 누수와는 다른 이유다.
#   - 식별자: 행을 특정할 뿐 예측력이 아니다. `id`는 시트에서 사전(1)으로 라벨돼 있어 명시 제외한다.
#   - 자유서술: 카디널리티가 수만~수십만이라 범주형으로 넣으면 사실상 식별자가 된다.
NON_FEATURE_PRE_APPROVAL: dict[str, str] = {
    "id": "식별자",
    "member_id": "식별자 (원본에 없음)",
    "url": "식별자 (대출 ID를 그대로 포함)",
    "emp_title": "자유서술 (고유값 수십만)",
    "title": "자유서술 (차주 작성)",
    "desc": "자유서술 (원본에 없음)",
}

# 사전 변수이고 식별자·자유서술도 아니지만 **팀 결정으로 피처에서 뺀 것**.
# 위 두 목록과 이유가 다르다 — 저 둘은 "원리상 피처가 아니다", 이쪽은 "실측해 보니 모형을
# 깎는다"다. 그래서 목록을 합치지 않고 따로 둔다(사유가 섞이면 나중에 판단 근거를 잃는다).
#
# `zip_code`: 911개 범주의 고카디널리티 변수로, XGBoost가 Train의 칸별 부도율을 암기한다.
#   같은 재분할에 `main` / `main − zip_code`를 짝지어 돌린 검증에서 제외 쪽이 **24 seed 전부
#   우세**했고(재투자 가정 2종 × 랭킹 기준 3종 = 6조합 모두 24/24), 국채·`q_score` 기준
#   짝지은 차이는 Δ Sharpe **+0.0122**(sd 0.0020)였다.
#   근거: `outputs/reports/model_comparison_kgj.md`, `outputs/model_comparison_stability.csv`.
EXCLUDED_BY_DECISION: dict[str, str] = {
    "zip_code": "고카디널리티(911범주) 과적합 — 짝지은 검증 24/24 제외 우세, Δ Sharpe +0.0122",
}

TARGET_COLUMN = "loan_status"


def raw_path() -> Path:
    """원본 대출 CSV 경로. 1.2GB이며 git에 없다 — 팀 공유 채널에서 받는다."""
    return load_config().paths.data_raw / RAW_FILENAME


def second_test_path() -> Path:
    """**2nd Test CSV** 경로 (1,170,198행 × 141열, 850MB, git 미추적).

    이름의 `2nd`는 **매니페스트 6:2:2의 Test(20%, 144,713건)와 다른 집합**임을 뜻한다 —
    둘을 같은 "Test"로 부르면 어느 쪽 Sharpe인지 구분되지 않는다.

    `RAW_FILENAME`(train)과 **id가 한 건도 겹치지 않는 별도 파일**이며 컬럼 구성은 동일하다.
    `filter_analysis_sample(..., verify=False)`를 통과하면 481,833건이 남는다
    (부도율 16.20%) — `EXPECTED_SAMPLE_SIZE`는 train 기준이므로 `verify=True`로 부르지 않는다.
    """
    return load_config().paths.data_raw / SECOND_TEST_FILENAME


def dictionary_path() -> Path:
    return load_config().paths.data_processed / DICT_FILENAME


# ---------------------------------------------------------------------------
# 변수 사전
# ---------------------------------------------------------------------------
def load_variable_dictionary(xlsx_path: Path | None = None) -> pd.DataFrame:
    """변수 사전 + 사전/사후 라벨을 로드한다 (`decision_log.md` #9 — 단일 원본)."""
    path = xlsx_path or dictionary_path()
    if not path.exists():
        raise FileNotFoundError(f"변수 사전이 없습니다: {path}")
    d = pd.read_excel(path)
    d["LoanStatNew"] = d["LoanStatNew"].astype(str).str.strip()
    return d


def pre_approval_columns(dictionary: pd.DataFrame | None = None) -> list[str]:
    """사전(pre-approval) 변수 목록. 시트 라벨에 `PRE_APPROVAL_OVERRIDES`를 얹은 결과다."""
    d = dictionary if dictionary is not None else load_variable_dictionary()
    labelled = set(d.loc[d["is_pre_approval"] == 1, "LoanStatNew"])
    return sorted(labelled | set(PRE_APPROVAL_OVERRIDES))


def verify_dictionary_overrides(dictionary: pd.DataFrame | None = None) -> list[str]:
    """오버라이드 중 시트가 이미 반영한 항목을 돌려준다.

    비어 있지 않으면 시트가 아직 개정 전이라는 뜻이고, 전부 반영되면 결과가 오버라이드
    전체와 같아진다 — 그때 `PRE_APPROVAL_OVERRIDES`를 비울 수 있다.
    """
    d = dictionary if dictionary is not None else load_variable_dictionary()
    already = set(d.loc[d["is_pre_approval"] == 1, "LoanStatNew"])
    return sorted(set(PRE_APPROVAL_OVERRIDES) & already)


def select_feature_columns(
    raw_columns: list[str],
    dictionary: pd.DataFrame | None = None,
    apply_decisions: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """피처로 쓸 컬럼과, 제외된 사전 변수의 사유를 함께 돌려준다.

    사후 변수는 애초에 `pre_approval_columns()`에 없으므로 여기서 다시 거르지 않는다
    (누수 방지 목록은 전부 사후로 라벨돼 있음을 `test_leakage_columns_are_post()`가 확인한다).

    `apply_decisions=True`(기본)면 `EXCLUDED_BY_DECISION`도 뺀다. **비교 실험에서만 False로
    둔다** — 제외 전후를 나란히 돌려야 하는 `model_comparison.py`가 그 경우다. 본 파이프라인은
    기본값을 쓴다.
    """
    pre = pre_approval_columns(dictionary)
    present = [c for c in pre if c in raw_columns]
    dropped = dict(NON_FEATURE_PRE_APPROVAL)
    if apply_decisions:
        dropped |= EXCLUDED_BY_DECISION

    excluded: dict[str, str] = {}
    for col in pre:
        if col not in raw_columns:
            excluded[col] = "원본에 없음"
        elif col in dropped:
            excluded[col] = dropped[col]

    features = [c for c in present if c not in dropped]
    return features, excluded


# ---------------------------------------------------------------------------
# 원본 로딩
# ---------------------------------------------------------------------------
def load_raw_loans(
    usecols: list[str] | None = None,
    nrows: int | None = None,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """원본 대출 CSV를 **읽기 전용**으로 로드한다.

    전수 1,755,295행 × 141열이다. `usecols`로 필요한 열만 받는 것을 권한다 —
    전체를 올리면 수 GB를 쓴다.
    """
    path = csv_path or raw_path()
    if not path.exists():
        raise FileNotFoundError(
            f"원본 대출 데이터가 없습니다: {path}\n"
            "1.2GB라 git에 없습니다 — 팀 공유 채널(Google Drive 등)에서 받아 두세요."
        )
    return pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)


def normalize_loan_status(loan_status: pd.Series) -> pd.Series:
    """`Does not meet the credit policy. Status:` 접두사를 떼어 상태를 통일한다.

    접두사가 붙은 건도 각각 `Fully Paid`/`Charged Off`와 동일하게 취급한다
    (`src/preprocessing/AGENTS.md` 행 필터링 규칙).
    """
    return (
        loan_status.astype(str)
        .str.replace(CREDIT_POLICY_PREFIX, "", regex=False)
        .str.strip()
    )


def parse_term_months(term: pd.Series) -> pd.Series:
    """`' 36 months'` → `36`."""
    return term.astype(str).str.extract(r"(\d+)")[0].astype(float)


def filter_analysis_sample(df: pd.DataFrame, verify: bool = True) -> pd.DataFrame:
    """분석 표본 필터 — **723,563건** (`decision_log.md` #16).

    두 조건의 교집합이다.

    1. **상태**: `Fully Paid` / `Charged Off`만 남긴다 (접두사 정규화 후).
       `Current`·`Late`·`In Grace Period`·`Default`·`Issued`는 완결건이 아니라 제외된다.
    2. **만기 + 버퍼 6개월**: `issue_d + term ≤ 2020-04`.
       스냅샷(2020-10) 시점에 만기가 도래하지 않은 완결건은 조기부도가 과대표집돼
       부도율이 8.5%p 높다.

    `Default` 268건(만기 내 113건)은 상각 미확정 단계라 **제외한다** —
    `decision_log.md` #24 ① 정정(2026-07-31). K=50·2nd Test 등 공표된 모든 수치가
    이 기준(723,563건)으로 산출됐다. 합치면 표본이 723,676건이 된다.
    """
    status = normalize_loan_status(df[TARGET_COLUMN])
    term_months = parse_term_months(df["term"])
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")

    # 연·월 산술로 절대 월 서수를 만든다. Period dtype을 거치면 NaT가 int 최솟값으로
    # 뭉개져 **만기 조건을 통과해버리므로** 쓰지 않는다.
    maturity_ord = (issue.dt.year * 12 + issue.dt.month) + term_months
    cutoff = pd.Timestamp(MATURITY_CUTOFF)
    cutoff_ord = cutoff.year * 12 + cutoff.month

    keep = status.isin(KEEP_STATUS) & (maturity_ord <= cutoff_ord) & issue.notna()
    out = df.loc[keep].copy()
    out[TARGET_COLUMN] = status.loc[keep]

    if verify and len(out) != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            f"표본 건수가 확정값과 다릅니다: {len(out):,} != {EXPECTED_SAMPLE_SIZE:,}. "
            "원본 파일이나 필터 정의가 바뀐 것입니다 — decision_log.md #16을 확인하세요. "
            "표본/부분 로딩으로 의도적으로 다르게 돌릴 때만 verify=False를 씁니다."
        )
    return out


def make_target(loan_status: pd.Series) -> pd.Series:
    """이진 타깃 — `Charged Off` = 1(부도), `Fully Paid` = 0."""
    normalized = normalize_loan_status(loan_status)
    unknown = set(normalized.unique()) - set(KEEP_STATUS)
    if unknown:
        raise ValueError(f"표본에 완결 상태가 아닌 값이 있습니다: {sorted(unknown)}")
    return (normalized == "Charged Off").astype("int8")


def validate_schema(df: pd.DataFrame, required_columns: list[str]) -> None:
    """필수 컬럼 존재 여부를 검증하고, 없으면 무엇이 빠졌는지 알려준다."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 {len(missing)}개가 없습니다: {missing}")


if __name__ == "__main__":
    print(f"저장소 루트: {repo_root()}")

    d = load_variable_dictionary()
    already = verify_dictionary_overrides(d)
    print(f"\n[변수 사전] {len(d)}행")
    print(f"  시트 라벨 사전(1): {(d['is_pre_approval'] == 1).sum()}  "
          f"사후(0): {(d['is_pre_approval'] == 0).sum()}  "
          f"미라벨(NaN): {d['is_pre_approval'].isna().sum()}")
    if already:
        print(f"  ✅ 시트가 이미 반영한 오버라이드: {already}")
        print("     → 시트가 개정된 항목입니다. PRE_APPROVAL_OVERRIDES에서 빼도 됩니다.")
    else:
        print(f"  ⚠️ 시트 미개정 — 오버라이드 {len(PRE_APPROVAL_OVERRIDES)}건을 코드에서 적용 중")
        for col, why in PRE_APPROVAL_OVERRIDES.items():
            print(f"       {col:22s} {why}")

    head = load_raw_loans(nrows=5)
    features, excluded = select_feature_columns(list(head.columns), d)
    print(f"\n[피처] {len(features)}개")
    print(f"  제외된 사전 변수 {len(excluded)}개:")
    for col, why in sorted(excluded.items()):
        print(f"       {col:22s} {why}")
