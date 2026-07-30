"""피처 테이블 구성 — dtype 정리와 6:2:2 분할.

`src/preprocessing/AGENTS.md`의 확정 규칙이 그대로 적용되는 대상이다. 핵심은 **하지 않는 일**이
많다는 것이다.

- **결측은 전부 NaN 그대로 둔다** (#13 ③·#17). 대체도 결측더미도 없다 — XGBoost가 native 처리한다.
  (구 뼈대에 있던 `add_missing_dummies()`는 이 확정으로 폐기됐다.)
- **표준화·로그변환·구간화·캡핑을 하지 않는다** (#17). 트리는 값의 순서만 쓴다.
  `dti` 999 같은 특수값도 원본 그대로 둔다 — 캡핑하면 진짜 100인 행과 뭉개진다.
- **파생(비율) 변수를 만들지 않는다** — 목록이 아직 미확정이다(`src/preprocessing/AGENTS.md` 말미).

따라서 여기서 하는 일은 **"문자열로 저장된 수치를 수치로 되돌리는 것"** 뿐이다.
`'  7.97%'` → `7.97`, `' 36 months'` → `36`, `'Sep-2005'` → 월 서수. 값의 순서를 바꾸지 않는
단조 변환이라 위 금지 사항과 충돌하지 않는다.

범주형은 pandas `category` dtype으로 두고 XGBoost `enable_categorical=True`로 넘긴다 —
원-핫으로 펼치지 않으므로 여기서도 정보를 더하거나 빼지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from preprocessing.loader import (
        TARGET_COLUMN,
        filter_analysis_sample,
        load_raw_loans,
        make_target,
        parse_term_months,
        select_feature_columns,
    )
    from utils.config import load_config
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.loader import (
        TARGET_COLUMN,
        filter_analysis_sample,
        load_raw_loans,
        make_target,
        parse_term_months,
        select_feature_columns,
    )
    from utils.config import load_config


# 값 끝에 `%`가 붙어 문자열로 저장된 수치형
PERCENT_COLUMNS = ("int_rate", "revol_util")

# `'Sep-2005'` 형식으로 저장된 월 단위 날짜
MONTH_COLUMNS = (
    "issue_d",
    "earliest_cr_line",
    "sec_app_earliest_cr_line",
    "last_credit_pull_d",
)

# 순서가 있는 범주 — 문자열 정렬이 의미 순서와 다르므로 명시한다.
EMP_LENGTH_ORDER = (
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years",
)

# 실현수익률(`R`·`rf`·`XR`)을 나중에 결합할 때 쓰는 열. **X에 넣지 않는다.**
META_COLUMNS = ("id", "issue_d", "term", "funded_amnt", "int_rate")


def parse_percent(series: pd.Series) -> pd.Series:
    """`'  7.97%'` → `7.97`. 파싱 불가는 NaN으로 둔다(대체하지 않는다)."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def parse_month(series: pd.Series) -> pd.Series:
    """`'Sep-2005'` → 월 서수(정수). 순서를 보존하는 단조 변환이다.

    연·월로 쪼개지 않는다 — 쪼개면 트리가 같은 해 12월과 다음 해 1월을 먼 값으로 보게 된다.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    # `.dt.to_period("M")`은 Period dtype이라 Int64로 바로 캐스팅되지 않는다(pandas 2.3).
    # 연·월 산술이 같은 순서를 주면서 NaT를 <NA>로 보존한다.
    return (parsed.dt.year * 12 + parsed.dt.month).astype("Int64")


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """문자열로 저장된 수치·날짜를 되돌리고, 나머지 object를 category로 만든다."""
    out = df.copy()

    for col in PERCENT_COLUMNS:
        if col in out.columns:
            out[col] = parse_percent(out[col])

    if "term" in out.columns:
        out["term"] = parse_term_months(out["term"])

    for col in MONTH_COLUMNS:
        if col in out.columns:
            out[col] = parse_month(out[col])

    if "emp_length" in out.columns:
        out["emp_length"] = pd.Categorical(
            out["emp_length"].astype("object"), categories=EMP_LENGTH_ORDER, ordered=True
        )

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype("category")

    return out


def build_feature_table(
    nrows: int | None = None, verify_sample: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """원본 → `(X, y, meta)`.

    `meta`는 threshold·Sharpe 단계에서 실현수익률을 결합할 때 쓰는 열이다.
    **`X`에는 넣지 않는다** — 붙여두면 실수로 학습에 들어간다
    (`src/preprocessing/AGENTS.md` 「누수 방지」).

    `nrows`를 주면 원본 앞부분만 읽으므로 표본 건수 검증을 건너뛴다 — 디버깅 전용이다.
    """
    head = load_raw_loans(nrows=5)
    features, _ = select_feature_columns(list(head.columns))

    needed = sorted(set(features) | set(META_COLUMNS) | {TARGET_COLUMN, "issue_d", "term"})
    available = [c for c in needed if c in head.columns]

    raw = load_raw_loans(usecols=available, nrows=nrows)
    sample = filter_analysis_sample(raw, verify=verify_sample and nrows is None)

    y = make_target(sample[TARGET_COLUMN])

    meta = sample[[c for c in META_COLUMNS if c in sample.columns]].copy()
    meta["term"] = parse_term_months(meta["term"])
    meta["int_rate"] = parse_percent(meta["int_rate"])
    meta["issue_d"] = parse_month(meta["issue_d"])

    X = coerce_dtypes(sample[features])
    return X, y, meta


def split_6_2_2(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame | None = None,
    seed: int | None = None,
) -> dict[str, tuple]:
    """랜덤 6:2:2 분할 (`config/config.yaml`이 단일 출처, `decision_log.md` #13 ④).

    `train_test_split`을 두 번 적용한다 — 전체를 80/20으로 나눠 Test를 떼고, 남은 80%를
    75/25로 갈라 Train 60% / Validation 20%를 만든다.

    부도율이 16.2%로 한쪽에 치우칠 수 있어 두 단계 모두 `stratify`를 건다.
    시간순 분할은 #13 ④에서 실측 기각됐다(Train의 T2 관측률이 0%가 됨).
    """
    cfg = load_config()
    seed = cfg.random_seed.default if seed is None else seed

    val_share_of_rest = cfg.split.validation / (cfg.split.train + cfg.split.validation)

    keys = pd.Series(X.index, index=X.index)
    rest_keys, test_keys = train_test_split(
        keys, test_size=cfg.split.test, random_state=seed, stratify=y
    )
    train_keys, val_keys = train_test_split(
        rest_keys,
        test_size=val_share_of_rest,
        random_state=seed,
        stratify=y.loc[rest_keys.index],
    )

    def take(part: pd.Series) -> tuple:
        idx = part.index
        if meta is None:
            return X.loc[idx], y.loc[idx]
        return X.loc[idx], y.loc[idx], meta.loc[idx]

    return {
        "train": take(train_keys),
        "validation": take(val_keys),
        "test": take(test_keys),
    }


if __name__ == "__main__":
    X, y, meta = build_feature_table()
    print(f"[피처 테이블] X={X.shape}  y={y.shape}  meta={meta.shape}")
    print(f"  부도율: {y.mean():.4%}")

    print(f"\n[dtype 분포]\n{X.dtypes.astype(str).value_counts().to_string()}")

    cat_cols = X.select_dtypes(include="category").columns
    print(f"\n[범주형 {len(cat_cols)}개] 고유값 수")
    for c in cat_cols:
        print(f"  {c:26s} {X[c].nunique():>6,}")

    nan_share = X.isna().mean().sort_values(ascending=False)
    print(f"\n[결측률 상위 5]\n{(nan_share.head(5) * 100).round(2).to_string()}")
    print(f"  결측 없는 컬럼: {(nan_share == 0).sum()}개 / 전체 {len(nan_share)}개")

    parts = split_6_2_2(X, y, meta)
    print("\n[6:2:2 분할]")
    for name, (Xp, yp, _) in parts.items():
        print(f"  {name:11s} n={len(Xp):>8,}  ({len(Xp) / len(X):.1%})  부도율 {yp.mean():.4%}")
