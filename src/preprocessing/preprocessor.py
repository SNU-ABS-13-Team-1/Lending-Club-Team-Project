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

import numpy as np
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
    nrows: int | None = None,
    verify_sample: bool = True,
    apply_decisions: bool = True,
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """원본 → `(X, y, meta)`.

    `meta`는 threshold·Sharpe 단계에서 실현수익률을 결합할 때 쓰는 열이다.
    **`X`에는 넣지 않는다** — 붙여두면 실수로 학습에 들어간다
    (`src/preprocessing/AGENTS.md` 「누수 방지」).

    `nrows`를 주면 원본 앞부분만 읽으므로 표본 건수 검증을 건너뛴다 — 디버깅 전용이다.

    `apply_decisions=False`면 `loader.EXCLUDED_BY_DECISION`(현재 `zip_code`)을 **빼지 않고**
    돌려준다 — 제외 전후를 나란히 비교해야 하는 `model_comparison.py` 전용이다.
    본 파이프라인은 기본값을 쓴다.

    `csv_path`는 train 원본 대신 다른 CSV를 같은 파이프라인으로 읽을 때만 준다
    (`loader.second_test_path()`의 2nd Test). 표본 건수 검증은 train 기준이므로
    그때는 `verify_sample=False`를 함께 준다.
    """
    head = load_raw_loans(nrows=5, csv_path=csv_path)
    features, _ = select_feature_columns(list(head.columns), apply_decisions=apply_decisions)

    needed = sorted(set(features) | set(META_COLUMNS) | {TARGET_COLUMN, "issue_d", "term"})
    available = [c for c in needed if c in head.columns]

    raw = load_raw_loans(usecols=available, nrows=nrows, csv_path=csv_path)
    sample = filter_analysis_sample(raw, verify=verify_sample and nrows is None)

    y = make_target(sample[TARGET_COLUMN])

    meta = sample[[c for c in META_COLUMNS if c in sample.columns]].copy()
    meta["term"] = parse_term_months(meta["term"])
    meta["int_rate"] = parse_percent(meta["int_rate"])
    meta["issue_d"] = parse_month(meta["issue_d"])

    X = coerce_dtypes(sample[features])
    return X, y, meta


# ---------------------------------------------------------------------------
# 분할 체계 — 셋이 공존한다
# ---------------------------------------------------------------------------
#: `{scheme: (train, validation, test)}`. **비율을 코드에 흩어 놓지 않는다.**
#:
#: - `"6_2_2"` — 1차 데이터 안에서 Test 20%를 떼는 원래 체계(#13 ④·#18). `config.yaml`이
#:   단일 출처이며 여기 적힌 값은 그 대조용이다.
#: - `"7_3"`  — **1차 데이터 전량을 Train/Validation으로 쓰고 Test는 별도 파일**
#:   (`lending_club_2020_test_2nd.csv`, 481,833건)로 삼는 체계(#30, 2026-07-31 확정).
#:   6:2:2 대비 Train +16.7%(434,137 → 506,494) · Validation +50.0%(144,713 → 217,069).
#: - `"8_2"`  — 7:3과 같이 Test를 별도 파일로 두되 Train을 더 준다(#32, 2026-07-31).
#:   Train 578,850 / Validation 144,713. 7:3 대비 Train +14.3% · Validation −33.3%다 —
#:   Validation이 6:2:2와 같은 크기로 돌아가므로 `τ*` 표본오차와 승자의 저주가 함께 커진다.
#:   그 대가를 알고 고른 것이며 근거는 이슈 #32에 있다.
#:
#: ⚠️ 체계를 늘릴 때는 `SPLIT_FUNCTIONS`(`export_split_manifest.py`)도 함께 채운다 —
#: 여기에만 추가하면 매니페스트를 만들 수 없다.
SPLIT_SCHEMES: dict[str, tuple[float, float, float]] = {
    "6_2_2": (0.6, 0.2, 0.2),
    "7_3": (0.7, 0.3, 0.0),
    "8_2": (0.8, 0.2, 0.0),
}

DEFAULT_SCHEME = "6_2_2"


def scheme_ratios(scheme: str) -> tuple[float, float, float]:
    """분할 체계의 `(train, validation, test)` 비율. 알 수 없는 이름이면 예외다.

    `"6_2_2"`는 `config.yaml`을 읽어 돌려준다 — 그쪽이 단일 출처이므로 값이 어긋나면
    **조용히 넘어가지 않고** 예외로 알린다.
    """
    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"알 수 없는 분할 체계: {scheme!r} (가능: {sorted(SPLIT_SCHEMES)})")
    if scheme != DEFAULT_SCHEME:
        return SPLIT_SCHEMES[scheme]

    cfg = load_config().split
    ratios = (cfg.train, cfg.validation, cfg.test)
    if not np.isclose(ratios, SPLIT_SCHEMES[scheme]).all():
        raise ValueError(
            f"config.yaml의 split {ratios}이 '{scheme}' 정의 {SPLIT_SCHEMES[scheme]}와 다르다 — "
            "둘 중 어느 쪽이 맞는지 정하고 맞춰라."
        )
    return ratios


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame | None = None,
    seed: int | None = None,
    scheme: str = "7_3",
) -> dict[str, tuple]:
    """**Test 칸이 없는** 체계의 2분할 — `"7_3"`(#30)과 `"8_2"`(#32)가 여기로 온다.

    1차 데이터(723,563건) 전량을 Train/Validation으로 가른다. Test는 이 표본 안에서
    떼지 않고 **별도 파일**(`loader.second_test_path()`)을 쓴다.

    비율은 `scheme_ratios(scheme)`에서 온다 — **비율을 이 함수에 박지 않는다.** 체계를
    늘릴 때 여기를 고치지 않아도 되게 하려는 것이다(#32에서 8:2를 넣을 때 `"7_3"`이
    하드코딩돼 있어 실제로 걸렸던 부분이다).

    부도율 16.2%가 한쪽에 치우치지 않도록 `stratify`를 건다.
    """
    seed = load_config().random_seed.default if seed is None else seed
    _, val_share, test_share = scheme_ratios(scheme)
    if test_share:
        raise ValueError(
            f"'{scheme}'은 Test 칸이 있는 체계다({test_share:.0%}) — "
            "split_train_validation()이 아니라 split_6_2_2()를 쓴다."
        )

    keys = pd.Series(X.index, index=X.index)
    train_keys, val_keys = train_test_split(
        keys, test_size=val_share, random_state=seed, stratify=y
    )

    def take(part: pd.Series) -> tuple:
        idx = part.index
        if meta is None:
            return X.loc[idx], y.loc[idx]
        return X.loc[idx], y.loc[idx], meta.loc[idx]

    return {"train": take(train_keys), "validation": take(val_keys)}


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


def load_split_manifest(
    seed: int | None = None, scheme: str = DEFAULT_SCHEME
) -> pd.Series:
    """`id → split` 매니페스트를 읽는다. 인덱스는 `id`(문자열)다.

    매니페스트는 `src/preprocessing/export_split_manifest.py`가 만든다. 없으면 예외다 —
    **임의로 seed 분할로 대체하지 않는다.** 조용히 다른 분할로 넘어가면 팀원 간 Test가
    어긋나는데 아무도 모르게 된다.

    `scheme`은 `SPLIT_SCHEMES`의 키(`"6_2_2"`·`"7_3"`·`"8_2"`)이며 **파일이 다르다** —
    서로 덮어쓰지 않는다.
    """
    from preprocessing.export_split_manifest import manifest_path

    cfg = load_config()
    seed = cfg.random_seed.default if seed is None else seed
    path = manifest_path(seed, scheme=scheme)
    if not path.exists():
        raise FileNotFoundError(
            f"분할 매니페스트가 없다: {path}\n"
            f"먼저 `python src/preprocessing/export_split_manifest.py --scheme {scheme}`로 "
            "만들거나, 팀 저장소에서 받아라."
        )
    m = pd.read_csv(path, dtype={"id": str})
    return m.set_index("id")["split"]


def split_from_manifest(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    seed: int | None = None,
    unlock_test: bool = False,
    scheme: str = DEFAULT_SCHEME,
) -> dict[str, tuple]:
    """**매니페스트 기준** 6:2:2 분할 — 팀원 전원이 동일한 분할을 쓰는 경로.

    `split_6_2_2()`(seed 기반)와 달리 `id`로 매칭하므로 **원본 CSV의 행 순서·pandas·sklearn
    버전과 무관**하다. 왜 그게 필요한지는 `export_split_manifest.py` docstring 참고 —
    요약하면 `train_test_split`은 위치를 셔플하므로 행 순서가 다르면 분할이 달라지는데
    723,563건 검증은 건수만 보므로 **조용히 어긋난다.**

    ⚠️ **Test는 기본적으로 반환하지 않는다.** `unlock_test=True`를 명시해야 나온다.
    실수로 `parts["test"]`를 집는 일을 막기 위한 것이다 — Test는 모형·threshold가 전부
    확정된 뒤 **단 1회** 적용한다(`src/analysis/AGENTS.md`).

    ## 반환 행 순서를 `id`로 고정한다 — 분할만 맞춰선 부족하다

    분할 집합이 같아도 **행 순서가 다르면 K-fold 배정이 달라진다.**
    `StratifiedKFold(shuffle=True, random_state=seed)`도 `train_test_split`처럼 **위치**를
    셔플하기 때문이다. 그러면 팀원마다 OOF PD가 미세하게 달라지고, 그 PD로 만든 분위 경계·
    칸별 통계표·`τ*`가 조금씩 어긋난다.

    그래서 각 split을 **`id` 오름차순으로 정렬해** 돌려준다. 이러면 OOF fold까지
    `(id 집합, seed)`만으로 결정되어 **원본 파일의 행 순서와 완전히 무관**해진다.

    ⚠️ 이 정렬 때문에 `split_6_2_2()`로 낸 기존 수치와 **소수 넷째 자리 수준의 차이**가 생긴다
    (분할 자체는 동일하고 fold 구성만 바뀐다). 파이프라인을 이 함수로 넘긴 뒤에는 산출물을
    한 번 다시 만들어야 한다.
    """
    manifest = load_split_manifest(seed, scheme=scheme)
    ids = meta["id"].astype(str)

    assigned = ids.map(manifest)
    missing = int(assigned.isna().sum())
    if missing:
        raise RuntimeError(
            f"매니페스트에 없는 id가 {missing:,}건이다 — 표본 필터가 매니페스트 생성 시점과 "
            "다르다. 필터를 고치기 전에 왜 달라졌는지부터 확인하라(#16)."
        )

    def take(name: str) -> tuple:
        idx = ids[assigned == name].sort_values(kind="mergesort").index
        return X.loc[idx], y.loc[idx], meta.loc[idx]

    parts = {"train": take("train"), "validation": take("validation")}
    n_test = int((assigned == "test").sum())

    if not n_test:
        # 7:3 — Test 칸이 아예 없다. 최종 평가는 별도 파일(2nd Test)로 한다(#30).
        if unlock_test:
            raise ValueError(
                f"'{scheme}' 매니페스트에는 Test 칸이 없다 — unlock_test는 쓸 수 없다. "
                "최종 평가는 `loader.second_test_path()`(2nd Test)로 한다(#30)."
            )
        return parts

    if unlock_test:
        print(f"⚠️  Test set을 열었다 ({n_test:,}건). 모형·threshold 확정 후 **1회만** 쓴다 — "
              "여기서 무언가를 고르거나 조정하면 규칙 위반이다.")
        parts["test"] = take("test")
    else:
        print(f"   Test {n_test:,}건은 잠긴 상태다 (unlock_test=True로 열 수 있다).")
    return parts


def resplit_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    seed: int,
    manifest_seed: int | None = None,
    scheme: str = DEFAULT_SCHEME,
) -> dict[str, tuple]:
    """**Test를 고정한 채** Train+Validation 풀만 재분할한다 (#18의 K=50 반복용).

    `README.md`의 규정 그대로다 — *"Test set은 그대로 고정해두고, 남은 Train+Validation 풀에서
    매번 다른 방식으로 Train/Validation을 재분할해 모형 학습 → threshold 탐색을 K=50회 반복"*.

    ⚠️ `split_6_2_2(seed=k)`를 반복에 쓰면 **안 된다.** 그건 전체를 다시 80/20으로 갈라
    **Test 구성까지 매번 바꾼다** — seed마다 다른 대출이 Test에 들어가므로 "Test를 고정해두고"가
    깨지고, 어떤 seed의 Test 건이 다른 seed의 Train에 들어가 사실상 Test가 오염된다.

    풀 안의 Validation 비율은 체계마다 다르다.

    | scheme | 풀 | Validation 비율 | Train / Validation |
    | --- | --- | --- | --- |
    | `6_2_2` | 매니페스트의 비Test 80% | `0.2/(0.6+0.2)` = **0.25** | 434,137 / 144,713 |
    | `7_3` | **표본 전량**(Test 칸 없음) | `0.3/(0.7+0.3)` = **0.30** | 506,494 / 217,069 |
    | `8_2` | **표본 전량**(Test 칸 없음) | `0.2/(0.8+0.2)` = **0.20** | 578,850 / 144,713 |

    `7_3`·`8_2`에서도 Test가 흔들리지 않는다 — Test가 애초에 이 표본 밖(별도 파일)이기
    때문이다(#30·#32).
    부도율이 16.2%로 치우칠 수 있어 `stratify`를 건다.
    """
    parts = split_from_manifest(X, y, meta, seed=manifest_seed, scheme=scheme)

    # 풀의 **순서를 `id`로 고정한다.** `train_test_split`이 위치를 셔플하므로, 순서가 팀원마다
    # 다르면 같은 seed로도 다른 Train/Validation이 나온다. 원본 행번호로 정렬하면 원본 파일의
    # 행 순서에 다시 의존하게 되므로 `id` 기준이어야 한다.
    pool_ids = pd.concat([parts["train"][2]["id"], parts["validation"][2]["id"]]).astype(str)
    pool_idx = pool_ids.sort_values(kind="mergesort").index

    tr_share, va_share, _ = scheme_ratios(scheme)
    val_share_of_pool = va_share / (tr_share + va_share)
    keys = pd.Series(pool_idx, index=pool_idx)
    train_keys, val_keys = train_test_split(
        keys,
        test_size=val_share_of_pool,
        random_state=seed,
        stratify=y.loc[pool_idx],
    )

    def take(part: pd.Series) -> tuple:
        idx = part.index
        return X.loc[idx], y.loc[idx], meta.loc[idx]

    return {"train": take(train_keys), "validation": take(val_keys)}


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
