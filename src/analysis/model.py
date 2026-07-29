"""부도확률(PD) 예측 XGBoost 모형 — 본 파이프라인.

피처 구성은 `data/processed/lending_club_변수분류_류성환.xlsx_v2.numbers`
(유지변수_84 시트, 2026-07-27 개정판)의 예측 피처 83개 + 타겟 `loan_status`를 따른다.

결측치는 채우지 않고 NaN 그대로 XGBoost에 넘긴다. `decision_log.md` #13 ③
("결측치 전부 NaN 유지")은 **XGBoost 단일화를 전제**로 조건부 확정된 결론이고, 이
스크립트가 바로 그 전제(모형 학습팀이 XGBoost로 확정)를 채우는 자리다. 범주형도
별도 인코딩 없이 XGBoost native categorical(`enable_categorical=True`)로 넘긴다
(`decision_log.md` #8의 GBM 민감도 비교 — "결측치 native 처리, 범주형 native 처리,
스케일링 불필요" 참고).

⚠️ 모형 후보 자체는 아직 `decision_log.md` #3에서 "진행 중"이다. 팀이 로지스틱 등
다른 모형으로 최종 결정하면 #13의 결측·인코딩 결론이 전제(XGBoost)와 함께 원복된다.

미확정 항목은 상수로 박지 않고 함수 인자로 노출한다 (`AGENTS.md` 공통 원칙):
- `include_default`: Default(268건) 부도 포함 여부 — 변수분류 시트 "팀 미결사항 3)".
  기본 False(엄격정의: Charged Off만 부도) — `preprocessing_validation.py`와 동일 정의.
- `apply_maturity_filter`: 만기 컷 표본 필터 — `decision_log.md` #12(방향 지지, 방식
  재협의 중). 기본 False.
둘 다 팀이 확정하면 이 파일의 기본값만 바꾸면 된다.

train/validation 6:2 분할만 다룬다 — test 20%는 팀 공통으로 별도 관리한다
(`AGENTS.md` "Train 60%/Validation 20%/Test 20%"의 두 번째 단계: 이미 test가 빠진
나머지 80%를 `config.yaml`의 train:validation 비율(0.6:0.2 → 75:25)로 분리).

실행:
    python src/analysis/model.py
    python src/analysis/model.py --raw-file data/processed/lending_club_2020_train_sample_9000.csv
        (스키마 동일 여부·코드 정상 동작만 확인하는 스모크 테스트용 — 결과 수치를
         분석/보고에 인용하지 않는다. `AGENTS.md` "표본은 분석에 쓰지 않는다" 참고.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.config import Config, load_config  # noqa: E402

# ── 변수분류 시트 "유지변수_84" 그대로 — 예측 피처 83개 (타겟 loan_status 제외) ──
PREDICTIVE_FEATURES = [
    # A — 핵심 대출·신청 정보 (22)
    "annual_inc", "delinq_2yrs", "dti", "emp_length", "fico_range_high", "fico_range_low",
    "funded_amnt", "funded_amnt_inv", "grade", "home_ownership", "inq_last_6mths",
    "installment", "int_rate", "loan_amnt", "mort_acc", "pub_rec", "pub_rec_bankruptcies",
    "purpose", "revol_util", "sub_grade", "term", "verification_status",
    # B — CB 상세 계좌통계, 결측 5.3% 이하 (40)
    "acc_open_past_24mths", "addr_state", "application_type", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "collections_12_mths_ex_med",
    "earliest_cr_line", "initial_list_status", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mths_since_recent_bc",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "open_acc", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "revol_bal", "tax_liens", "tot_coll_amt", "tot_cur_bal",
    "tot_hi_cred_lim", "total_acc", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "total_rev_hi_lim",
    # C — 결측 12.7% 이상, 특별처리(D_X) 대상이던 변수. 여기서는 D_X를 만들지 않고
    #     NaN을 그대로 둔다 — T2/T3 처방 차이는 XGBoost가 분기로 흡수한다(#13 근거).
    "mths_since_last_record", "mths_since_recent_bc_dlq", "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq", "mths_since_last_delinq", "il_util",
    "mths_since_rcnt_il", "all_util", "inq_fi", "inq_last_12m", "max_bal_bc",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m",
    "open_rv_24m", "total_bal_il", "total_cu_tl", "mths_since_recent_inq",
    # 특수 — 시점 변수 (1)
    "issue_d",
]

TARGET_COLUMN = "loan_status"
POLICY_PREFIX = "Does not meet the credit policy. Status:"

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
    "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}

# 원-핫 대신 XGBoost native categorical(enable_categorical=True)로 넘길 컬럼
CATEGORICAL_FEATURES = [
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
]

# 문자열에 '%'가 섞여 들어오는 컬럼 — 제거 후 float 변환
PERCENT_FEATURES = ["int_rate", "revol_util"]

# 날짜 문자열("%b-%Y") — 파생 피처만 남기고 원본은 버린다
DATE_FEATURES = ["issue_d", "earliest_cr_line"]


def _read_raw(raw_file: Path) -> pd.DataFrame:
    usecols = sorted(set(PREDICTIVE_FEATURES) | {TARGET_COLUMN, "earliest_cr_line"})
    return pd.read_csv(raw_file, usecols=usecols, low_memory=False)


def _build_target(loan_status: pd.Series, include_default: bool) -> pd.Series:
    """loan_status → 부도 이진 타겟. 대상 외 상태(Current/Late 등)는 NaN으로 남겨 이후 제거."""
    status = loan_status.astype(str).str.replace(POLICY_PREFIX, "", regex=False)
    bad_labels = {"Charged Off"} | ({"Default"} if include_default else set())
    good_labels = {"Fully Paid"}
    y = pd.Series(np.nan, index=loan_status.index)
    y[status.isin(bad_labels)] = 1
    y[status.isin(good_labels)] = 0
    return y


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼을 모델 입력용으로 변환한다. 결측은 대체하지 않고 NaN 그대로 둔다."""
    X = df.copy()

    issue_dt = pd.to_datetime(X["issue_d"], format="%b-%Y", errors="coerce")
    earliest_dt = pd.to_datetime(X["earliest_cr_line"], format="%b-%Y", errors="coerce")
    X["issue_year"] = issue_dt.dt.year.astype("float64")
    month_angle = 2 * np.pi * issue_dt.dt.month / 12
    X["issue_month_sin"] = np.sin(month_angle)
    X["issue_month_cos"] = np.cos(month_angle)
    X["credit_history_months"] = (
        (issue_dt.dt.year - earliest_dt.dt.year) * 12 + (issue_dt.dt.month - earliest_dt.dt.month)
    ).astype("float64")
    X = X.drop(columns=DATE_FEATURES)

    X["fico_avg"] = (X["fico_range_high"] + X["fico_range_low"]) / 2
    X = X.drop(columns=["fico_range_high", "fico_range_low"])

    X["emp_length"] = X["emp_length"].map(EMP_LENGTH_MAP)
    X["term"] = X["term"].astype(str).str.extract(r"(\d+)").astype("float64")

    for col in PERCENT_FEATURES:
        if X[col].dtype == object:
            X[col] = X[col].astype(str).str.rstrip("%").astype("float64")

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    return X


def load_training_data(
    raw_file: Path,
    include_default: bool = False,
    apply_maturity_filter: bool = False,
    maturity_cutoff: pd.Period | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """원본 CSV를 읽어 (X, y)를 반환한다.

    include_default, apply_maturity_filter는 팀에서 아직 확정하지 않은 판단이라
    상수로 박지 않고 인자로 노출한다 — 모듈 docstring 참고.
    """
    raw = _read_raw(raw_file)
    y = _build_target(raw[TARGET_COLUMN], include_default=include_default)
    keep = y.notna()

    if apply_maturity_filter:
        if maturity_cutoff is None:
            raise ValueError("apply_maturity_filter=True면 maturity_cutoff이 필요하다 (decision_log #12).")
        issue_dt = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
        term_m = raw["term"].astype(str).str.extract(r"(\d+)")[0].astype("float64")
        matured = (issue_dt.dt.to_period("M") + term_m) <= maturity_cutoff
        keep &= matured

    df = raw.loc[keep].copy()
    y = y.loc[keep].astype(int)

    X = _build_features(df[PREDICTIVE_FEATURES])
    return X, y


def split_train_validation(
    X: pd.DataFrame, y: pd.Series, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """cfg.split의 train:validation 비율로 분리한다 (test는 이미 빠진 데이터를 가정).

    `AGENTS.md`의 6:2:2 절차 중 두 번째 단계(남은 80%를 75/25로 Train/Validation)에
    해당한다. test 20% 분리는 팀 공통으로 별도 처리하므로 이 함수 밖에서 이뤄진다.
    """
    val_ratio = cfg.split.validation / (cfg.split.train + cfg.split.validation)
    return train_test_split(
        X, y, test_size=val_ratio, stratify=y, random_state=cfg.random_seed.default,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    random_state: int = 42,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=50 if eval_set else None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def predict_default_probability(model: XGBClassifier, X: pd.DataFrame) -> pd.Series:
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="pd_hat")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=None, help="기본값: config.yaml의 data_raw/lending_club_2020_train.csv")
    parser.add_argument("--include-default", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    raw_file = args.raw_file or (cfg.paths.data_raw / "lending_club_2020_train.csv")
    if not raw_file.exists():
        raise FileNotFoundError(
            f"{raw_file} 이 없다. 팀 공유 채널에서 원본(1.2GB, git 미추적)을 받아 "
            f"{cfg.paths.data_raw}/ 에 두거나 --raw-file로 다른 경로를 지정한다."
        )

    X, y = load_training_data(raw_file, include_default=args.include_default)
    X_train, X_val, y_train, y_val = split_train_validation(X, y, cfg)
    print(f"train={len(X_train):,}행 (부도율 {y_train.mean():.4f})  "
          f"validation={len(X_val):,}행 (부도율 {y_val.mean():.4f})")

    model = train_model(X_train, y_train, eval_set=[(X_val, y_val)], random_state=cfg.random_seed.default)

    val_proba = predict_default_probability(model, X_val)
    val_auc = roc_auc_score(y_val, val_proba)
    # AUC는 참고용 모니터링 지표일 뿐, 승인/거절 기준은 아니다 — src/analysis/AGENTS.md.
    print(f"validation AUC = {val_auc:.4f} (참고용 — 승인/거절 threshold는 Sharpe Ratio로 별도 결정)")


if __name__ == "__main__":
    main()
