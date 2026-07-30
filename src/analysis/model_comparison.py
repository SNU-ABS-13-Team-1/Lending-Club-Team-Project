"""PD 모형 후보 교차검증 — `outputs/reports/model_comparison_kgj.md`의 표를 재생성한다.

팀원 3인이 각자 브랜치에 만든 모형 후보를 **동일 프로토콜**로 붙여 비교한다.
후보 스크립트들은 `.fit()` 1회 + Validation AUC 1회 구조라 OOF·보정·threshold 단계가 없어
그대로는 비교가 성립하지 않는다 — 그래서 **스펙(피처 + 하이퍼파라미터)만 추출해 본
파이프라인에 이식**한다. 이 파일이 그 이식 코드다.

## 성격 — 재현·검증 스크립트다

`AGENTS.md` 「자료 위치 색인」 ② 범주. **본 파이프라인이 아니다.** 다만 탐색용 비교
스크립트 4종과 달리 **`config.yaml`의 6:2:2 매니페스트를 그대로 쓴다** — 비교 대상이
"팀 표준 분할에서 어느 스펙이 나은가"이므로 자체 분할을 쓰면 질문이 달라진다.
Test는 열지 않는다.

⚠️ **AUC는 스펙 간 상대 비교 전용이다.** 승인/거절 기준은 Sharpe로만 정한다
(`src/analysis/AGENTS.md`). 그리고 `--only decompose`의 A행(0.731)은 **#16 필터 미적용**
값이므로 어떤 문서에도 성능으로 인용하지 않는다.

## 다섯 블록

| 블록 | 재현 대상 | 산출 |
| --- | --- | --- |
| `cv` | 리포트 4절 — 6스펙 5-fold 교차검증 + Δ Sharpe | `model_comparison_crossvalidation.csv` |
| `decompose` | 리포트 3절 전반 — 필터·분할·early stopping·피처 4단 분해 | `model_comparison_ayh_decompose.csv` |
| `ablation` | 리포트 3절 후반 — 20개 그룹별 기여 | `model_comparison_zip_ablation.csv` |
| `seed` | 리포트 4절 보강 — 학습 seed 흔들림 + 페어드 부트스트랩 | `model_comparison_seed_stability.csv` |
| `zipdiag` | 리포트 5절 — `zip_code` 암기 진단 (학습 없음, 수초) | `model_comparison_zip_diagnostic.csv`<br>`model_comparison_zip_examples.csv` |

## 실행

    /opt/anaconda3/bin/python src/analysis/model_comparison.py                  # 전부 (~20분)
    /opt/anaconda3/bin/python src/analysis/model_comparison.py --only cv        # 블록 하나만
    /opt/anaconda3/bin/python src/analysis/model_comparison.py --only zipdiag   # 수초

**전제**: ⓐ 원본 `data/raw/lending_club_2020_train.csv` (1.2GB, git 미추적) —
`cv`·`decompose`가 읽는다 ⓑ 공유 parquet `data/processed/shared/` —
`ablation`·`seed`·`zipdiag`가 읽는다 ⓒ 분할 매니페스트 (`config.yaml`의 seed 기준).
scikit-learn·xgboost가 필요하다 — macOS 시스템 `python3`에는 없다(`AGENTS.md` 실행 환경).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.model import (  # noqa: E402
    apply_calibrator,
    assign_quantile_by_term,
    calibration_metrics,
    fit_calibrator,
    quantile_edges_by_term,
)
from analysis.realized_return import (  # noqa: E402
    ReturnAssumptions,
    build_excess_returns,
    build_return_inputs,
    cash_reinvestment,
    default_cell_stats,
    expected_excess_return,
    q_score,
    variance_excess_return,
)
from analysis.sharpe_optimizer import (  # noqa: E402
    append_row_csv,
    compare_ranking_criteria,
    completed_keys,
    find_optimal_threshold,
)
from preprocessing.export_shared_dataset import load_shared  # noqa: E402
from preprocessing.loader import MATURITY_CUTOFF, raw_path  # noqa: E402
from preprocessing.preprocessor import (  # noqa: E402
    build_feature_table,
    load_split_manifest,
    resplit_train_validation,
    split_from_manifest,
)
from utils.config import load_config, repo_root  # noqa: E402

N_FOLDS = 5
N_BOOT = 2_000
SEED_SWEEP = [42, 7, 2026, 99]  # config seed는 런타임에 맨 앞으로 붙인다

# ---------------------------------------------------------------------------
# 비교 대상 스펙 — 하이퍼파라미터
# ---------------------------------------------------------------------------
# ⚠️ `random_state`는 넣지 않는다. 실행 시 `config.yaml`의 seed를 주입한다
#    (팀원 원본은 각자 다른 seed를 썼고, 그 차이가 곧 seed 잡음이다 — 블록 `seed` 참고).


def params_main() -> dict:
    """`src/analysis/model.py` `default_params()` (origin/main)와 동일."""
    return dict(
        n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1,
    )


def params_ljh() -> dict:
    """`origin/feature/#5-xgb-model-ljh` : `run_xgboost_candidate.py` (`77d857d`).

    원본은 `eval_set=[(X_val, y_val)]`을 넘기지만 `early_stopping_rounds`가 없어 학습에
    영향이 없다 — 여기서는 넘기지 않는다(넘기면 Validation이 학습 신호가 될 위험만 남는다).
    """
    return dict(
        n_estimators=300, learning_rate=0.04, max_depth=4, min_child_weight=10,
        subsample=0.75, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0,
        enable_categorical=True, tree_method="hist", eval_metric="logloss", n_jobs=-1,
    )


def params_ayh() -> dict:
    """커밋 `fe88d6c` : `src/analysis/model.py` `train_model()`.

    ⚠️ 이 코드는 브랜치 `feature/#9-setup-ayh`의 tip에 **없다** — 머지(`da2d94f`)가
    `model.py`를 main 쪽으로 해소하면서 소실됐다. 원문은 `git show fe88d6c:src/analysis/model.py`.
    원본의 `early_stopping_rounds=50`은 500트리 내내 미발동해 기여가 0으로 실측됐으므로
    (블록 `decompose`의 B행 = C행) 여기서는 뺀다.
    """
    return dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        enable_categorical=True, eval_metric="auc", n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 비교 대상 스펙 — 안예환 피처 목록 (커밋 `fe88d6c`에서 전사)
# ---------------------------------------------------------------------------
# 출처: `data/processed/lending_club_변수분류_류성환.xlsx_v2.numbers` 「유지변수_84」 시트
# (2026-07-27). 팀 단일 원본은 `variable_dictionary_byGJ.xlsx`이며 그쪽은 102개를 낸다 —
# 두 문서가 답하는 질문이 다르다(리포트 6절). 여기 목록은 **비교 대상 스펙의 정의**로서
# 값을 박아 둔다. 팀원 모듈을 import하지 않는 이유는 그 파일이 브랜치 tip에 없기 때문이다.
AYH_FEATURES = [
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
    # C — 결측 12.7% 이상 (20)
    "mths_since_last_record", "mths_since_recent_bc_dlq", "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq", "mths_since_last_delinq", "il_util",
    "mths_since_rcnt_il", "all_util", "inq_fi", "inq_last_12m", "max_bal_bc",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m",
    "open_rv_24m", "total_bal_il", "total_cu_tl", "mths_since_recent_inq",
    # 특수 — 시점 변수 (1)
    "issue_d",
]

AYH_CATEGORICAL = [
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
]
AYH_EMP_LENGTH = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
    "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}
POLICY_PREFIX = "Does not meet the credit policy. Status:"

# 팀 102피처 중 안예환이 쓰지 않은 20개 — 블록 `ablation`의 그룹 정의
SEC_APP = [
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "sec_app_earliest_cr_line", "sec_app_fico_range_high", "sec_app_fico_range_low",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_num_rev_accts",
    "sec_app_open_acc", "sec_app_open_act_il", "sec_app_revol_util",
]
JOINT = ["annual_inc_joint", "dti_joint", "revol_bal_joint"]
RARE_DELINQ = ["acc_now_delinq", "delinq_amnt", "num_tl_120dpd_2m", "num_tl_30dpd"]


def hdr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def out_dir() -> Path:
    d = repo_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def maturity_cutoff_ord() -> int:
    """`issue_d + term ≤ 2020-04`의 절대 월 서수 (#16). `loader.MATURITY_CUTOFF`가 원본."""
    c = pd.Timestamp(MATURITY_CUTOFF)
    return c.year * 12 + c.month


# ---------------------------------------------------------------------------
# 안예환 피처 파이프라인 (`fe88d6c`의 `_build_target` · `_build_features` 재현)
# ---------------------------------------------------------------------------
def _ayh_target(loan_status: pd.Series) -> pd.Series:
    """`Charged Off`=1 / `Fully Paid`=0, 나머지는 NaN. 신용정책 접두사를 떼고 판정한다."""
    status = loan_status.astype(str).str.replace(POLICY_PREFIX, "", regex=False)
    y = pd.Series(np.nan, index=loan_status.index)
    y[status == "Charged Off"] = 1
    y[status == "Fully Paid"] = 0
    return y


def _ayh_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 → 파생 4개, FICO 2개 → 평균 1개. 결측은 채우지 않는다.

    83개 목록이 **84개 학습 피처**가 되는 계산: 83 − 2(날짜) + 4(파생) − 2(FICO) + 1(평균).
    """
    X = df.copy()
    issue = pd.to_datetime(X["issue_d"], format="%b-%Y", errors="coerce")
    earliest = pd.to_datetime(X["earliest_cr_line"], format="%b-%Y", errors="coerce")
    X["issue_year"] = issue.dt.year.astype("float64")
    angle = 2 * np.pi * issue.dt.month / 12
    X["issue_month_sin"] = np.sin(angle)
    X["issue_month_cos"] = np.cos(angle)
    X["credit_history_months"] = (
        (issue.dt.year - earliest.dt.year) * 12 + (issue.dt.month - earliest.dt.month)
    ).astype("float64")
    X = X.drop(columns=["issue_d", "earliest_cr_line"])

    X["fico_avg"] = (X["fico_range_high"] + X["fico_range_low"]) / 2
    X = X.drop(columns=["fico_range_high", "fico_range_low"])

    X["emp_length"] = X["emp_length"].map(AYH_EMP_LENGTH)
    X["term"] = X["term"].astype(str).str.extract(r"(\d+)").astype("float64")
    for c in ("int_rate", "revol_util"):
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.rstrip("%").astype("float64")
    for c in AYH_CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def load_ayh_raw(apply_maturity_filter: bool) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """원본을 **읽기 전용**으로 열어 안예환 스펙 테이블을 만든다. → `(X, y, id)`

    `apply_maturity_filter=False`가 `fe88d6c` `main()`의 실제 동작이다 — #16 필터가 빠진다.
    """
    usecols = sorted(set(AYH_FEATURES) | {"loan_status", "earliest_cr_line", "id"})
    raw = pd.read_csv(raw_path(), usecols=usecols, low_memory=False)
    y = _ayh_target(raw["loan_status"])
    keep = y.notna()

    if apply_maturity_filter:
        issue = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
        term = raw["term"].astype(str).str.extract(r"(\d+)")[0].astype("float64")
        keep &= (((issue.dt.year * 12 + issue.dt.month) + term) <= maturity_cutoff_ord())
        keep &= issue.notna()

    X = _ayh_features(raw.loc[keep, AYH_FEATURES])
    return X, y.loc[keep].astype(int), raw.loc[keep, "id"].astype(str)


# ---------------------------------------------------------------------------
# 공통 — OOF · 평가 · Sharpe
# ---------------------------------------------------------------------------
def compute_oof_local(X, y, params: dict, seed: int) -> tuple[pd.Series, list[float]]:
    """Train 안 5-fold OOF. `model.compute_oof()`와 같은 절차이나 임의 params를 받는다."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = pd.Series(np.nan, index=X.index, name="pd_oof")
    aucs: list[float] = []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = XGBClassifier(**params, random_state=seed).fit(X.iloc[tr], y.iloc[tr], verbose=False)
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof.iloc[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
        print(f"    fold {k + 1}/{N_FOLDS}  AUC={aucs[-1]:.5f}", flush=True)
    if oof.isna().any():
        raise RuntimeError("OOF에 결측이 남았다 — fold 분할을 확인하라.")
    return oof, aucs


def sharpe_for(pd_oof, p_va_raw, p_va_cal, xr, tr_idx, va_idx) -> pd.DataFrame:
    """승인선 3종 + approve-all. 분위·점수는 보정 전 PD, `E[XR]`의 `p̂`는 보정 후 (진단 C-4).

    칸별 통계표(`mu_부도`·`var_부도`)는 **Train에서만** 만들어 Validation에 적용한다 —
    재분위하지 않는다(#20).
    """
    term_tr, term_va = xr["term"].loc[tr_idx], xr["term"].loc[va_idx]
    edges = quantile_edges_by_term(pd_oof, term_tr)
    q_tr = assign_quantile_by_term(pd_oof, term_tr, edges)
    q_va = assign_quantile_by_term(p_va_raw, term_va, edges)

    st = default_cell_stats(xr["xr_default"].loc[tr_idx], q_tr, term_tr)
    mu_map, var_map = st["mu"].to_dict(), st["var"].to_dict()
    keys = list(zip(term_va, q_va))
    mu_d = pd.Series([mu_map.get(k, np.nan) for k in keys], index=va_idx)
    var_d = pd.Series([var_map.get(k, np.nan) for k in keys], index=va_idx)

    xr_va = xr.loc[va_idx]
    e_xr = expected_excess_return(p_va_cal, xr_va["xr_normal"], mu_d)
    v_xr = variance_excess_return(p_va_cal, xr_va["xr_normal"], mu_d, var_d, var_normal=0.0)
    qs = q_score(e_xr, v_xr)
    realized = xr_va["xr_realized"]

    ok = realized.notna() & e_xr.notna() & qs.notna() & p_va_raw.notna()
    return compare_ranking_criteria(
        {"pd": (p_va_raw[ok], True), "E[XR]": (e_xr[ok], False), "q_score": (qs[ok], False)},
        realized[ok],
    )


# ---------------------------------------------------------------------------
# 블록 cv — 리포트 4절
# ---------------------------------------------------------------------------
def block_cv(seed: int) -> pd.DataFrame:
    hdr("블록 cv — 6스펙 5-fold 교차검증 + Δ Sharpe (리포트 4절)")

    X_team, y, meta = build_feature_table()
    parts = split_from_manifest(X_team, y, meta)
    tr_idx = parts["train"][0].index
    va_idx = parts["validation"][0].index
    print(f"표본 {len(X_team):,}  팀 피처 {X_team.shape[1]}  "
          f"train {len(tr_idx):,}  validation {len(va_idx):,}", flush=True)

    X_ayh, y_ayh, _ = load_ayh_raw(apply_maturity_filter=True)
    if not X_ayh.index.equals(X_team.index):
        raise RuntimeError("안예환 표본 인덱스가 #16 표본과 다르다 — 필터 정의를 확인하라.")
    if not y_ayh.equals(y.astype(int)):
        raise RuntimeError("타깃 정의가 어긋난다 — 접두사 정규화를 확인하라.")
    print(f"안예환 피처 {X_ayh.shape[1]} (목록 {len(AYH_FEATURES)} + 파생)", flush=True)

    outcome = build_return_inputs()
    xr_by = {"treasury": build_excess_returns(outcome, ReturnAssumptions()),
             "cash": build_excess_returns(outcome, cash_reinvestment(ReturnAssumptions()))}

    X_nozip = X_team.drop(columns=["zip_code"])
    specs = [
        ("1 main 기본 (팀102·main파라)", X_team, params_main()),
        ("2 이지희 77d857d (팀102·ljh파라)", X_team, params_ljh()),
        ("3 안예환 fe88d6c (ayh84·ayh파라)", X_ayh, params_ayh()),
        ("4 ayh파라만 (팀102·ayh파라)", X_team, params_ayh()),
        ("5 main − zip_code (팀101)", X_nozip, params_main()),
        ("6 이지희 − zip_code (팀101)", X_nozip, params_ljh()),
    ]

    rows = []
    for name, Xf, params in specs:
        print(f"\n--- {name}  (피처 {Xf.shape[1]}) ---", flush=True)
        t0 = time.time()
        X_tr, y_tr = Xf.loc[tr_idx], y.loc[tr_idx]
        X_va, y_va = Xf.loc[va_idx], y.loc[va_idx]

        pd_oof, fold_auc = compute_oof_local(X_tr, y_tr, params, seed)
        calib = fit_calibrator(pd_oof, y_tr)
        final = XGBClassifier(**params, random_state=seed).fit(X_tr, y_tr, verbose=False)
        p_va_raw = pd.Series(final.predict_proba(X_va)[:, 1], index=X_va.index)
        p_va_cal = apply_calibrator(calib, p_va_raw)
        p_tr_in = pd.Series(final.predict_proba(X_tr)[:, 1], index=X_tr.index)

        m_raw, m_cal = calibration_metrics(y_va, p_va_raw), calibration_metrics(y_va, p_va_cal)
        auc_in, auc_oof = roc_auc_score(y_tr, p_tr_in), roc_auc_score(y_tr, pd_oof)
        row = {
            "spec": name, "n_features": Xf.shape[1],
            "cv_auc_mean": float(np.mean(fold_auc)), "cv_auc_sd": float(np.std(fold_auc, ddof=1)),
            "cv_auc_min": min(fold_auc), "cv_auc_max": max(fold_auc),
            "oof_auc": auc_oof, "auc_train_insample": auc_in, "auc_val": m_raw["auc"],
            "gap_insample": auc_in - m_raw["auc"], "gap_oof": auc_oof - m_raw["auc"],
            "ece_raw_pp": m_raw["ece_pp"], "ece_cal_pp": m_cal["ece_pp"],
            "brier_cal": m_cal["brier"],
        }
        print(f"  CV {row['cv_auc_mean']:.5f} ± {row['cv_auc_sd']:.5f}  "
              f"val {row['auc_val']:.5f}  ECE {m_raw['ece_pp']:.3f}→{m_cal['ece_pp']:.3f}%p  "
              f"Brier(보정후) {m_cal['brier']:.6f}", flush=True)

        for akey, xr in xr_by.items():
            cmp = sharpe_for(pd_oof, p_va_raw, p_va_cal, xr, tr_idx, va_idx)
            for _, r in cmp.iterrows():
                if r["criterion"] == "approve_all(대조군)":
                    continue
                row[f"dS_{akey}_{r['criterion']}"] = r["delta_sharpe"]
                row[f"approv_{akey}_{r['criterion']}"] = r["approval_rate"]
            print(f"  [{akey}] " + "  ".join(
                f"{r['criterion']}: Δ{r['delta_sharpe']:+.4f}" for _, r in cmp.iterrows()
                if r["criterion"] != "approve_all(대조군)"), flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 블록 decompose — 리포트 3절 전반
# ---------------------------------------------------------------------------
def _fit_eval(X_tr, y_tr, X_va, y_va, params, early: bool, label: str, seed: int) -> dict:
    t0 = time.time()
    p = dict(params)
    if early:
        p["early_stopping_rounds"] = 50
    m = XGBClassifier(**p, random_state=seed)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)] if early else None, verbose=False)
    p_tr, p_va = m.predict_proba(X_tr)[:, 1], m.predict_proba(X_va)[:, 1]
    bi = getattr(m, "best_iteration", None)
    row = {
        "variant": label, "n_train": len(X_tr), "n_val": len(X_va),
        "default_rate": float(y_tr.mean()), "n_features": X_tr.shape[1],
        "early_stopping": early,
        "trees_used": (bi + 1) if (early and bi is not None) else params["n_estimators"],
        "auc_train": roc_auc_score(y_tr, p_tr), "auc_val": roc_auc_score(y_va, p_va),
        "logloss_val": log_loss(y_va, p_va), "brier_val": brier_score_loss(y_va, p_va),
    }
    print(f"  [{label}] n_tr={len(X_tr):,} feat={X_tr.shape[1]} trees={row['trees_used']} "
          f"→ val AUC {row['auc_val']:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    return row


def block_decompose(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hdr("블록 decompose — 필터·분할·early stopping·피처 4단 분해 (리포트 3절)")
    params = params_ayh()
    rows = []

    # A. as-committed — #16 필터 없음 · 자체 75/25 분할 · early stopping ON
    X, y, ids = load_ayh_raw(apply_maturity_filter=False)
    print(f"완결건(#16 필터 없음): {len(X):,}  부도율 {y.mean():.4%}", flush=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    rows.append(_fit_eval(Xtr, ytr, Xva, yva, params, True, "A. as-committed", seed))

    # 팀 Test 유입량 — 리포트 2절 ②
    team = ids.map(load_split_manifest())
    ayh_split = pd.Series("ayh_train", index=ids.index)
    ayh_split.loc[Xva.index] = "ayh_validation"
    overlap = pd.crosstab(ayh_split, team.fillna("(#16 필터로 팀 표본에서 제외)"))
    n_test = int(team.eq("test").sum())
    if "test" in overlap.columns:
        used = int(overlap.loc["ayh_train", "test"])
        print(f"  ⚠️ 팀 Test {n_test:,}건 중 {used:,}건({used / n_test:.1%})이 학습에 유입", flush=True)
    del X, y, Xtr, Xva, ytr, yva

    # B·C·D — #16 필터 적용
    X, y, ids = load_ayh_raw(apply_maturity_filter=True)
    print(f"완결+만기버퍼(#16 확정): {len(X):,}  부도율 {y.mean():.4%}", flush=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    rows.append(_fit_eval(Xtr, ytr, Xva, yva, params, True, "B. +#16 필터", seed))
    rows.append(_fit_eval(Xtr, ytr, Xva, yva, params, False, "C. +ES OFF", seed))
    del Xtr, Xva, ytr, yva

    team = ids.map(load_split_manifest())
    tr, va = (team == "train").to_numpy(), (team == "validation").to_numpy()
    print(f"  매니페스트 매칭: train {tr.sum():,}  validation {va.sum():,}  "
          f"미매칭 {int(team.isna().sum()):,}", flush=True)
    rows.append(_fit_eval(X[tr], y[tr], X[va], y[va], params, False, "D. +팀 분할", seed))
    del X, y

    # E — 팀 102피처
    Xs, ys, _, sp = load_shared("trainval")
    rows.append(_fit_eval(Xs[sp == "train"], ys[sp == "train"],
                          Xs[sp == "validation"], ys[sp == "validation"],
                          params, False, "E. +팀 피처", seed))

    out = pd.DataFrame(rows)
    out["delta_vs_prev"] = out["auc_val"].diff()
    return out, overlap


# ---------------------------------------------------------------------------
# 블록 ablation — 리포트 3절 후반
# ---------------------------------------------------------------------------
def block_ablation(seeds: list[int]) -> pd.DataFrame:
    hdr("블록 ablation — 팀102 중 20개 그룹별 기여 (리포트 3절)")
    print("⚠️ 하이퍼파라미터는 **안예환 것**을 쓴다 — '그의 스펙에서 무엇이 이득인가'가 질문이다.\n"
          "   main 파라미터 기준 동일 효과는 블록 cv의 스펙1 → 스펙5에서 확인된다.", flush=True)

    X, y, _, sp = load_shared("trainval")
    tr, va = sp == "train", sp == "validation"
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]
    params = params_ayh()

    groups = {
        "(기준) 팀102 전체": [],
        "− zip_code": ["zip_code"],
        "− sec_app_* (11)": SEC_APP,
        "− joint (3)": JOINT,
        "− policy_code": ["policy_code"],
        "− 희소연체카운터 (4)": RARE_DELINQ,
        "− 20개 전부 (=ayh)": ["zip_code", *SEC_APP, *JOINT, "policy_code", *RARE_DELINQ],
    }

    rows = []
    for label, drop in groups.items():
        A_tr = X_tr.drop(columns=drop) if drop else X_tr
        A_va = X_va.drop(columns=drop) if drop else X_va
        aucs = [
            roc_auc_score(y_va, XGBClassifier(**params, random_state=s)
                          .fit(A_tr, y_tr, verbose=False).predict_proba(A_va)[:, 1])
            for s in seeds
        ]
        rows.append({"variant": label, "n_features": A_tr.shape[1],
                     "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)),
                     "auc_min": min(aucs), "auc_max": max(aucs)})
        print(f"  [{label}] feat={A_tr.shape[1]} → AUC {np.mean(aucs):.5f} "
              f"(sd {np.std(aucs, ddof=1):.5f}, n_seed={len(seeds)})", flush=True)

    out = pd.DataFrame(rows)
    out["delta_vs_base"] = out["auc_mean"] - out.loc[0, "auc_mean"]
    return out


# ---------------------------------------------------------------------------
# 블록 seed — 리포트 4절 보강
# ---------------------------------------------------------------------------
def block_seed(seeds: list[int]) -> pd.DataFrame:
    hdr("블록 seed — 학습 seed 흔들림 + 페어드 부트스트랩 (리포트 4절)")
    X, y, _, sp = load_shared("trainval")
    X_tr, y_tr = X[sp == "train"], y[sp == "train"]
    X_va, y_va = X[sp == "validation"], y[sp == "validation"]
    yv = y_va.to_numpy(dtype="int8")

    preds, rows = {}, []
    for name, params in (("main", params_main()), ("ljh", params_ljh())):
        for s in seeds:
            p = XGBClassifier(**params, random_state=s).fit(
                X_tr, y_tr, verbose=False).predict_proba(X_va)[:, 1]
            rows.append({"model": name, "seed": s, "auc_val": roc_auc_score(yv, p)})
            print(f"  {name:5s} seed={s:<9} val AUC={rows[-1]['auc_val']:.5f}", flush=True)
            if s == seeds[0]:
                preds[name] = p

    tbl = pd.DataFrame(rows)
    print("\n모델별 흔들림:\n" +
          tbl.groupby("model")["auc_val"].agg(["mean", "std", "min", "max"]).to_string())

    rng = np.random.default_rng(seeds[0])
    n = len(yv)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        yb = yv[i]
        d[b] = (np.nan if yb.sum() in (0, len(yb))
                else roc_auc_score(yb, preds["ljh"][i]) - roc_auc_score(yb, preds["main"][i]))
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"\n페어드 부트스트랩 ΔAUC(ljh − main), 동일 Validation, B={len(d)}")
    print(f"  관측 Δ {roc_auc_score(yv, preds['ljh']) - roc_auc_score(yv, preds['main']):+.5f}"
          f"   95% CI [{lo:+.5f}, {hi:+.5f}]   양측 p≈{p_two:.3f}")
    return tbl


# ---------------------------------------------------------------------------
# 블록 zipdiag — 리포트 5절
# ---------------------------------------------------------------------------
def block_zipdiag() -> tuple[pd.DataFrame, pd.DataFrame]:
    hdr("블록 zipdiag — zip_code 암기 진단 (리포트 5절, 학습 없음)")
    X, y, _, sp = load_shared("trainval")
    tr, va = sp == "train", sp == "validation"
    base = float(y[tr].mean())

    rows, examples = [], pd.DataFrame()
    for col in ("zip_code", "grade", "addr_state", "purpose"):
        a = pd.DataFrame({"k": X.loc[tr, col].astype(str), "y": y[tr]}).groupby("k")["y"].agg(["mean", "size"])
        b = pd.DataFrame({"k": X.loc[va, col].astype(str), "y": y[va]}).groupby("k")["y"].agg(["mean", "size"])
        j = a.join(b, lsuffix="_tr", rsuffix="_va", how="inner").dropna()

        # 칸이 전부 기저 부도율이라도 표본 때문에 생기는 sd
        noise = float(np.mean(np.sqrt(base * (1 - base) / j["size_tr"])))
        obs = float(j["mean_tr"].std(ddof=1))
        w = j["size_va"]
        mt, mv = np.average(j["mean_tr"], weights=w), np.average(j["mean_va"], weights=w)
        r = (np.average((j["mean_tr"] - mt) * (j["mean_va"] - mv), weights=w)
             / np.sqrt(np.average((j["mean_tr"] - mt) ** 2, weights=w)
                       * np.average((j["mean_va"] - mv) ** 2, weights=w)))
        # 관측분산 = 신호분산 + 잡음분산
        signal = float(np.sqrt(max(obs ** 2 - noise ** 2, 0.0)))
        rows.append({"variable": col, "n_categories": len(j),
                     "median_n_train": float(j["size_tr"].median()),
                     "observed_sd": obs, "noise_sd": noise, "signal_sd": signal,
                     "observed_over_noise": obs / noise, "corr_train_val": float(r)})
        print(f"  {col:12s} 범주 {len(j):>4,}  칸당중위 {j['size_tr'].median():>7.0f}  "
              f"실측sd {obs:.4f}  잡음sd {noise:.4f}  비율 {obs / noise:.2f}배  r={r:.3f}", flush=True)

        if col == "zip_code":
            big = j[j["size_tr"] >= 300]
            examples = big.nlargest(5, "mean_tr").reset_index().rename(columns={"k": "zip_code"})
            examples["base_default_rate"] = base
            print(f"\n  Train 부도율 상위 5개 (300건 이상) — 전체 평균 {base:.1%}")
            for _, e in examples.iterrows():
                print(f"    {e['zip_code']}: Train {e['mean_tr']:.1%} (n={e['size_tr']:.0f})"
                      f"  →  Validation {e['mean_va']:.1%} (n={e['size_va']:.0f})", flush=True)

    return pd.DataFrame(rows), examples


# ---------------------------------------------------------------------------
# 블록 stability — `zip_code` 제외 결정의 K회 반복 안정성 (#18)
# ---------------------------------------------------------------------------
def parse_seeds(spec: str) -> list[int]:
    """`"0-24"` · `"0-9,20,30-32"` → seed 목록. 구간이 겹치면 중복을 제거한다."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def block_stability(seed_list: list[int], out_path: Path) -> pd.DataFrame:
    """Train+Validation 풀을 seed마다 재분할해 `τ*`·Δ Sharpe의 **분포**를 본다 (#18).

    Validation 하나로 잰 Δ Sharpe는 표본에 따라 흔들리는 추정치다. `main`과
    `main − zip_code`를 **같은 재분할**에 함께 돌려 짝지은 비교(paired)를 만든다 —
    seed별 차이의 분포를 보면 +0.015가 우연인지 구조인지 갈린다.

    ⚠️ Test는 `resplit_train_validation()`이 고정한다 — 재분할되는 건 비Test 80%뿐이다.
    ⚠️ 칸별 통계표는 **seed마다 다시 만든다** (`sharpe_optimizer.scores_for_assumptions`
       docstring의 잠정 확정 — seed 하나로 고정하면 다른 seed의 Validation이 섞여 누수다).

    반복마다 CSV에 덧붙이고 이미 있는 `(spec, seed)`는 건너뛴다 — **구간을 나눠 여러 번에
    걸쳐 돌릴 수 있다.** seed 집합이 같으면 한 번에 돌린 것과 결과가 같다.
    """
    hdr(f"블록 stability — seed {seed_list[0]}~{seed_list[-1]} ({len(seed_list)}개) 재분할 반복")
    print(f"산출(덧붙임) → {out_path.name}", flush=True)

    X_team, y, meta = build_feature_table()
    outcome = build_return_inputs()
    xr_by = {"treasury": build_excess_returns(outcome, ReturnAssumptions()),
             "cash": build_excess_returns(outcome, cash_reinvestment(ReturnAssumptions()))}
    variants = {
        "main": X_team,
        "main_nozip": X_team.drop(columns=["zip_code"]),
    }
    params = params_main()

    done = completed_keys(out_path, ("spec", "seed"))
    if done:
        print(f"  이미 계산된 (spec, seed) {len(done)}쌍 — 건너뛴다", flush=True)

    for seed in seed_list:
        if all((name, str(seed)) in done for name in variants):
            print(f"\nseed {seed}: 전부 계산됨 — 건너뜀", flush=True)
            continue

        t0 = time.time()
        parts = resplit_train_validation(X_team, y, meta, seed=seed)
        tr_idx = parts["train"][0].index
        va_idx = parts["validation"][0].index
        print(f"\nseed {seed}  train {len(tr_idx):,}  validation {len(va_idx):,}", flush=True)

        for name, Xf in variants.items():
            if (name, str(seed)) in done:
                print(f"  [{name}] 건너뜀", flush=True)
                continue
            X_tr, y_tr = Xf.loc[tr_idx], y.loc[tr_idx]
            X_va = Xf.loc[va_idx]

            pd_oof, _ = compute_oof_local(X_tr, y_tr, params, seed)
            calib = fit_calibrator(pd_oof, y_tr)
            final = XGBClassifier(**params, random_state=seed).fit(X_tr, y_tr, verbose=False)
            p_va_raw = pd.Series(final.predict_proba(X_va)[:, 1], index=X_va.index)
            p_va_cal = apply_calibrator(calib, p_va_raw)

            for akey, xr in xr_by.items():
                term_tr, term_va = xr["term"].loc[tr_idx], xr["term"].loc[va_idx]
                edges = quantile_edges_by_term(pd_oof, term_tr)
                q_tr = assign_quantile_by_term(pd_oof, term_tr, edges)
                q_va = assign_quantile_by_term(p_va_raw, term_va, edges)
                st = default_cell_stats(xr["xr_default"].loc[tr_idx], q_tr, term_tr)
                mu_map, var_map = st["mu"].to_dict(), st["var"].to_dict()
                keys = list(zip(term_va, q_va))
                mu_d = pd.Series([mu_map.get(k, np.nan) for k in keys], index=va_idx)
                var_d = pd.Series([var_map.get(k, np.nan) for k in keys], index=va_idx)

                xr_va = xr.loc[va_idx]
                e_xr = expected_excess_return(p_va_cal, xr_va["xr_normal"], mu_d)
                v_xr = variance_excess_return(p_va_cal, xr_va["xr_normal"], mu_d, var_d,
                                              var_normal=0.0)
                qs = q_score(e_xr, v_xr)
                realized = xr_va["xr_realized"]
                ok = realized.notna() & e_xr.notna() & qs.notna() & p_va_raw.notna()

                for crit, (score, lower) in {
                    "pd": (p_va_raw[ok], True),
                    "E[XR]": (e_xr[ok], False),
                    "q_score": (qs[ok], False),
                }.items():
                    res = find_optimal_threshold(score, realized[ok], lower)
                    append_row_csv(out_path, {"spec": name, "seed": seed, "reinvest": akey,
                                              "criterion": crit, **res})
                    if crit == "q_score":
                        print(f"  [{name}·{akey}] q_score τ*={res['threshold']:.4f} "
                              f"승인율 {res['approval_rate']:.1%} Δ{res['delta_sharpe']:+.4f}",
                              flush=True)
        print(f"  (seed {seed} {time.time() - t0:.0f}s)", flush=True)

    return pd.read_csv(out_path)


def summarize_stability(df: pd.DataFrame) -> None:
    """seed별 Δ Sharpe 분포와 **짝지은 차이**(nozip − main)를 요약한다."""
    for akey in sorted(df["reinvest"].unique()):
        for crit in ("q_score", "E[XR]", "pd"):
            sub = df[(df["reinvest"] == akey) & (df["criterion"] == crit)]
            piv = sub.pivot_table(index="seed", columns="spec", values="delta_sharpe")
            if not {"main", "main_nozip"} <= set(piv.columns):
                continue
            piv = piv.dropna()
            d = piv["main_nozip"] - piv["main"]
            print(f"\n[{akey} · {crit}]  K={len(piv)}")
            print(f"  main       Δ Sharpe  평균 {piv['main'].mean():.4f}  "
                  f"sd {piv['main'].std(ddof=1):.4f}  중앙값 {piv['main'].median():.4f}")
            print(f"  main_nozip Δ Sharpe  평균 {piv['main_nozip'].mean():.4f}  "
                  f"sd {piv['main_nozip'].std(ddof=1):.4f}  중앙값 {piv['main_nozip'].median():.4f}")
            print(f"  짝지은 차이(nozip−main) 평균 {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
                  f"nozip 승 {int((d > 0).sum())}/{len(d)}")
            if len(d) > 1:
                se = d.std(ddof=1) / np.sqrt(len(d))
                print(f"    표준오차 {se:.5f} → 평균/표준오차 = {d.mean() / se:.1f}배  "
                      f"95% CI [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]")
            tau = sub.pivot_table(index="seed", columns="spec", values="threshold").dropna()
            print(f"  τ* 중앙값  main {tau['main'].median():.4f}  "
                  f"nozip {tau['main_nozip'].median():.4f}  "
                  f"(sd {tau['main'].std(ddof=1):.4f} / {tau['main_nozip'].std(ddof=1):.4f})")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only",
                    choices=["cv", "decompose", "ablation", "seed", "zipdiag", "stability"],
                    action="append", help="블록 선택 (반복 가능). 생략하면 stability 제외 전부.")
    ap.add_argument("--seed", type=int, default=None, help="기본값: config.yaml의 seed")
    ap.add_argument("--seeds", default="0-24",
                    help="stability 블록의 재분할 seed 구간 (예: '0-24', '25-49', '0-9,20').")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else load_config().random_seed.default
    # stability는 수십 분~수 시간이라 기본 실행에 넣지 않는다 — 명시해야 돈다.
    blocks = args.only or ["cv", "decompose", "ablation", "seed", "zipdiag"]
    seeds = [seed, *SEED_SWEEP]
    o = out_dir()
    t0 = time.time()
    print(f"seed={seed}  블록={blocks}  산출={o}", flush=True)

    if "zipdiag" in blocks:
        diag, ex = block_zipdiag()
        diag.to_csv(o / "model_comparison_zip_diagnostic.csv", index=False)
        ex.to_csv(o / "model_comparison_zip_examples.csv", index=False)
    if "ablation" in blocks:
        block_ablation(seeds[:3]).to_csv(o / "model_comparison_zip_ablation.csv", index=False)
    if "seed" in blocks:
        block_seed(seeds).to_csv(o / "model_comparison_seed_stability.csv", index=False)
    if "decompose" in blocks:
        dec, overlap = block_decompose(seed)
        dec.to_csv(o / "model_comparison_ayh_decompose.csv", index=False)
        overlap.to_csv(o / "model_comparison_ayh_test_overlap.csv")
    if "cv" in blocks:
        block_cv(seed).to_csv(o / "model_comparison_crossvalidation.csv", index=False)
    if "stability" in blocks:
        df = block_stability(parse_seeds(args.seeds), o / "model_comparison_stability.csv")
        hdr("stability 요약 — Δ Sharpe 분포와 짝지은 차이")
        summarize_stability(df)

    print(f"\n완료 — {time.time() - t0:.0f}s. 리포트: outputs/reports/model_comparison_kgj.md")


if __name__ == "__main__":
    main()
