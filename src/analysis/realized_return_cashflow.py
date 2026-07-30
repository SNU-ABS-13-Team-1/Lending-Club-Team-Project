"""**실현 현금흐름 기반 수익률** — B팀(유명곤) 명세의 재현 구현.

`realized_return.py`의 `contract_return()`은 **계약대로** 만기까지 갚는다고 보고 `R`을
계산한다. 그런데 실측하면 **Fully Paid 36개월 대출의 66.03%가 만기 전에 상환**됐고
(K 중앙값 28개월), 그만큼 `R`이 과대추정된다 — 36m **+1.06%p** / 60m **+2.24%p**.
`decision_log.md` #20이 "B팀 1순위"로 남긴 조기상환 보정이 바로 이 항목이다.

B팀이 건별 실제 현금흐름을 재구성해 산출물을 냈고
(`lending_club_realized_return_rf_xr_eligible_2020-10.csv`, 722,353건), 이 모듈은 **같은
계산을 코드로 재현**한다.

## 왜 CSV를 그냥 쓰지 않고 재현하는가

B팀 CSV는 **train 원본 id만** 담고 있다. 최종 Test로 쓰는
`lending_club_2020_test_2nd.csv`(481,833건)에 대응하는 행이 **한 건도 없어서**, CSV만으로는
Test에서 `XR`을 계산할 수 없다. 그래서 입력 CSV를 갈아 끼울 수 있는 형태로 구현하고,
train에 대해서는 **B팀 CSV와 대조해 일치를 확인**한다(`verify_against_teamb()`).

## 계산 규칙 (B팀 노트북 `realized_return_preprocessing_step_by_step.ipynb`)

월별 납입 내역이 원본에 없으므로 총액을 다음처럼 배치한다.

| `K` (발행 → 최종납입 개월) | 정규 현금흐름 배치 |
| --- | --- |
| `K = 0` | 전액을 `t=0`에 |
| `K = 1` | 전액을 `t=1`에 |
| `K ≥ 2` | `C_regular − L`을 `t=1..K−1`에 균등, 마지막 납입액 `L`을 `t=K`에 |

- 순Recovery `C_recovery = recoveries − collection_recovery_fee`는 `t = K+6`에 일시 배치.
- 정상 납입이 없는 Recovery-only `Charged Off`는 `t=6`에 배치한다(**544건**).

**재투자·역할인은 `GS1M` 실제 경로**를 쓴다 — 수령월에는 이자를 안 주고 **다음 달부터**
계약만기까지 굴리며, 만기 이후 수령분(회수금)은 같은 경로로 만기 시점까지 **역할인**한다.

    W(T) = Σ_m CF_m · exp(Λ_만기 − Λ_m),   Λ = 누적 로그성장률
    R    = (W(T) / funded_amnt)^(12/T) − 1

⚠️ **이 재투자 기준은 `contract_return()`과 다르다.** 저쪽은 ⓒ발행시점 고정 국채
(`rf` = GS3/GS5)를 쓰고, 여기는 ⓐ실제경로(GS1M)다 — #20 미확정 3건 중 "국채 금리 기준"에
해당한다. **`rf` 자체(초과수익률의 차감항)는 양쪽 모두 발행시점 × 만기매칭 GS3/GS5로
동일하다**(#18 확정). 실측 대조에서 부도 건 `XR`은 두 방식이 36m −0.0002 / 60m +0.0034로
거의 같았다 — 차이는 대부분 조기상환에서 온다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.config import load_config
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import load_config


GS1M_FILENAME = "us_treasury_GS1M_monthly_2007-07_to_2025-09.csv"

#: 회수금 수령 시점 — 최종납입 K개월 뒤 6개월(`realized_return.RECOVERY_LAG_MONTHS`와 같은 값).
RECOVERY_LAG_MONTHS = 6

#: 원본에서 읽어야 하는 컬럼. 전부 **사후(post-approval)** 라 피처로 쓰지 않는다.
CASHFLOW_COLUMNS = [
    "id", "loan_status", "funded_amnt", "term", "issue_d", "last_pymnt_d",
    "total_pymnt", "last_pymnt_amnt", "recoveries", "collection_recovery_fee",
    "installment",
]

#: B팀 산출물의 검증 기대값 (train 원본 기준).
TEAMB_EXPECTED = {"eligible": 722_353, "excluded": 1_210, "recovery_only": 544}


# ---------------------------------------------------------------------------
# GS1M 월수익률 경로
# ---------------------------------------------------------------------------
def load_gs1m_monthly_return() -> pd.Series:
    """`GS1M` → 월 등가수익률. 인덱스는 절대 월 서수(`year*12 + month`).

    FRED `GS1M`은 **bond-equivalent yield(%)** 라 12로 나누지 않는다. 반년 복리 관례를
    풀어 `(1 + GS1M/200)^(1/6) − 1`로 월율을 만든다(B팀 명세).
    """
    path = load_config().paths.data_processed / GS1M_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"GS1M 데이터가 없습니다: {path}\n"
            "`python src/preprocessing/fetch_treasury_gs1m.py`로 받으세요."
        )
    g = pd.read_csv(path, parse_dates=["observation_date"])
    if g["GS1M"].isna().any():
        raise ValueError("GS1M에 결측이 있습니다 — 수집 스크립트를 다시 확인하세요.")
    idx = g["observation_date"].dt.year * 12 + g["observation_date"].dt.month
    out = pd.Series(
        (1.0 + g["GS1M"].to_numpy() / 200.0) ** (1.0 / 6.0) - 1.0,
        index=pd.Index(idx, name="month_ord"), name="monthly_return",
    ).sort_index()
    if not out.index.equals(pd.RangeIndex(out.index[0], out.index[-1] + 1)):
        raise ValueError("GS1M 월 서수가 연속이 아닙니다 — 빠진 달이 있습니다.")
    return out


# ---------------------------------------------------------------------------
# 현금흐름 스케줄
# ---------------------------------------------------------------------------
def build_cashflow_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """원본 행 → 현금흐름 스케줄 + 계산 가능 여부.

    `filter_analysis_sample()`을 **이미 통과한** 프레임을 받는다 — 만기+버퍼 6개월과
    `Fully Paid`/`Charged Off` 필터는 거기서 걸린다. 여기서 추가로 거르는 것은
    **현금흐름을 배치할 수 없는 행**뿐이며, 사유를 `exclude_reason`에 남긴다
    (버리지 않고 감사 테이블로 보존한다 — `src/preprocessing/AGENTS.md`).
    """
    d = raw.copy()
    num = ["funded_amnt", "total_pymnt", "last_pymnt_amnt", "recoveries",
           "collection_recovery_fee", "installment"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["status_norm"] = (
        d["loan_status"].astype("string")
        .str.replace(r"^Does not meet the credit policy\. Status:\s*", "", regex=True)
        .str.strip()
    )
    # `term`·`issue_d`는 이미 파싱된 프레임(`build_return_inputs()`의 반환)으로도 들어온다.
    d["term_months"] = (
        pd.to_numeric(d["term"], errors="coerce")
        if pd.api.types.is_numeric_dtype(d["term"])
        else pd.to_numeric(d["term"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
    )
    if pd.api.types.is_numeric_dtype(d["issue_d"]):
        issue_ord = pd.to_numeric(d["issue_d"], errors="coerce")
        issue_missing = issue_ord.isna()
    else:
        issue = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce")
        issue_ord = issue.dt.year * 12 + issue.dt.month
        issue_missing = issue.isna()
    last = pd.to_datetime(d["last_pymnt_d"], format="%b-%Y", errors="coerce")
    d["issue_month_ord"] = issue_ord
    last_ord = last.dt.year * 12 + last.dt.month

    d["K"] = (last_ord - d["issue_month_ord"]).clip(lower=0)
    d["C_regular"] = d["total_pymnt"] - d["recoveries"]
    d["C_recovery"] = d["recoveries"] - d["collection_recovery_fee"]
    d["L"] = d["last_pymnt_amnt"]

    # --- 제외 사유 (여러 개면 '|'로 잇는다) ---
    reason = pd.Series("", index=d.index, dtype="object")

    def flag(mask: pd.Series, text: str) -> None:
        m = mask.fillna(False).to_numpy()
        reason.values[m] = np.where(
            reason.values[m] == "", text, reason.values[m] + "|" + text
        )

    required = ["funded_amnt", "total_pymnt", "recoveries",
                "collection_recovery_fee", "last_pymnt_amnt"]
    flag(d[required].isna().any(axis=1), "필수금액결측")
    flag(d["funded_amnt"] <= 0, "funded_amnt_0이하")
    flag(~d["term_months"].isin([36, 60]), "term_오류")
    flag(issue_missing, "issue_d_결측")

    no_last = last.isna()
    d["zero_cash_R_minus1"] = (
        d["status_norm"].eq("Charged Off") & no_last
        & d["total_pymnt"].eq(0) & d["recoveries"].eq(0)
    )
    d["recovery_only_no_payment"] = (
        d["status_norm"].eq("Charged Off") & no_last
        & d["C_regular"].abs().le(0.01) & d["recoveries"].gt(0)
        & d["L"].abs().le(0.01)
    )
    flag(no_last & ~(d["zero_cash_R_minus1"] | d["recovery_only_no_payment"]),
         "last_pymnt_d_결측_시점미확정")
    flag(d["L"] < 0, "last_pymnt_amnt_음수")
    flag(d["K"].ge(2) & (d["L"] > d["C_regular"] + 1e-6), "L이_C_regular_초과")
    flag(d["C_regular"] < -0.01, "C_regular_음수")
    flag(d["C_recovery"] < -0.01, "C_recovery_음수")

    d["exclude_reason"] = reason
    d["cashflow_eligible"] = reason.eq("")

    # --- 배치 ---
    K = d["K"].round()
    k_ge2 = K.ge(2).fillna(False)
    d["K_effective"] = K
    d["regular_early_count"] = np.where(k_ge2, K - 1, 0)
    d["regular_early_each"] = np.where(
        k_ge2, (d["C_regular"] - d["L"]) / np.where(k_ge2, K - 1, 1), 0.0
    )
    d["regular_final_t"] = K
    d["regular_final_amount"] = np.where(k_ge2, d["L"], d["C_regular"])
    d.loc[d["zero_cash_R_minus1"], "regular_final_amount"] = 0.0
    d["recovery_t"] = (K + RECOVERY_LAG_MONTHS)
    d.loc[d["recovery_only_no_payment"], "recovery_t"] = RECOVERY_LAG_MONTHS
    d["recovery_amount"] = d["C_recovery"]

    d["cashflow_rule"] = np.select(
        [d["zero_cash_R_minus1"], d["recovery_only_no_payment"],
         K.eq(0).fillna(False), K.eq(1).fillna(False), k_ge2],
        ["현금유입 없음", "Recovery-only (발행+6)", "K=0", "K=1", "K>=2"],
        default="확인 필요",
    )
    return d


# ---------------------------------------------------------------------------
# W(T)와 R
# ---------------------------------------------------------------------------
def realized_return_actual(
    schedule: pd.DataFrame, gs1m: pd.Series | None = None
) -> pd.DataFrame:
    """현금흐름 스케줄 → `W_T`·`R` (**계약만기 등가 연율**).

    `GS1M` 누적 로그성장률의 prefix를 미리 만들어 **반복문 없이** 벡터로 계산한다.
    수령월에는 이자를 주지 않고 다음 달부터 굴린다 — 계약만기 이후 수령분은
    같은 경로로 만기 시점까지 역할인되므로 계수가 1보다 작아진다.

    `cashflow_eligible`이 아닌 행은 `W_T`·`R`이 NaN이다.
    """
    g = load_gs1m_monthly_return() if gs1m is None else gs1m
    ok = schedule["cashflow_eligible"].to_numpy()
    d = schedule.loc[ok]

    rate_min = int(g.index.min())
    # prefix[b] = 시작월 직전부터 b번째 달까지의 누적 로그성장률. prefix[0] = 0.
    prefix = np.concatenate([[0.0], np.cumsum(np.log1p(g.to_numpy()))])
    inv_cumsum = np.cumsum(np.exp(-prefix))

    issue_b = (d["issue_month_ord"].astype("int64").to_numpy() - rate_min + 1)
    term = d["term_months"].astype("int64").to_numpy()
    k = d["K_effective"].fillna(0).astype("int64").to_numpy()
    rec_t = d["recovery_t"].fillna(0).astype("int64").to_numpy()
    mat_b = issue_b + term

    bounds = np.concatenate([issue_b, mat_b, issue_b + k, issue_b + rec_t])
    if bounds.min() < 0 or bounds.max() >= len(prefix):
        raise ValueError(
            "GS1M 경로가 대출 기간을 덮지 못합니다 — 수집 구간을 확인하세요 "
            f"(필요 월 서수 {bounds.min() + rate_min - 1}~{bounds.max() + rate_min - 1})."
        )

    early_n = d["regular_early_count"].to_numpy()
    m = early_n > 0
    # t=1..K−1 각 달의 계수 합 = exp(Λ_만기) · Σ_{j} exp(−Λ_j)
    sum_early = np.zeros(len(d))
    sum_early[m] = np.exp(prefix[mat_b[m]]) * (
        inv_cumsum[issue_b[m] + k[m] - 1] - inv_cumsum[issue_b[m]]
    )

    final_factor = np.exp(prefix[mat_b] - prefix[issue_b + k])
    rec_factor = np.exp(prefix[mat_b] - prefix[issue_b + rec_t])

    W = (d["regular_early_each"].to_numpy() * sum_early
         + d["regular_final_amount"].to_numpy() * final_factor
         + d["recovery_amount"].to_numpy() * rec_factor)
    P = d["funded_amnt"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.power(W / P, 12.0 / term) - 1.0

    out = pd.DataFrame(
        {"W_T": np.nan, "R": np.nan, "regular_final_factor": np.nan,
         "recovery_factor": np.nan},
        index=schedule.index,
    )
    out.loc[ok, "W_T"] = W
    out.loc[ok, "R"] = R
    out.loc[ok, "regular_final_factor"] = final_factor
    out.loc[ok, "recovery_factor"] = rec_factor
    return out


def build_realized_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """`filter_analysis_sample()` 통과 프레임 → 실현 `R` + `rf` + `XR` 한 장.

    `rf`는 **발행시점 × 만기매칭 GS3/GS5**다(#18 확정) — 재투자에 쓴 GS1M과 역할이 다르다.
    `GS1M`은 현금을 굴리는 이율, `rf`는 초과수익률의 차감 기준이다.
    """
    from analysis.realized_return import issue_risk_free_rate

    sched = build_cashflow_schedule(raw)
    ret = realized_return_actual(sched)
    rf = issue_risk_free_rate(sched["issue_month_ord"], sched["term_months"])

    return pd.DataFrame({
        "id": sched["id"].astype(str).str.strip(),
        "status_norm": sched["status_norm"],
        "term": sched["term_months"],
        "issue_month_ord": sched["issue_month_ord"],
        "K": sched["K"],
        "funded_amnt": sched["funded_amnt"],
        "installment": sched["installment"],
        "cashflow_rule": sched["cashflow_rule"],
        "cashflow_eligible": sched["cashflow_eligible"],
        "exclude_reason": sched["exclude_reason"],
        "W_T": ret["W_T"],
        "R": ret["R"],
        "rf": rf,
        "XR": ret["R"] - rf,
        "is_default": sched["status_norm"].eq("Charged Off").astype("int8"),
    })


def load_raw_for_cashflow(csv_path: Path | None = None) -> pd.DataFrame:
    """원본 CSV에서 현금흐름 계산에 필요한 열만 읽고 분석 표본 필터를 건다.

    `csv_path`를 주면 2nd Test 등 다른 파일을 읽는다(건수 검증은 자동으로 꺼진다).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.loader import filter_analysis_sample, load_raw_loans

    raw = load_raw_loans(usecols=CASHFLOW_COLUMNS, csv_path=csv_path)
    return filter_analysis_sample(raw, verify=csv_path is None)


# ---------------------------------------------------------------------------
# B팀 산출물과의 대조
# ---------------------------------------------------------------------------
def verify_against_teamb(ours: pd.DataFrame, teamb_csv: Path) -> pd.DataFrame:
    """우리 재현값과 B팀 CSV를 `id`로 맞춰 대조한다.

    이 검증이 통과해야 2nd Test에 같은 코드를 돌린 결과를 믿을 수 있다 —
    **train에서 일치를 확인한 뒤에만** Test로 넘어간다.
    """
    b = pd.read_csv(teamb_csv, usecols=["id", "R", "rf", "XR", "W_T", "K", "cashflow_rule"],
                    low_memory=False)
    b["id"] = b["id"].astype(str).str.strip()
    m = ours.merge(b, on="id", how="inner", suffixes=("", "_b"))
    rows = []
    for col in ("R", "rf", "XR", "W_T"):
        diff = (m[col] - m[f"{col}_b"]).abs()
        rows.append({
            "column": col, "n": int(diff.notna().sum()),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "n_over_1e-6": int((diff > 1e-6).sum()),
        })
    rows.append({
        "column": "cashflow_rule", "n": len(m),
        "max_abs_diff": float("nan"), "mean_abs_diff": float("nan"),
        "n_over_1e-6": int((m["cashflow_rule"] != m["cashflow_rule_b"]).sum()),
    })
    return pd.DataFrame(rows)
