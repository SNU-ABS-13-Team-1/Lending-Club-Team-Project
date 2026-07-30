"""
실현수익률(HPR) 산출 — 이슈 #15
잠정(provisional) · 0% 재투자 가정 · 민감도 비교용

decision_log.md #18에서 "0% 재투자(HPR) 결과를 민감도로 병기한다"고 확정된 항목.
본체(재투자 반영 수익률)는 유명곤·류성환님이 별도 산출. 이 스크립트는 그 비교 대상.

공식:
  HPR_i = (Total Amount Received_i - Loan Amount_i) / Loan Amount_i
  r_i   = (1 + HPR_i)^(12/T_i) - 1,  T_i = 약정만기(36/60개월)
  0% 재투자 가정 (암묵), IRR 미사용, 상환·부도 전 건 동일 수식

표본 정의 (t2_contribution_reassessment.py와 동일 — 팀 기준):
  1) loan_status가 Fully Paid / Charged Off인 건만
     (credit policy 접두사는 제거 후 판정, Current·Late·Default 등 미확정 상태 제외)
  2) 만기+버퍼 6개월: issue_d + term <= 2020-04 (월 단위 Period 계산)
  → PD 모형 학습 모집단 723,563건 (decision_log #16)
건수 기준에 대한 주의:
  decision_log #16은 실현수익률 산출 모집단을 721,809건(계산 불가 1,754건 제외)으로
  구분하나, 본 스크립트는 723,563건 전체 기준으로 산출한다. 본 스크립트의 입력(total_pymnt·loan_amnt)에는 결측이 없어 전 건 계산 가능했다.
  감사 테이블 로직은 유지한다(이번 실행에서는 0건).
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "lending_club_2020_train.csv"
OUT_MAIN = ROOT / "data" / "processed" / "hpr_0pct_reinvest_sensitivity_full.csv"
OUT_AUDIT = ROOT / "data" / "processed" / "hpr_0pct_reinvest_excluded_audit_full.csv"

# 1. 원본 읽기
cols = ["id", "issue_d", "term", "loan_amnt", "total_pymnt", "loan_status"]
df = pd.read_csv(RAW, usecols=cols, low_memory=False)
print(f"원본: {len(df):,}행")

# 2. 상태 필터: Fully Paid / Charged Off만 (팀 정의와 동일)
POLICY_PREFIX = "Does not meet the credit policy. Status:"
status = df["loan_status"].astype(str).str.replace(POLICY_PREFIX, "", regex=False)
keep = status.isin(["Fully Paid", "Charged Off"])
df = df[keep].copy()
df["loan_status"] = status[keep]
print(f"상태 필터(Fully Paid/Charged Off만): {len(df):,}행")

# 3. 만기+버퍼 필터: issue_d + term <= 2020-04 (월 단위 Period — 팀 코드와 동일 방식)
df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
df["term_months"] = pd.to_numeric(
    df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
)
n_bad_term = int(df["term_months"].isna().sum() + df["issue_dt"].isna().sum())
if n_bad_term > 0:
    print(f"issue_d/term 결측(만기 계산 불가, 제외): {n_bad_term:,}행")
df = df[df["term_months"].notna() & df["issue_dt"].notna()].copy()
df["term_months"] = df["term_months"].astype(int)
df["maturity"] = df["issue_dt"].dt.to_period("M") + df["term_months"]
MATURITY_CUTOFF = pd.Period("2020-04", freq="M")
sample = df[df["maturity"] <= MATURITY_CUTOFF].copy()
print(f"PD 모형 학습 모집단: {len(sample):,}행 (기대값 723,563)")

# 4. 계산 가능 여부 판정 — R 계산에 필요한 금액이 온전한 건만
sample["exclude_reason"] = None
sample.loc[sample["total_pymnt"].isna(), "exclude_reason"] = "total_pymnt 결측"
sample.loc[sample["loan_amnt"].isna(), "exclude_reason"] = "loan_amnt 결측"
sample.loc[sample["loan_amnt"] <= 0, "exclude_reason"] = "loan_amnt 0 이하"

calculable = sample[sample["exclude_reason"].isna()].copy()
excluded = sample[sample["exclude_reason"].notna()].copy()
print(f"실현수익률 산출 모집단(계산 가능): {len(calculable):,}행 (기대값 721,809)")
print(f"계산 불가(감사 테이블로 보존): {len(excluded):,}행 (기대값 1,754)")
if len(excluded) > 0:
    print(excluded["exclude_reason"].value_counts())

# 5. HPR 계산
calculable["hpr"] = (
    (calculable["total_pymnt"] - calculable["loan_amnt"]) / calculable["loan_amnt"]
)

# 6. 연율화: r = (1+HPR)^(12/T) - 1  (0% 재투자 가정)
calculable["realized_return_annual_0pct_reinvest"] = (
    (1 + calculable["hpr"]) ** (12 / calculable["term_months"]) - 1
)

# 7. 요약 출력 (검증용)
print("\n== 전체 요약 (0% 재투자 가정) ==")
print(calculable[["hpr", "realized_return_annual_0pct_reinvest"]].describe())
print("\n== loan_status별 연율화 수익률 ==")
print(
    calculable.groupby("loan_status")["realized_return_annual_0pct_reinvest"]
    .agg(["count", "mean", "median"])
)
print("\n== term별 연율화 수익률 ==")
print(
    calculable.groupby("term_months")["realized_return_annual_0pct_reinvest"]
    .agg(["count", "mean", "median"])
)

# 8. 저장 — 본 산출물
out_cols = ["id", "issue_dt", "term_months", "loan_amnt", "total_pymnt",
            "loan_status", "hpr", "realized_return_annual_0pct_reinvest"]
calculable[out_cols].to_csv(OUT_MAIN, index=False)
print(f"\n저장 완료 (본 산출물): {OUT_MAIN}")

# 9. 저장 — 감사 테이블 (제외된 건, 사유 포함)
audit_cols = ["id", "issue_dt", "term_months", "loan_amnt", "total_pymnt",
              "loan_status", "exclude_reason"]
excluded[audit_cols].to_csv(OUT_AUDIT, index=False)
print(f"저장 완료 (감사 테이블): {OUT_AUDIT}")