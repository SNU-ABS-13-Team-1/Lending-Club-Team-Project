"""
실현수익률(HPR) 산출 — 이슈 #15 (v2: 팀 공유 데이터셋 기준)
잠정(provisional) · 0% 재투자 가정 · 민감도 비교용

decision_log.md #18에서 "0% 재투자(HPR) 결과를 민감도로 병기한다"고 확정된 항목.
본체(재투자 반영 수익률)는 유명곤·류성환님 별도 산출. 이 스크립트는 그 비교 대상.

공식:
  HPR_i = (total_pymnt_i - funded_amnt_i) / funded_amnt_i
  r_i   = (1 + HPR_i)^(12/T_i) - 1,  T_i = 약정만기(36/60개월)
  0% 재투자 가정 (암묵), IRR 미사용, 상환·부도 전 건 동일 수식

입력 (v2에서 변경 — 2026-07-30 팀 공유 데이터셋):
  load_shared_outcome("trainval") 사용 — 팀 표준 읽기 함수
  (src/preprocessing/export_shared_dataset.py, 578,850건 = train 434,137 + val 144,713)
  - 표본 필터(723,563건)는 공유 데이터셋에 이미 적용돼 있음 — 재필터하지 않음
  - test 20%(144,713건)는 접근하지 않음 (out-of-sample 유지)
  - 원금은 팀 표준에 맞춰 funded_amnt 사용 (v1의 loan_amnt에서 변경)
  - v1(원본 723,563건 전체 기준)은 test 정보가 포함돼 폐기

산출물은 parquet으로 저장 (팀 지침: CSV 변환 금지 — dtype 보장)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.export_shared_dataset import load_shared_outcome

OUT_MAIN = ROOT / "data" / "processed" / "hpr_0pct_reinvest_sensitivity_trainval_full.parquet"
OUT_AUDIT = ROOT / "data" / "processed" / "hpr_0pct_reinvest_excluded_audit_trainval_full.parquet"

# 1. 공유 데이터셋 읽기 — 팀 표준 함수 사용 (필터 이미 적용됨)
oc = load_shared_outcome("trainval")
print(f"trainval outcome: {len(oc):,}행 (기대값 578,850)")

# 2. term을 개월 수 정수로
oc["term_months"] = pd.to_numeric(
    oc["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
).astype("Int64")

# 3. 계산 가능 여부 판정
oc["exclude_reason"] = None
oc.loc[oc["total_pymnt"].isna(), "exclude_reason"] = "total_pymnt 결측"
oc.loc[oc["funded_amnt"].isna(), "exclude_reason"] = "funded_amnt 결측"
oc.loc[oc["funded_amnt"] <= 0, "exclude_reason"] = "funded_amnt 0 이하"
oc.loc[oc["term_months"].isna(), "exclude_reason"] = "term 결측"

calculable = oc[oc["exclude_reason"].isna()].copy()
excluded = oc[oc["exclude_reason"].notna()].copy()
print(f"계산 가능: {len(calculable):,}행")
print(f"계산 불가(감사 테이블 보존): {len(excluded):,}행")
if len(excluded) > 0:
    print(excluded["exclude_reason"].value_counts())

# 4. HPR 계산
calculable["hpr"] = (
    (calculable["total_pymnt"] - calculable["funded_amnt"]) / calculable["funded_amnt"]
)

# 5. 연율화: r = (1+HPR)^(12/T) - 1  (0% 재투자 가정)
calculable["realized_return_annual_0pct_reinvest"] = (
    (1 + calculable["hpr"]) ** (12 / calculable["term_months"].astype(float)) - 1
)

# 6. 자가 점검 출력
print("\n== 전체 요약 (0% 재투자 가정, trainval) ==")
print(calculable[["hpr", "realized_return_annual_0pct_reinvest"]].describe())
print("\n== loan_status별 (부도는 음수, 상환은 양수여야 정상) ==")
print(
    calculable.groupby("loan_status", observed=True)["realized_return_annual_0pct_reinvest"]
    .agg(["count", "mean", "median"])
)
print("\n== term별 ==")
print(
    calculable.groupby("term_months", observed=True)["realized_return_annual_0pct_reinvest"]
    .agg(["count", "mean", "median"])
)

# 7. 저장 — parquet (팀 지침: CSV 금지)
out_cols = ["id", "issue_d", "term_months", "funded_amnt", "total_pymnt",
            "loan_status", "hpr", "realized_return_annual_0pct_reinvest"]
calculable[out_cols].to_parquet(OUT_MAIN, index=False)
print(f"\n저장 완료 (본 산출물): {OUT_MAIN}")

audit_cols = ["id", "issue_d", "term_months", "funded_amnt", "total_pymnt",
              "loan_status", "exclude_reason"]
excluded[audit_cols].to_parquet(OUT_AUDIT, index=False)
print(f"저장 완료 (감사 테이블): {OUT_AUDIT}")