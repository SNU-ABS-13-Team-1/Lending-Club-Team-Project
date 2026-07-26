"""
variable_dictionary_byGJ.xlsx('LoanStats' 시트)에 is_pre_approval 라벨을 추가해 variable_labels_rule_based.xlsx로 저장.

라벨링 기준 (LendingClub 신용평가 모형에서 흔히 쓰이는 leakage 방지 기준):
- 0 (사후): 대출 실행 후 시간이 지나야(상환/연체가 진행돼야) 값이 생기는 변수.
  즉 대출 상태/상환/연체/회수 이력, hardship/settlement(연체 후 구제 프로그램) 관련 필드.
- 1 (사전): 대출 신청 시점(및 그 이전 신용이력 조회 결과)에 이미 알 수 있는 변수.
  grade/sub_grade/int_rate/installment/funded_amnt/funded_amnt_inv/issue_d/initial_list_status는
  대출 발행(origination) 시점에 함께 확정되며, 신규 심사 대상 데이터에도 이미 채워져 들어오므로
  사전 변수로 간주한다 (근거: `src/preprocessing/AGENTS.md` 참고).
"""
import pandas as pd

SRC = "variable_dictionary_byGJ.xlsx"
DST = "variable_labels_rule_based.xlsx"
SHEET = "LoanStats"

# 대출 실행 후 시간이 지나야 발생·확정되는 변수(사후, 0)
POST_APPROVAL_VARS = {
    # 대출 상태/상환 이력 (펀딩 이후 계속 갱신됨)
    "loan_status", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "pymnt_plan",
    # 상환곤란(hardship) 프로그램 - 대출 실행 후 연체/곤란 발생 시에만 생김
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # 자금 지급 방식 및 채무 정산(settlement) - 대출 실행/연체 이후 발생
    "disbursement_method",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
}

df = pd.read_excel(SRC, sheet_name=SHEET)

unknown = POST_APPROVAL_VARS - set(df["LoanStatNew"])
if unknown:
    raise ValueError(f"variable_dictionary_byGJ.xlsx에 없는 변수명이 POST_APPROVAL_VARS에 있습니다: {unknown}")

df["is_pre_approval"] = df["LoanStatNew"].apply(
    lambda x: 0 if x in POST_APPROVAL_VARS else 1
)

df.to_excel(DST, index=False)

counts = df["is_pre_approval"].value_counts().sort_index()
print(f"저장 완료: {DST}")
print(f"전체 변수 수: {len(df)}")
print(f"사후(0) 변수 개수: {counts.get(0, 0)}")
print(f"사전(1) 변수 개수: {counts.get(1, 0)}")
