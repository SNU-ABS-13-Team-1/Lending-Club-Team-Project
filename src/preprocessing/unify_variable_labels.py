"""
세 AI(gemini, vscode, claude)가 각각 매긴 사전/사후(is_pre_approval) 라벨을 비교해
variable_labels_unified.xlsx를 생성한다.

규칙:
1. 세 파일 모두 값이 같은(0,0,0 또는 1,1,1) 변수는 그 값을 그대로 확정한다.
2. 세 파일 중 하나라도 값이 다른 변수는 is_pre_approval을 빈칸(NaN)으로 비운다.

주의: 입력 3개 파일(variable_labels_{gemini,vscode,claude}.xlsx)은 중간 산출물이라
data/processed/에서 삭제했다. 재실행하려면 커밋 c089792 시점에서 복원해야 한다
(예: git show c089792:data/processed/variable_labels_gemini.xlsx > variable_labels_gemini.xlsx).
결과물 variable_labels_unified.xlsx는 그대로 남아 있다.
"""
import pandas as pd

BASE = "variable_dictionary_byGJ.xlsx"
OUT = "variable_labels_unified.xlsx"

base = pd.read_excel(BASE, sheet_name="LoanStats")

gemini = pd.read_excel("variable_labels_gemini.xlsx")[["LoanStatNew", "is_pre_approval"]] \
    .rename(columns={"is_pre_approval": "gemini_label"})
vscode = pd.read_excel("variable_labels_vscode.xlsx")[["LoanStatNew", "is_pre_approval"]] \
    .rename(columns={"is_pre_approval": "vscode_label"})
claude = pd.read_excel("variable_labels_claude.xlsx", sheet_name="LoanStats")[["LoanStatNew", "사전_사후_라벨"]] \
    .rename(columns={"사전_사후_라벨": "claude_label"})

merged = base.merge(gemini, on="LoanStatNew", how="left") \
             .merge(vscode, on="LoanStatNew", how="left") \
             .merge(claude, on="LoanStatNew", how="left")

label_cols = ["gemini_label", "vscode_label", "claude_label"]
if merged[label_cols].isna().any().any():
    missing = merged[merged[label_cols].isna().any(axis=1)]["LoanStatNew"].tolist()
    raise ValueError(f"세 파일 중 일부에서 매칭되지 않은 변수가 있습니다: {missing}")

agree = merged[label_cols].nunique(axis=1) == 1
merged["is_pre_approval"] = merged["gemini_label"].where(agree)  # 불일치 시 NaN(빈칸)

merged.to_excel(OUT, index=False)

n_agree = agree.sum()
n_disagree = (~agree).sum()
counts = merged.loc[agree, "is_pre_approval"].value_counts().sort_index()

print(f"저장 완료: {OUT}")
print(f"전체 변수 수: {len(merged)}")
print(f"세 파일 일치(값 확정): {n_agree}개  (사전=1: {int(counts.get(1, 0))}개, 사후=0: {int(counts.get(0, 0))}개)")
print(f"세 파일 불일치(빈칸 처리): {n_disagree}개")
if n_disagree:
    print("불일치 변수 목록:")
    print(merged.loc[~agree, ["LoanStatNew", "gemini_label", "vscode_label", "claude_label"]].to_string(index=False))
