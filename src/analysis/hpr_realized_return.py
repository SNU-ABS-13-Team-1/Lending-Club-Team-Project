"""보고서 비교용 — 참고논문(HPR 복리 연율화) 방식 실현수익률.

참고논문(Park et al., JDQS 34-2)의 연율화 관례를 표본 필터 통과 723,563건
**전체**에 그대로 적용한다. 본 파이프라인(현금흐름 재구성, `realized_return_cashflow.py`)
과의 관례 차이를 실측 대조하기 위한 **보고서 비교용 산출물**이며, 모델 학습·threshold
탐색 어디에도 들어가지 않는다.

    HPR_i = (Total Amount Received_i − Loan Amount_i) / Loan Amount_i
    r_i   = (1 + HPR_i)^(12/T_i) − 1        (T_i = 36 | 60)

- 팀 확정: "IRR" 표현은 쓰지 않는다 (Slack 2026-07-31).
- 표본: `loader.filter_analysis_sample` 기준 723,563건 (decision_log #16,
  Default 제외 재정정 #24 ① 반영). 80% 분할이 아니라 전체다.

실행:
    /opt/anaconda3/bin/python src/analysis/hpr_realized_return.py
산출:
    outputs/hpr_realized_return_full.csv     (건별 723,563행 — **git 미추적**)
    outputs/hpr_realized_return_summary.csv  (상태×만기 요약 통계 — 커밋한다)

⚠️ 건별 산출물은 반드시 `*_full.csv`로 둔다. `src/preprocessing/AGENTS.md`의 전수/표본
명명 규칙이자, `.gitignore`가 전수 결과를 걸러내는 규칙이 `*_full.csv`이기 때문이다 —
이름을 바꾸면 58MB CSV가 그대로 추적 대상이 된다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 저장소의 다른 모듈과 **같은 방식**으로 경로를 잡는다 — `src/`를 sys.path에 넣고
# `preprocessing.…`로 import한다. 루트를 넣고 `src.preprocessing.…`로 부르면 같은 파일이
# 별개 모듈 객체로 두 번 올라가, `loader.py`의 모듈 수준 상수(`PRE_APPROVAL_OVERRIDES` 등)가
# 두 벌 생긴다. `src/__init__.py`도 없어 namespace package에 의존하게 된다.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.loader import (  # noqa: E402
    filter_analysis_sample,
    load_raw_loans,
    parse_term_months,
)

USECOLS = ["id", "loan_status", "term", "issue_d", "funded_amnt", "total_pymnt"]
OUT_DIR = REPO_ROOT / "outputs"


def main() -> None:
    df = load_raw_loans(usecols=USECOLS)
    df = filter_analysis_sample(df)  # verify=True — 723,563건 아니면 즉시 실패

    term_months = parse_term_months(df["term"])

    # 참고논문 방식 그대로. total_pymnt = Total Amount Received.
    hpr = (df["total_pymnt"] - df["funded_amnt"]) / df["funded_amnt"]
    r_annual = (1.0 + hpr) ** (12.0 / term_months) - 1.0

    out = pd.DataFrame(
        {
            "id": df["id"],
            "loan_status": df["loan_status"],
            "term_months": term_months.astype(int),
            "funded_amnt": df["funded_amnt"],
            "total_pymnt": df["total_pymnt"],
            "hpr": hpr,
            "r_annualized_hpr": r_annual,
        }
    )

    OUT_DIR.mkdir(exist_ok=True)
    detail_path = OUT_DIR / "hpr_realized_return_full.csv"
    out.to_csv(detail_path, index=False)

    # 상태 × 만기 요약 — 보고서 표에 바로 인용할 수 있는 형태
    summary = (
        out.groupby(["loan_status", "term_months"])["r_annualized_hpr"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    overall = (
        out.groupby("loan_status")["r_annualized_hpr"]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    overall.insert(1, "term_months", "all")
    summary = pd.concat([summary, overall], ignore_index=True)
    summary_path = OUT_DIR / "hpr_realized_return_summary.csv"
    summary.to_csv(summary_path, index=False)

    n = len(out)
    print(f"표본: {n:,}건 (필터 검증 통과)")
    print(f"건별 산출: {detail_path.relative_to(REPO_ROOT)}")
    print(f"요약 산출: {summary_path.relative_to(REPO_ROOT)}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
