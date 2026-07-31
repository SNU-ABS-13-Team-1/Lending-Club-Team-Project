"""**계산 제외 건 감사 테이블** — 현금흐름 분해 불가 건의 전수 사유 집계.

`final_report.md` 6.1과 `decision_log.md` #24 ⑤가 약속한 감사 테이블의 구현이다.
실현수익률 계산에서 제외된 건(train 1,210 · 2nd Test 836)을 **사유 × 상태 × 만기**로
집계해 한 파일로 남긴다 — 원본 행은 수정하지 않고, 제외가 모형·승인 판단과 무관한
기계적 제외임을 사후에 검증할 수 있게 한다.

## 실측 요지 (2026-07-31, B팀 교차검증)

두 표본 모두 제외 사유는 사실상 하나다 — **관측된 마지막 납입액(`last_pymnt_amnt`)이
정규 수령 총액(`total_pymnt − recoveries`)을 초과**해 균등배분식 월별 현금흐름 분해가
성립하지 않는 건. 필수금액 결측·국채 커버리지 사유는 양쪽 모두 0건이다.

| 표본 | 제외 | L > C_regular | last_pymnt_amnt 음수 |
| --- | ---: | ---: | ---: |
| train (필터 723,563) | 1,210 | 1,209 | 1 |
| 2nd Test (필터 481,833) | 836 | 836 | 0 |

이 결과는 B팀 독립 구현(`실현수익률_계산_v3`)과의 교차검증에서 id 단위로 일치를
확인했다(train 7/30 · 2nd Test 7/31, R·XR 최대차 1e-15 수준).

실행 (원본 CSV가 `data/raw/`에 없으면 경로를 인자로 준다):
    /opt/anaconda3/bin/python src/analysis/excluded_audit.py \
        [--train-csv 경로] [--test-csv 경로]
    → outputs/realized_return_excluded_audit.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from utils.config import repo_root
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import repo_root

from analysis.realized_return_cashflow import build_cashflow_schedule, load_raw_for_cashflow
from preprocessing.loader import second_test_path

OUT_NAME = "realized_return_excluded_audit.csv"


def audit_one(csv_path: Path | None, label: str) -> pd.DataFrame:
    """표본 하나를 필터 → 현금흐름 스케줄에 태워 **제외 건만** 사유별로 집계한다."""
    raw = load_raw_for_cashflow(csv_path=csv_path)
    sched = build_cashflow_schedule(raw)
    ex = sched.loc[~sched["cashflow_eligible"]]
    table = (
        ex.groupby(["exclude_reason", "status_norm", "term_months"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    table.columns = ["exclude_reason", "status", "term", "n"]
    table.insert(0, "sample", label)
    table.insert(1, "n_filtered", len(sched))
    print(f"[{label}] 필터 통과 {len(sched):,} / 제외 {int(table['n'].sum()):,}")
    print(table.to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-csv", type=Path, default=None,
                        help="train 원본 CSV 경로 (기본: data/raw)")
    parser.add_argument("--test-csv", type=Path, default=None,
                        help="2nd Test 원본 CSV 경로 (기본: data/raw)")
    args = parser.parse_args()

    tables = [
        audit_one(args.train_csv, "train"),
        audit_one(args.test_csv or second_test_path(), "2nd_test"),
    ]
    out_path = repo_root() / "outputs" / OUT_NAME
    pd.concat(tables, ignore_index=True).to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
