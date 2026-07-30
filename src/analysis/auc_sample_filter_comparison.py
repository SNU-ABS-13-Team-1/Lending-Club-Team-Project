"""문서의 "AUC 0.71대"와 실측 0.70대의 차이가 어디서 오는지 가른다.

`decision_log.md` #13 ⑤·#17 ③은 조건변수 투입 시 최종 모형 AUC 기준을 **0.71대**로 적고
있는데, 확정 표본(#16, 723,563건)으로 재보면 **0.70대**가 나온다. 원인 후보는 둘이었다 —
표본 필터가 다르거나, 피처·하이퍼파라미터가 다르거나.

**피처·모델·분할을 완전히 고정하고 표본 필터만 바꿔** 원인을 분리한다.

실행 결과 (2026-07-30, conda base):

    완결만 (만기필터 없음)        n=1,117,571  부도율 19.49%  AUC=0.72827
    완결 + 만기 + 버퍼 6m (#16)   n=  723,563  부도율 16.21%  AUC=0.70732

→ **−2.1%p가 오롯이 #16 만기필터 효과다.** 문서의 0.713은 필터 없는 쪽 범위에 있다.

만기가 도래하지 않은 완결건은 **조기부도가 과대표집**돼(부도율 19.49% vs 16.21%) 부도 신호가
인위적으로 뚜렷하다. 필터로 그 편향을 제거하면 **AUC가 내려가는 것이 정상**이다 — 성능이
나빠진 게 아니라 부풀려진 값이 빠진 것이다. #16이 필터를 넣은 이유가 바로 그것이다.

성격: **탐색·검증(재현)** 스크립트다. `config.yaml`의 6:2:2가 아니라 단일 80/20 분할을 쓴다 —
두 표본을 같은 조건으로 비교하는 것이 목적이라 의도된 차이다. 여기서 쓰는 AUC는 **표본 간
상대 비교**이며 승인/거절 기준을 정하는 데 쓰지 않는다(`src/analysis/AGENTS.md`).

재현 대상: `outputs/reports/oof_diagnostics_kgj.md` 「문서의 "AUC 0.71대"」 절.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.model import predict_default_probability, train_model  # noqa: E402
from preprocessing.loader import (  # noqa: E402
    TARGET_COLUMN,
    load_raw_loans,
    make_target,
    normalize_loan_status,
    parse_term_months,
    select_feature_columns,
)
from preprocessing.preprocessor import coerce_dtypes  # noqa: E402
from utils.config import load_config, repo_root  # noqa: E402

MATURITY_CUTOFF_ORD = 2020 * 12 + 4  # issue_d + term ≤ 2020-04


def main() -> None:
    cfg = load_config()
    head = load_raw_loans(nrows=5)
    features, _ = select_feature_columns(list(head.columns))
    needed = sorted(set(features) | {TARGET_COLUMN, "issue_d", "term"})
    raw = load_raw_loans(usecols=[c for c in needed if c in head.columns])

    status = normalize_loan_status(raw[TARGET_COLUMN])
    term = parse_term_months(raw["term"])
    issue = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
    maturity_ord = (issue.dt.year * 12 + issue.dt.month) + term

    completed = status.isin(["Fully Paid", "Charged Off"]) & issue.notna()
    scenarios = [
        ("완결만 (만기필터 없음)", completed),
        ("완결 + 만기 + 버퍼 6m (#16 확정)", completed & (maturity_ord <= MATURITY_CUTOFF_ORD)),
    ]

    rows = []
    for name, mask in scenarios:
        sub = raw.loc[mask]
        y = make_target(sub[TARGET_COLUMN])
        X = coerce_dtypes(sub[features])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=cfg.random_seed.default, stratify=y
        )
        model = train_model(X_tr, y_tr)
        auc = roc_auc_score(y_te, predict_default_probability(model, X_te))
        print(f"{name:32s} n={len(sub):>9,}  부도율 {y.mean():.2%}  AUC={auc:.5f}")
        rows.append({"scenario": name, "n": len(sub), "default_rate": round(float(y.mean()), 6),
                     "auc": round(float(auc), 5)})

    out = repo_root() / "outputs" / "auc_sample_filter_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["auc_diff_vs_unfiltered"] = (df["auc"] - df["auc"].iloc[0]).round(5)
    df.to_csv(out, index=False)
    print(f"\n차이: {(df['auc'].iloc[1] - df['auc'].iloc[0]) * 100:+.2f}%p  → {out.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
