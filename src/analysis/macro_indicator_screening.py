"""거시경제지표 4종의 부도확률 관련성 탐색 분석.

outputs/reports/macro_indicator_selection.md 에 실린 표를 전부 재생성한다.
문서의 숫자를 검증하거나 표본이 바뀐 뒤 다시 계산할 때 이 스크립트를 돌린다.

이 스크립트는 **탐색(screening)** 용이다 — 모형 학습이나 threshold 결정과는 무관하며,
src/analysis/AGENTS.md의 Sharpe Ratio 기반 의사결정 규칙이 적용되는 단계가 아니다.

표본 필터링은 src/preprocessing/AGENTS.md 규칙을 따른다:
- loan_status가 Current / Late (16-30 days) / Late (31-120 days)인 행 제외
- "Does not meet the credit policy. Status:*" 는 접두어를 떼고 동일하게 취급

⚠️ 입력이 9,000건 표본이다. 모든 분석은 원본 전수로 한다는 규칙(AGENTS.md 「데이터 규모」)의
예외로, **문서에 이미 실린 표를 그대로 재현하기 위해** 표본 경로를 유지한다.
거시지표를 실제로 모형에 쓰기로 하면 그때 전수로 바꾸고 문서 수치도 함께 재산출해야 한다.

실행:
    python src/analysis/macro_indicator_screening.py
"""
import functools
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

LOAN_FILE = PROCESSED / "lending_club_2020_train_sample_9000.csv"
MACRO_FILES = [
    "macro_unemployment_rate_monthly_2007-01_to_2020-09.csv",
    "macro_initial_claims_monthly_2007-01_to_2020-09.csv",
    "macro_yield_spread_10y2y_monthly_2007-01_to_2020-09.csv",
    "macro_cpi_monthly_2007-01_to_2020-09.csv",
]

# 결과가 확정된 상태만 남긴다 (진행 중인 대출은 부도 여부를 알 수 없음).
RESOLVED = ["Fully Paid", "Charged Off", "Default"]
EXCLUDED = ["Current", "Late (16-30 days)", "Late (31-120 days)"]

INDICATORS = ["UNRATE", "ICSA", "spread_10y2y", "cpi_yoy_pct"]


def load_macro() -> pd.DataFrame:
    """거시지표 4종을 observation_date로 병합한다."""
    dfs = [
        pd.read_csv(PROCESSED / f, parse_dates=["observation_date"]) for f in MACRO_FILES
    ]
    return functools.reduce(
        lambda a, b: a.merge(b, on="observation_date", how="outer"), dfs
    )


def load_loans() -> pd.DataFrame:
    """대출 표본을 읽어 전처리 규칙대로 필터링하고 default 라벨을 만든다."""
    raw = pd.read_csv(LOAN_FILE, usecols=["loan_status", "issue_d"], low_memory=False)

    # 141컬럼 원본에 컬럼을 덧붙이면 단편화 경고가 나므로 필요한 것만 새로 구성한다.
    df = pd.DataFrame(
        {
            # "Does not meet the credit policy. Status:Fully Paid" -> "Fully Paid"
            "status": raw["loan_status"].str.replace(
                "Does not meet the credit policy. Status:", "", regex=False
            ),
            "observation_date": pd.to_datetime(raw["issue_d"], format="%b-%Y"),
        }
    )

    df = df[~df["status"].isin(EXCLUDED)]
    df = df[df["status"].isin(RESOLVED)].copy()
    df["default"] = (df["status"] != "Fully Paid").astype(int)
    df["year"] = df["observation_date"].dt.year
    return df


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    macro = load_macro()
    loans = load_loans()
    df = loans.merge(macro, on="observation_date", how="left")

    section("0. 표본")
    raw_n = len(pd.read_csv(LOAN_FILE, usecols=["id"], low_memory=False))
    print(f"원 표본 {raw_n:,}건 -> 결과 확정분 {len(df):,}건")
    print(f"부도 {df['default'].sum():,}건 (부도율 {df['default'].mean() * 100:.2f}%)")
    print(f"거시지표 결합 결측: {df[INDICATORS].isna().sum().sum()}개")

    section("1. 거시지표 상호 상관 (월별 165개월 기준)")
    print(macro[INDICATORS + ["CPIAUCSL"]].corr().round(3).to_string())

    section("2. 지표 4분위별 부도율 (지표 값으로 줄 세워 4등분)")
    for col in INDICATORS:
        binned, edges = pd.qcut(df[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], retbins=True)
        tbl = (
            df.assign(q=binned)
            .groupby("q", observed=True)
            .agg(
                건수=("default", "size"),
                지표최소=(col, "min"),
                지표최대=(col, "max"),
                부도건수=("default", "sum"),
                부도율=("default", "mean"),
            )
        )
        tbl["부도율"] = (tbl["부도율"] * 100).round(2)
        print(f"\n-- {col} (경계 {[round(e, 2) for e in edges]}) --")
        print(tbl.to_string())

    section("3. 교란 진단 ① — 거시지표가 발행연도의 대리변수인가")
    for col in INDICATORS:
        print(f"  corr({col}, 발행연도) = {df[[col, 'year']].corr().iloc[0, 1]:+.3f}")
    print("\n실업률 4분위 x 발행연도 구성비(%):")
    q = pd.qcut(df["UNRATE"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    ct = pd.crosstab(q, df["year"], normalize="index").mul(100).round(1)
    print(ct.loc[:, [c for c in ct.columns if c >= 2012]].to_string())

    section("4. 교란 진단 ② — seasoning(관측 편향)")
    allrows = pd.read_csv(LOAN_FILE, usecols=["issue_d", "loan_status"], low_memory=False)
    allrows["y"] = pd.to_datetime(allrows["issue_d"], format="%b-%Y").dt.year
    s = allrows["loan_status"].str.replace(
        "Does not meet the credit policy. Status:", "", regex=False
    )
    tbl = allrows.assign(s=s).groupby("y").apply(
        lambda g: pd.Series(
            {
                "전체": len(g),
                "Current비중": (g["s"] == "Current").mean() * 100,
                "결과확정비중": g["s"].isin(RESOLVED).mean() * 100,
            }
        ),
        include_groups=False,
    )
    print(tbl.round(1).tail(9).to_string())

    section("5. 발행연도 고정 시 (2015~2018)")
    sub = df[df["year"].between(2015, 2018)].copy()
    print(f"대상 {len(sub):,}건 (결과 확정 표본의 {len(sub) / len(df) * 100:.0f}%)\n")
    for col in INDICATORS:
        sub["qq"] = pd.qcut(sub[col], 3, labels=["하", "중", "상"])
        r = sub.groupby("qq", observed=True)["default"].agg(["size", "mean"])
        cells = " | ".join(f"{i} {v * 100:5.2f}% (n={n:,})" for i, (n, v) in r.iterrows())
        print(f"  {col:<14} {cells}")

    section("6. 금리차 -0.870 분해 — '금리가 낮아졌다'는 오독 방지")
    ys = pd.read_csv(
        PROCESSED / "macro_yield_spread_10y2y_monthly_2007-01_to_2020-09.csv",
        parse_dates=["observation_date"],
    )
    ys["year"] = ys["observation_date"].dt.year
    print("연도별 평균(165개월 전체):")
    print(ys.groupby("year")[["GS10", "GS2", "spread_10y2y"]].mean().round(2).to_string())
    print("\n대출이 몰린 2013~2019 구간의 연도 상관:")
    win = ys[ys["year"].between(2013, 2019)]
    for col in ["GS10", "GS2", "spread_10y2y"]:
        print(f"  corr({col}, 연도) = {win[[col, 'year']].corr().iloc[0, 1]:+.3f}")
    print("\n같은 상관을 어디서 재느냐에 따라 값이 달라진다:")
    print(f"  165개월 전체     corr(spread, 연도) = {ys[['spread_10y2y', 'year']].corr().iloc[0, 1]:+.3f}")
    print(f"  대출 표본 기준   corr(spread, 연도) = {df[['spread_10y2y', 'year']].corr().iloc[0, 1]:+.3f}")

    section("7. 발행연도별 부도율 (seasoning 편향 있음 — 해석 주의)")
    g = df.groupby("year").agg(건수=("default", "size"), 부도율=("default", "mean"))
    g["부도율"] = (g["부도율"] * 100).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()
