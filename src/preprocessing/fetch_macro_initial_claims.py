"""신규 실업수당청구(ICSA) 주간 시계열을 FRED에서 받아 월별로 집계해 저장한다. (이슈 #2)

규격은 docs/macro_indicators_spec.md 를 따른다.
- 시리즈: ICSA (Initial Claims, 주간, 계절조정(SA), 단위 명(건))
- 기간: 2007-01 ~ 2020-09 (월별 165개월 / 주간 717주)
- 산출물 2개:
    macro_initial_claims_raw_weekly_2007-01_to_2020-09.csv  (주간 원자료, 717행)
    macro_initial_claims_monthly_2007-01_to_2020-09.csv     (월별 집계, 165행)

주간 -> 월별 집계 방식 (스펙 3.2):
    ICSA의 관측일은 그 주의 **토요일(week ending)** 이다. week ending 날짜가 속한 달로
    주를 배정한 뒤 **그 달에 속한 주간 관측치의 평균**을 취한다.

    합계(sum)가 아니라 평균(mean)을 쓰는 이유: 한 달에 속하는 주가 4주인 달이 108개,
    5주인 달이 57개로 갈린다. 합계로 집계하면 5주 달의 값이 구조적으로 약 25% 부풀려져
    실제 청구 수준과 무관한 계절적 톱니가 생긴다.

발표시차를 미리 적용하지 않는다 — 원시 시계열을 발표 기준주/월 그대로 저장하고,
lag 적용 여부는 거시지표 결합 단계에서 팀이 일괄 결정한다(스펙 2.6).

실행:
    python src/preprocessing/fetch_macro_initial_claims.py
"""
import io
from pathlib import Path

import pandas as pd
import requests

SERIES = "ICSA"
START, END = "2007-01-01", "2020-09-30"
EXPECTED_WEEKS = 717
EXPECTED_MONTHS = 165

URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id={SERIES}&cosd={START}&coed={END}"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_WEEKLY = PROCESSED / "macro_initial_claims_raw_weekly_2007-01_to_2020-09.csv"
OUT_MONTHLY = PROCESSED / "macro_initial_claims_monthly_2007-01_to_2020-09.csv"

# 스펙 2.9 값 검증 — 알려진 사건과 대조한다. (기준월, 최솟값, 최댓값, 설명)
SANITY_CHECKS = [
    ("2007-01-01", 250_000, 400_000, "금융위기 이전 평시 수준(주당 30만 건대)"),
    ("2009-03-01", 550_000, 750_000, "금융위기 청구 피크 구간"),
    ("2019-12-01", 190_000, 280_000, "코로나 직전 역사적 최저 수준"),
    ("2020-04-01", 4_000_000, 5_500_000, "코로나 급등 피크(주당 400만 건대 평균)"),
]


def main() -> None:
    # pandas.read_csv(URL)을 직접 쓰지 않고 requests를 거치는 이유:
    # python.org 빌드 Python은 CA 인증서가 설치돼 있지 않아 표준 urllib이
    # SSLCertVerificationError로 실패한다. requests는 certifi 번들을 쓴다.
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()

    # FRED는 결측을 "."으로 내려주므로 NaN으로 받는다 (스펙 2.4).
    weekly = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    weekly.columns = ["observation_date", SERIES]
    weekly["observation_date"] = pd.to_datetime(weekly["observation_date"])
    weekly = weekly.sort_values("observation_date").reset_index(drop=True)

    if len(weekly) != EXPECTED_WEEKS:
        raise ValueError(f"주간 행 수가 {EXPECTED_WEEKS}가 아닙니다: {len(weekly)}행")
    if weekly[SERIES].isna().any():
        raise ValueError(f"주간 원자료에 결측이 있습니다: {weekly[SERIES].isna().sum()}개")

    # 관측일이 전부 토요일(week ending)인지 확인 — 집계 규칙의 전제다.
    weekdays = set(weekly["observation_date"].dt.day_name())
    if weekdays != {"Saturday"}:
        raise ValueError(f"관측일이 토요일이 아닌 행이 있습니다: {weekdays}")

    # week ending 날짜가 속한 달로 배정 후 월평균.
    monthly = (
        weekly.set_index("observation_date")[SERIES]
        .resample("MS")
        .mean()
        .rename(SERIES)
        .reset_index()
    )

    if len(monthly) != EXPECTED_MONTHS:
        raise ValueError(f"월별 행 수가 {EXPECTED_MONTHS}가 아닙니다: {len(monthly)}행")
    if not monthly["observation_date"].equals(
        pd.Series(pd.date_range("2007-01-01", "2020-09-01", freq="MS"))
    ):
        raise ValueError("월이 연속적이지 않거나 중복/누락된 달이 있습니다")
    if monthly[SERIES].isna().any():
        raise ValueError("주가 하나도 배정되지 않은 달이 있습니다")

    failed = []
    for date, lo, hi, note in SANITY_CHECKS:
        value = monthly.loc[monthly["observation_date"] == pd.Timestamp(date), SERIES].iloc[0]
        ok = lo <= value <= hi
        print(
            f"  [{'OK' if ok else '실패'}] {date[:7]} {SERIES}={value:,.0f} "
            f"(기대 {lo:,}~{hi:,}) — {note}"
        )
        if not ok:
            failed.append(date)
    if failed:
        raise ValueError(f"값 검증 실패 — 시리즈를 잘못 받았을 수 있습니다: {failed}")

    # 월평균은 정수 건수의 평균이라 소수가 생긴다. 원자료 단위(명)에 맞춰 반올림한다.
    monthly[SERIES] = monthly[SERIES].round().astype("int64")

    for df, path in ((weekly, OUT_WEEKLY), (monthly, OUT_MONTHLY)):
        out = df.copy()
        out["observation_date"] = out["observation_date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)

    weeks_per_month = weekly.groupby(weekly["observation_date"].dt.to_period("M")).size()
    peak_week = weekly.loc[weekly[SERIES].idxmax()]

    print(f"\n저장 완료: {OUT_WEEKLY.relative_to(REPO_ROOT)} ({len(weekly)}행, 주간 원자료)")
    print(f"저장 완료: {OUT_MONTHLY.relative_to(REPO_ROOT)} ({len(monthly)}행, 월평균)")
    print(
        f"\n한 달에 속한 주 수: "
        f"{weeks_per_month.value_counts().sort_index().to_dict()} (4주/5주 달 개수)"
    )
    print(
        f"주간 최대: {peak_week['observation_date']:%Y-%m-%d} 마감 주 "
        f"{peak_week[SERIES]:,.0f}건"
    )
    print(f"\n월별 기간: {monthly['observation_date'].iloc[0]} ~ {monthly['observation_date'].iloc[-1]}")
    print(f"결측: {monthly[SERIES].isna().sum()}개")
    stats = monthly[SERIES].describe()
    print(
        f"min {stats['min']:,.0f} / max {stats['max']:,.0f} / "
        f"mean {stats['mean']:,.0f} / std {stats['std']:,.0f}"
    )


if __name__ == "__main__":
    main()
