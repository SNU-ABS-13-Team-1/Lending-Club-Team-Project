"""10년-2년 국채 금리차를 FRED에서 받아 저장한다. (이슈 #3)

규격은 docs/macro_indicators_spec.md 를 따른다.
- 시리즈: GS10 (10-Year Treasury Constant Maturity Rate), GS2 (2-Year), 둘 다 월별, 단위 %
- 기간: 2007-01 ~ 2020-09 (165개월)
- 컬럼: observation_date, GS10, GS2, spread_10y2y (= GS10 - GS2)

일별 시리즈(DGS10/DGS2/T10Y2Y)를 월평균 내지 않고 월별 GS 계열을 쓰는 이유:
이미 확보한 무위험수익률 파일(us_treasury_GS3_GS5_monthly_*.csv)이 같은 GS 계열이라
만기 구조가 일관되게 맞고, 추가 가공 단계가 없어 재현이 단순하다.

다만 월평균은 짧은 역전을 지워버린다 — 2019년 10y-2y 역전은 일별 기준 단 3일
(2019-08-27~29, 최저 -0.04)이었고 월평균으로는 2019-08이 +0.06으로 양수다.
자세한 내용은 outputs/reports/macro_yield_spread.md 참고.

발표시차: 이 지표는 시장가격이라 사실상 실시간이다(스펙 2.6의 발표시차 이슈가
거의 없는 유일한 지표). 그래도 다른 지표와 규격을 맞추기 위해 시차는 적용하지 않는다.

실행:
    python src/preprocessing/fetch_macro_yield_spread.py
"""
import io
from pathlib import Path

import pandas as pd
import requests

SERIES = ["GS10", "GS2"]
START, END = "2007-01-01", "2020-09-01"
EXPECTED_ROWS = 165

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = (
    REPO_ROOT / "data" / "processed" / "macro_yield_spread_10y2y_monthly_2007-01_to_2020-09.csv"
)

# 스펙 2.9 값 검증 — 알려진 사건과 대조한다. (기준월, 컬럼, 최솟값, 최댓값, 설명)
SANITY_CHECKS = [
    ("2007-02-01", "spread_10y2y", -0.30, -0.05, "금융위기 직전 장단기 금리 역전"),
    ("2010-02-01", "spread_10y2y", 2.60, 3.00, "위기 후 완화정책으로 곡선 최대 급경사"),
    ("2019-08-01", "spread_10y2y", -0.05, 0.20, "2019년 곡선 평탄화(월평균은 0 근처 양수)"),
    ("2020-09-01", "GS10", 0.50, 0.90, "코로나 이후 초저금리 — 10년물 1% 미만"),
    ("2007-01-01", "GS10", 4.50, 5.00, "금융위기 이전 정상 금리 수준"),
]


def fetch(series: str) -> pd.DataFrame:
    """FRED에서 시리즈 하나를 받아 DataFrame으로 반환한다.

    여러 시리즈를 `id=GS10,GS2` 처럼 한 번에 받는 URL은 cosd/coed(기간)를 무시하고
    일부 컬럼을 빈 값으로 내려주는 경우가 있어, 시리즈별로 따로 받아 병합한다.

    pandas.read_csv(URL)을 직접 쓰지 않고 requests를 거치는 이유: python.org 빌드
    Python은 CA 인증서가 없어 표준 urllib이 SSLCertVerificationError로 실패한다.
    """
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&cosd={START}&coed={END}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # FRED는 결측을 "."으로 내려주므로 NaN으로 받는다 (스펙 2.4).
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df.columns = ["observation_date", series]
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{series}: 행 수가 {EXPECTED_ROWS}가 아닙니다 ({len(df)}행)")
    if df[series].isna().any():
        raise ValueError(f"{series}: 결측이 {df[series].isna().sum()}개 있습니다")
    return df


def main() -> None:
    df = fetch(SERIES[0])
    for series in SERIES[1:]:
        df = df.merge(fetch(series), on="observation_date", how="outer")

    df = df.sort_values("observation_date").reset_index(drop=True)
    if not df["observation_date"].equals(
        pd.Series(pd.date_range(START, END, freq="MS"))
    ):
        raise ValueError("월이 연속적이지 않거나 중복/누락된 달이 있습니다")

    # 원자료가 소수 둘째 자리이므로 차이도 둘째 자리로 맞춘다.
    # (부동소수점 뺄셈이 4.76-4.88 = -0.12000000000000011 처럼 나오는 것을 막는다)
    df["spread_10y2y"] = (df["GS10"] - df["GS2"]).round(2)

    failed = []
    for date, col, lo, hi, note in SANITY_CHECKS:
        value = df.loc[df["observation_date"] == pd.Timestamp(date), col].iloc[0]
        ok = lo <= value <= hi
        print(f"  [{'OK' if ok else '실패'}] {date[:7]} {col}={value:+.2f} (기대 {lo:+.2f}~{hi:+.2f}) — {note}")
        if not ok:
            failed.append((date, col))
    if failed:
        raise ValueError(f"값 검증 실패 — 시리즈를 잘못 받았을 수 있습니다: {failed}")

    df["observation_date"] = df["observation_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUT_PATH, index=False)

    inverted = df[df["spread_10y2y"] < 0]
    print(f"\n저장 완료: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"행 수: {len(df)}  기간: {df['observation_date'].iloc[0]} ~ {df['observation_date'].iloc[-1]}")
    print(f"결측: {df[['GS10', 'GS2', 'spread_10y2y']].isna().sum().sum()}개")
    for col in ["GS10", "GS2", "spread_10y2y"]:
        s = df[col].describe()
        print(
            f"  {col:<12} min {s['min']:+.2f} / max {s['max']:+.2f} / "
            f"mean {s['mean']:+.3f} / std {s['std']:.3f}"
        )
    print(f"\n금리차 역전(음수) 개월: {len(inverted)}개월")
    if len(inverted):
        print(inverted[["observation_date", "GS10", "GS2", "spread_10y2y"]].to_string(index=False))


if __name__ == "__main__":
    main()
