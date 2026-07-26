"""CPI(CPIAUCSL) 월별 시계열과 YoY 인플레이션율을 FRED에서 받아 저장한다. (이슈 #4)

규격은 docs/macro_indicators_spec.md 를 따른다.
- 시리즈: CPIAUCSL (CPI for All Urban Consumers: All Items, 월별, 계절조정(SA),
  단위 지수 1982-84=100)
- 기간: 2007-01 ~ 2020-09 (165개월)
- 컬럼: observation_date, CPIAUCSL, cpi_yoy_pct

YoY 계산을 위해 **2006-01부터 받아** 12개월 전 대비 변화율을 구한 뒤 2007-01 이후만
저장한다(스펙 3.4 권장안). 2007-01부터 받으면 2007년 12개월치 YoY가 전부 결측이 되지만,
이 방식은 결측 0으로 떨어진다.

지수 레벨(CPIAUCSL) 자체는 단조증가라 피처로서 의미가 약하다. 실제 모델에 쓸 후보는
cpi_yoy_pct 쪽이지만, 판단은 결합 단계로 미루고 둘 다 저장한다.

발표시차를 미리 적용하지 않는다 — 원시 시계열을 발표 기준월 그대로 저장하고,
lag 적용 여부는 거시지표 결합 단계에서 팀이 일괄 결정한다(스펙 2.6).

실행:
    python src/preprocessing/fetch_macro_cpi.py
"""
import io
from pathlib import Path

import pandas as pd
import requests

SERIES = "CPIAUCSL"
# YoY 12개월치를 확보하려고 저장 시작월(2007-01)보다 1년 앞에서 받는다.
FETCH_START = "2006-01-01"
START, END = "2007-01-01", "2020-09-01"
EXPECTED_ROWS = 165

URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id={SERIES}&cosd={FETCH_START}&coed={END}"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "processed" / "macro_cpi_monthly_2007-01_to_2020-09.csv"

# 스펙 2.9 값 검증 — 알려진 사건과 대조한다. (기준월, 컬럼, 최솟값, 최댓값, 설명)
# 기대값은 "지금 FRED에서 받으면 나오는 값" 기준으로 적는다(스펙 2.9 주의사항).
SANITY_CHECKS = [
    ("2008-07-01", "cpi_yoy_pct", 5.0, 6.0, "유가 급등기 인플레이션 정점(+5.5%)"),
    ("2009-07-01", "cpi_yoy_pct", -2.5, -1.5, "금융위기 후 디플레이션 저점(-2.0%)"),
    ("2020-05-01", "cpi_yoy_pct", -0.5, 0.8, "코로나 수요 급감으로 인플레 거의 소멸"),
    ("2007-01-01", SERIES, 195.0, 210.0, "지수 레벨(1982-84=100 기준) 타당 범위"),
    ("2020-09-01", SERIES, 250.0, 270.0, "기간 말 지수 레벨"),
]


def main() -> None:
    # pandas.read_csv(URL)을 직접 쓰지 않고 requests를 거치는 이유:
    # python.org 빌드 Python은 CA 인증서가 설치돼 있지 않아 표준 urllib이
    # SSLCertVerificationError로 실패한다. requests는 certifi 번들을 쓴다.
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()

    # FRED는 결측을 "."으로 내려주므로 NaN으로 받는다 (스펙 2.4).
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df.columns = ["observation_date", SERIES]
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    if not df["observation_date"].equals(
        pd.Series(pd.date_range(FETCH_START, END, freq="MS"))
    ):
        raise ValueError("받은 원자료의 월이 연속적이지 않거나 중복/누락된 달이 있습니다")
    if df[SERIES].isna().any():
        raise ValueError(f"원자료에 결측이 있습니다: {df[SERIES].isna().sum()}개")

    # 12개월 전 대비 변화율. shift(12)는 위에서 월 연속성을 검증했으므로 안전하다.
    df["cpi_yoy_pct"] = ((df[SERIES] / df[SERIES].shift(12) - 1) * 100).round(3)

    # YoY 계산용으로만 쓴 2006년 12개월을 잘라낸다.
    df = df[df["observation_date"] >= pd.Timestamp(START)].reset_index(drop=True)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"행 수가 {EXPECTED_ROWS}가 아닙니다: {len(df)}행")
    if df[["CPIAUCSL", "cpi_yoy_pct"]].isna().any().any():
        raise ValueError("저장 대상 구간에 결측이 있습니다 — YoY 계산용 선행 12개월을 확인하세요")

    failed = []
    for date, col, lo, hi, note in SANITY_CHECKS:
        value = df.loc[df["observation_date"] == pd.Timestamp(date), col].iloc[0]
        ok = lo <= value <= hi
        print(f"  [{'OK' if ok else '실패'}] {date[:7]} {col}={value:+.3f} (기대 {lo:+.1f}~{hi:+.1f}) — {note}")
        if not ok:
            failed.append((date, col))
    if failed:
        raise ValueError(f"값 검증 실패 — 시리즈를 잘못 받았을 수 있습니다: {failed}")

    df["observation_date"] = df["observation_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUT_PATH, index=False)

    deflation = df[df["cpi_yoy_pct"] < 0]
    print(f"\n저장 완료: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"행 수: {len(df)}  기간: {df['observation_date'].iloc[0]} ~ {df['observation_date'].iloc[-1]}")
    print(f"결측: {df[[SERIES, 'cpi_yoy_pct']].isna().sum().sum()}개")
    for col in [SERIES, "cpi_yoy_pct"]:
        s = df[col].describe()
        print(
            f"  {col:<12} min {s['min']:+.3f} / max {s['max']:+.3f} / "
            f"mean {s['mean']:+.3f} / std {s['std']:.3f}"
        )
    print(f"\nYoY 음수(디플레이션) 개월: {len(deflation)}개월")
    if len(deflation):
        print(deflation[["observation_date", "cpi_yoy_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
