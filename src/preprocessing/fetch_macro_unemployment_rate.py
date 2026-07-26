"""실업률(UNRATE) 월별 시계열을 FRED에서 받아 data/processed/에 저장한다. (이슈 #1)

규격은 docs/macro_indicators_spec.md 를 따른다.
- 시리즈: UNRATE (Unemployment Rate, 월별, 계절조정(SA), 단위 %)
- 기간: 2007-01 ~ 2020-09 (165개월)
- 컬럼: observation_date(YYYY-MM-01), UNRATE

발표시차를 미리 적용하지 않는다 — 원시 시계열을 발표 기준월 그대로 저장하고,
lag 적용 여부는 거시지표 결합 단계에서 팀이 일괄 결정한다(스펙 2.6).

실행:
    python src/preprocessing/fetch_macro_unemployment_rate.py
"""
import io
from pathlib import Path

import pandas as pd
import requests

SERIES = "UNRATE"
START, END = "2007-01-01", "2020-09-01"
EXPECTED_ROWS = 165

URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id={SERIES}&cosd={START}&coed={END}"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "processed" / "macro_unemployment_rate_monthly_2007-01_to_2020-09.csv"

# 스펙 2.9 값 검증 — 알려진 사건과 대조한다. (기준월, 최솟값, 최댓값)
SANITY_CHECKS = [
    ("2007-01-01", 4.4, 4.8, "금융위기 이전 완전고용 수준"),
    ("2009-10-01", 9.8, 10.2, "금융위기 실업률 피크(약 10%)"),
    ("2019-12-01", 3.4, 3.8, "코로나 직전 50년래 최저 수준"),
    ("2020-04-01", 14.0, 15.0, "코로나 급등 피크(약 14.7%)"),
]


def main() -> None:
    # pandas.read_csv(URL)을 직접 쓰지 않고 requests를 거치는 이유:
    # python.org 빌드 Python은 CA 인증서가 설치돼 있지 않아 표준 urllib이
    # SSLCertVerificationError로 실패한다. requests는 certifi 번들을 쓰므로
    # 별도 설정 없이 팀원 환경에서도 동일하게 동작한다.
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()

    # FRED는 결측을 "."으로 내려주므로 NaN으로 받는다 (스펙 2.4).
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df.columns = ["observation_date", SERIES]
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"행 수가 {EXPECTED_ROWS}가 아닙니다: {len(df)}행")
    if df["observation_date"].min() != pd.Timestamp(START):
        raise ValueError(f"시작월 불일치: {df['observation_date'].min():%Y-%m}")
    if df["observation_date"].max() != pd.Timestamp(END):
        raise ValueError(f"종료월 불일치: {df['observation_date'].max():%Y-%m}")
    if not df["observation_date"].equals(
        pd.Series(pd.date_range(START, END, freq="MS"))
    ):
        raise ValueError("월이 연속적이지 않거나 중복/누락된 달이 있습니다")

    failed = []
    for date, lo, hi, note in SANITY_CHECKS:
        value = df.loc[df["observation_date"] == pd.Timestamp(date), SERIES].iloc[0]
        ok = lo <= value <= hi
        print(f"  [{'OK' if ok else '실패'}] {date[:7]} {SERIES}={value} (기대 {lo}~{hi}) — {note}")
        if not ok:
            failed.append(date)
    if failed:
        raise ValueError(f"값 검증 실패 — 시리즈를 잘못 받았을 수 있습니다: {failed}")

    df["observation_date"] = df["observation_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(OUT_PATH, index=False)

    print(f"\n저장 완료: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"행 수: {len(df)}  기간: {df['observation_date'].iloc[0]} ~ {df['observation_date'].iloc[-1]}")
    print(f"결측: {df[SERIES].isna().sum()}개")
    stats = df[SERIES].describe()
    print(
        f"min {stats['min']:.1f} / max {stats['max']:.1f} / "
        f"mean {stats['mean']:.3f} / std {stats['std']:.3f}"
    )


if __name__ == "__main__":
    main()
