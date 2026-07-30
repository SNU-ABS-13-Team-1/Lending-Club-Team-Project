"""실현수익률의 재투자·역할인에 사용할 월별 GS1M을 FRED에서 수집한다.

대상 시리즈
-----------
- GS1M: Market Yield on U.S. Treasury Securities at 1-Month Constant Maturity,
  Quoted on an Investment Basis
- 원출처: Board of Governors of the Federal Reserve System, H.15
- 주기/단위: 월별(영업일 평균), 연율 %, 계절조정 없음
- 기간: 2007-07 ~ 2025-09 (219개월)

실행
----
python src/preprocessing/fetch_treasury_gs1m.py
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import requests

SERIES = "GS1M"
START = "2007-07-01"
END = "2025-09-01"
EXPECTED_ROWS = 219

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "us_treasury_GS1M_monthly_2007-07_to_2025-09.csv"
)
CROSS_CHECK_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "us_treasury_GS3_GS5_monthly_2007-06_to_2020-09.csv"
)

DOWNLOAD_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    f"?id={SERIES}&cosd={START}&coed={END}"
)

# 현재 FRED/H.15 값에 넉넉한 허용범위를 둔 사건 기반 점검.
SANITY_CHECKS = [
    ("2007-07-01", 4.70, 4.90, "금융위기 이전 단기금리 4%대"),
    ("2008-12-01", 0.00, 0.10, "금융위기 이후 제로금리 근접"),
    ("2018-12-01", 2.25, 2.50, "2015~2018 금리 정상화"),
    ("2020-04-01", 0.05, 0.20, "코로나 충격 이후 재차 제로금리 근접"),
    ("2023-10-01", 5.40, 5.70, "2022~2023 긴축기의 고금리"),
    ("2025-09-01", 4.10, 4.40, "필요 범위 마지막 달"),
]


def fetch() -> pd.DataFrame:
    """FRED CSV를 내려받아 검증 전 DataFrame으로 반환한다."""
    response = requests.get(
        DOWNLOAD_URL,
        timeout=30,
        headers={"User-Agent": "LendingClub-Team-Project/1.0"},
    )
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text), na_values=["."])
    if frame.shape[1] != 2:
        raise ValueError(f"예상하지 못한 FRED 컬럼 구조: {frame.columns.tolist()}")

    frame.columns = ["observation_date", SERIES]
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], errors="raise"
    )
    frame[SERIES] = pd.to_numeric(frame[SERIES], errors="coerce")
    return frame


def validate(frame: pd.DataFrame) -> None:
    """저장 전에 기간·연속성·결측·중복·사건값을 검증한다."""
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"행 수가 {EXPECTED_ROWS}가 아닙니다: {len(frame)}행"
        )

    if frame["observation_date"].duplicated().any():
        raise ValueError("observation_date 중복이 있습니다")

    expected_dates = pd.date_range(START, END, freq="MS")
    if not frame["observation_date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="observation_date")
    ):
        raise ValueError("월이 연속적이지 않거나 기간·정렬이 예상과 다릅니다")

    missing = int(frame[SERIES].isna().sum())
    if missing:
        raise ValueError(f"{SERIES} 결측이 {missing}개 있습니다")

    if (frame[SERIES] < 0).any():
        bad = frame.loc[frame[SERIES] < 0, ["observation_date", SERIES]]
        raise ValueError(f"{SERIES} 음수값을 확인해야 합니다:\n{bad}")

    failed = []
    for date, lower, upper, note in SANITY_CHECKS:
        value = frame.loc[
            frame["observation_date"] == pd.Timestamp(date), SERIES
        ].iloc[0]
        passed = lower <= value <= upper
        print(
            f"  [{'OK' if passed else '실패'}] {date[:7]} "
            f"{SERIES}={value:.2f}% "
            f"(기대 {lower:.2f}~{upper:.2f}) — {note}"
        )
        if not passed:
            failed.append(date)

    if failed:
        raise ValueError(
            f"사건 기반 값 검증 실패: {failed}. 시리즈와 최신값을 확인하세요."
        )


def cross_check_existing_treasury(frame: pd.DataFrame) -> None:
    """기존 GS3·GS5 파일과 월 키 및 대표 수익률곡선 구간을 교차 검증한다."""
    if not CROSS_CHECK_PATH.exists():
        print("기존 GS3·GS5 파일이 없어 교차 검증은 건너뜁니다.")
        return

    longer = pd.read_csv(CROSS_CHECK_PATH, parse_dates=["observation_date"])
    overlap = frame.merge(longer, on="observation_date", how="inner")
    expected_overlap = len(pd.date_range(START, "2020-09-01", freq="MS"))

    if len(overlap) != expected_overlap:
        raise ValueError(
            f"GS3·GS5 교차검증 월 수 불일치: "
            f"{len(overlap)} != {expected_overlap}"
        )
    if overlap[[SERIES, "GS3", "GS5"]].isna().any().any():
        raise ValueError("GS1M·GS3·GS5 교차검증 구간에 결측이 있습니다")

    april_2020 = overlap.loc[
        overlap["observation_date"] == pd.Timestamp("2020-04-01")
    ].iloc[0]
    if not april_2020[SERIES] < april_2020["GS3"] < april_2020["GS5"]:
        raise ValueError(
            "2020-04 대표 구간에서 GS1M < GS3 < GS5 관계가 성립하지 않습니다"
        )

    print(
        f"  [OK] 기존 GS3·GS5와 {expected_overlap}개월 날짜 완전 매칭, "
        "2020-04 만기구조 교차검증 통과"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frame = fetch().sort_values("observation_date").reset_index(drop=True)
    validate(frame)
    cross_check_existing_treasury(frame)

    output = frame.copy()
    output["observation_date"] = output["observation_date"].dt.strftime(
        "%Y-%m-%d"
    )
    output.to_csv(OUT_PATH, index=False)

    stats = output[SERIES].describe()
    print(f"\n저장 완료: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(
        f"행 수: {len(output)}  "
        f"기간: {output['observation_date'].iloc[0]} ~ "
        f"{output['observation_date'].iloc[-1]}"
    )
    print(f"결측: {int(output[SERIES].isna().sum())}개")
    print(
        f"{SERIES}: min {stats['min']:.2f}% / max {stats['max']:.2f}% / "
        f"mean {stats['mean']:.3f}% / std {stats['std']:.3f}%"
    )
    print(f"SHA-256: {sha256(OUT_PATH)}")


if __name__ == "__main__":
    main()
