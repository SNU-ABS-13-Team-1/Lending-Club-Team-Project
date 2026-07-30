# 출처 카드 — us_treasury_GS1M_monthly_2007-07_to_2025-09.csv

실현수익률의 계약만기 전 재투자와 계약만기 후 역할인에 사용하는 1개월 만기 미 국채
수익률 데이터의 출처와 검증 방법을 기록한다. 상세 내용은
`outputs/reports/treasury_gs1m.md`에 있다.

| 항목 | 내용 |
| --- | --- |
| 지표 | 미국 국채 1개월물 Constant Maturity Market Yield |
| 제공처 | FRED — Federal Reserve Bank of St. Louis |
| 원출처 | Board of Governors of the Federal Reserve System, H.15 Selected Interest Rates |
| 시리즈 ID | FRED `GS1M`; Federal Reserve H.15 `RIFLGFCM01_N.M` |
| 단위 | 연율 %, investment basis(bond-equivalent yield) |
| 계절조정 | 해당 없음(Not Seasonally Adjusted) |
| 주기 | 월별 — 해당 월 영업일 관측치의 평균 |
| 기간 | 2007-07 ~ 2025-09, 219행 |
| 받은 날짜 | 2026-07-29 |
| 수집 스크립트 | `src/preprocessing/fetch_treasury_gs1m.py` |
| SHA-256 | `888e1efce0085ccb20d98ef722f8312d3107bfb2703dbdc7684685d3466cea36` |

컬럼은 `observation_date`, `GS1M`이며 날짜는 `YYYY-MM-01`, 금리는 % 단위다.

## 다운로드 URL

```text
https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS1M&cosd=2007-07-01&coed=2025-09-01
```

- FRED 시리즈 설명: <https://fred.stlouisfed.org/series/GS1M>
- Federal Reserve H.15: <https://www.federalreserve.gov/releases/h15/>
- 원출처 월별 시리즈 확인: <https://www.federalreserve.gov/datadownload/Preview.aspx?pi=400&preview=H15%2FH15%2FRIFLGFCM01_N.M&rel=H15>
- 미 재무부 CMT 금리 설명: <https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions>

## 검증 방법

### 1. 스크립트 재실행과 체크섬 대조

```bash
python src/preprocessing/fetch_treasury_gs1m.py
shasum -a 256 data/processed/us_treasury_GS1M_monthly_2007-07_to_2025-09.csv
```

스크립트는 저장 전에 219행, 기간, 월 연속성, 날짜 중복, 결측, 음수, 사건별 값 범위를
검증한다. 기존 `GS3`·`GS5` 파일과 겹치는 159개월의 날짜도 전부 매칭하며,
2020-04의 `GS1M < GS3 < GS5` 만기구조를 교차 검증한다.

### 2. FRED 원자료와 직접 비교

위 다운로드 URL의 결과에서 시작월 `2007-07 = 4.82`, 마지막 달
`2025-09 = 4.24`가 CSV와 같은지 확인한다.

### 3. Federal Reserve H.15와 교차 검증

FRED의 원출처인 Federal Reserve Data Download Program의 월별
`RIFLGFCM01_N.M`과 대조했다. 2025-07~09 값 `4.37`, `4.46`, `4.24`가 FRED 및
본 CSV와 일치한다.

## 발표시차

`GS1M`은 시장에서 매일 형성되는 CMT 수익률의 월평균이므로 월 중 금리는 당시에 관측
가능하지만, 해당 월의 최종 월평균은 월말 이후 확정된다. 이 프로젝트에서는 `GS1M`을 대출
승인시점 예측변수로 쓰지 않고 사후 실현 현금흐름의 재투자·역할인에 사용하므로 예측
leakage 문제와는 구분한다.

## 개정 주의사항

금리 시장자료라 계절조정이나 정기 벤치마크 개정 대상은 아니지만, FRED/H.15의 오류 정정으로
과거 값이 바뀔 가능성은 있다. 재현 시에는 받은 날짜와 위 SHA-256을 함께 확인한다.
