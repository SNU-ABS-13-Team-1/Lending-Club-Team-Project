# 출처 카드 — macro_cpi_monthly_2007-01_to_2020-09.csv

CPI 데이터가 **어디서 온 것인지** 기록한 파일이다. 검증하는 사람은 이 문서만 보고 같은
데이터를 다시 받아 대조할 수 있다. 상세 리포트는 `outputs/reports/macro_cpi.md`.

| 항목 | 내용 |
| --- | --- |
| 지표 | 미국 소비자물가지수(CPI) 및 전년동월대비 인플레이션율 |
| 제공처 | FRED — Federal Reserve Bank of St. Louis |
| 원출처 | U.S. Bureau of Labor Statistics (BLS) |
| 시리즈 ID | `CPIAUCSL` — CPI for All Urban Consumers: All Items in U.S. City Average |
| 단위 | **지수** (1982-84년 = 100). 파생 컬럼 `cpi_yoy_pct`는 **%** |
| 계절조정 | SA (계절조정 있음). 원계열은 별도 시리즈 `CPIAUCNS` |
| 주기 | 월별 (원자료가 이미 월별 — 집계 없음) |
| 기간 | 2007-01 ~ 2020-09 (165행, 결측 0) |
| 받은 날짜 | 2026-07-26 |
| 수집 스크립트 | `src/preprocessing/fetch_macro_cpi.py` |
| SHA-256 | `7caf88b8b78ea947c1c33bcc2f82f27c1fc84caa623ae5fc057f5e2550cdee56` |

컬럼: `observation_date`, `CPIAUCSL`(지수), `cpi_yoy_pct`(전년동월대비 %, 소수 셋째 자리)

## 다운로드 URL

```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd=2006-01-01&coed=2020-09-01
```

**시작월이 2006-01인 것에 주의** — 저장 기간은 2007-01부터지만, 2007년 12개월치 YoY를
계산하려면 12개월 전 값이 필요해 1년 앞에서부터 받는다. 계산 후 2006년 12개월은 잘라낸다.
덕분에 `cpi_yoy_pct`의 결측이 **0개**다 (2007-01부터 받으면 2007년 전체가 결측이 된다).

사람이 눈으로 보려면: <https://fred.stlouisfed.org/series/CPIAUCSL>

## 파생 컬럼 계산식

```
cpi_yoy_pct = (CPIAUCSL / CPIAUCSL_12개월전 - 1) * 100      (소수 셋째 자리 반올림)
```

지수 레벨(`CPIAUCSL`) 자체는 거의 단조증가라 피처로서 의미가 약하다. 실제 모델에 쓸
후보는 `cpi_yoy_pct` 쪽이지만, 판단은 결합 단계로 미루고 둘 다 저장했다.

## 검증 방법 3가지

**① 스크립트 재실행 — 가장 간단**

```bash
python src/preprocessing/fetch_macro_cpi.py
shasum -a 256 data/processed/macro_cpi_monthly_2007-01_to_2020-09.csv
```

위 표의 SHA-256과 같으면 동일한 데이터다. 스크립트에는 월 연속성·결측·행 수·값 범위
검증이 내장되어 있어, 어긋나면 파일을 쓰지 않고 예외로 중단한다.

**② FRED에서 직접 받아 비교**

위 URL을 브라우저에 넣으면 CSV가 받아진다. `CPIAUCSL` 컬럼은 그대로 같아야 하고,
위 계산식대로 YoY를 계산하면 `cpi_yoy_pct`와 같아야 한다.

**③ 원출처(BLS)와 대조 — 단, 아래 "SA/NSA 함정"을 먼저 읽을 것**

BLS는 매월 둘째 주에 CPI 뉴스 릴리스를 낸다. 아카이브:
<https://www.bls.gov/bls/news-release/cpi.htm>

실제로 대조해 확인한 사례:

| 대조 대상 | BLS 원문 (2020-04-10 발표) | 이 CSV | 일치 |
| --- | --- | --- | --- |
| 2020-03 전년동월대비 | "increased **1.5 percent**" | 1.494 | O (반올림 시 1.5) |

> 참고: bls.gov는 자동 수집 도구의 접근을 차단(403)하므로 **브라우저로 직접 열어야** 한다.

### SA/NSA 함정 — BLS 헤드라인과 소수점이 안 맞는 이유

BLS 뉴스 릴리스의 "over the last 12 months, the all items index increased X percent"는
**원계열(NSA, `CPIAUCNS`) 기준**으로 계산한 값이다. 반면 이 파일의 `cpi_yoy_pct`는
**계절조정 지수(SA, `CPIAUCSL`)** 로 계산했다.

두 값은 가깝지만 정확히 같지는 않다. 2019-01~2020-09 구간에서 실측한 차이:

| 기준월 | 우리 값 (SA 기준) | BLS 헤드라인 방식 (NSA 기준) | 차이 |
| --- | --- | --- | --- |
| 2020-01 | 2.600 | 2.487 | +0.113 |
| 2020-03 | 1.494 | 1.539 | -0.045 |
| 2020-05 | 0.198 | 0.118 | +0.080 |

**평균 절대차 0.043%p, 최대 0.113%p.** 소수 둘째 자리에서 차이가 나는 것은 정상이며
오류가 아니다. 스펙 2.5가 "4개 지표 모두 SA로 통일"을 요구하므로 SA 기준을 유지한다.

## 알아둘 것 두 가지

### 발표 시차 — 약 2주

BLS는 CPI를 **기준월 다음 달 둘째 주**(오전 8:30 ET)에 발표한다. 확인한 사례:
- 2020년 3월분 → 2020-04-10 발표 (월말 +10일)
- 2026년 7월분 → 2026-08-12 발표 (월말 +12일)

즉 2015년 3월에 대출을 심사하던 시점에 알 수 있는 최신 CPI는 2015년 2월치다.
`observation_date`는 발표일이 아니라 **기준월**이며, **이 파일에는 시차를 적용하지 않았다.**
**1개월 lag이면 leakage가 해소된다** (실업률·실업수당청구와 결론 동일).

### 개정(revision)

CPI 지수의 **원계열(NSA)은 발표 후 개정되지 않는다** — 이 점에서 실업률·실업수당청구와
다르다. 다만 **계절조정 계수는 매년 재산정**되며 통상 직전 5년치 SA 값이 개정된다.
이 파일은 SA 기준이므로 그 영향을 받는다.

- YoY 변화율은 12개월 간격의 비율이라 레벨 개정의 영향이 상당 부분 상쇄된다.
- 그래도 vintage가 아닌 최신 개정치라는 점은 동일하다. 엄밀하게 하려면
  ALFRED(<https://alfred.stlouisfed.org>)가 필요하다.
- 개정 폭은 실업수당청구(2020년 -11%)와 비교하면 훨씬 작다.
