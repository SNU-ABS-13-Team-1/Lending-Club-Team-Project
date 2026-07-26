# 출처 카드 — macro_unemployment_rate_monthly_2007-01_to_2020-09.csv

같은 이름의 CSV가 **어디서 온 데이터인지** 기록한 파일이다. 검증하는 사람은 이 문서만 보고
같은 데이터를 다시 받아 대조할 수 있다. 상세 리포트는 `outputs/reports/macro_unemployment_rate.md`.

| 항목 | 내용 |
| --- | --- |
| 지표 | 미국 실업률 (전국 기준, 주별 아님) |
| 제공처 | FRED — Federal Reserve Bank of St. Louis |
| 원출처 | U.S. Bureau of Labor Statistics (BLS), Current Population Survey |
| 시리즈 ID | `UNRATE` |
| 단위 | % (경제활동인구 대비 실업자 비율) |
| 계절조정 | SA (계절조정 있음). 원계열은 별도 시리즈 `UNRATENSA` |
| 주기 | 월별 (원자료가 이미 월별 — 집계·가공 없음) |
| 기간 | 2007-01 ~ 2020-09 (165행, 결측 0) |
| 받은 날짜 | 2026-07-26 |
| 수집 스크립트 | `src/preprocessing/fetch_macro_unemployment_rate.py` |
| SHA-256 | `d02a67b0cfc1a7f2f9936f282ab12f9c74c61cc44a88e144f3166ce970afe721` |

## 다운로드 URL (그대로 붙여넣으면 같은 파일이 받아짐)

```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE&cosd=2007-01-01&coed=2020-09-01
```

사람이 눈으로 보려면: <https://fred.stlouisfed.org/series/UNRATE>

## 검증 방법 3가지

**① 스크립트 재실행 — 가장 간단**

```bash
python src/preprocessing/fetch_macro_unemployment_rate.py
shasum -a 256 data/processed/macro_unemployment_rate_monthly_2007-01_to_2020-09.csv
```

위 표의 SHA-256과 같으면 동일한 데이터다. 스크립트에는 행 수·기간·월 연속성·값 범위
검증이 내장되어 있어, 어긋나면 파일을 쓰지 않고 예외로 중단한다.

**② FRED에서 직접 받아 비교 — 우리 코드를 안 믿고 확인하고 싶을 때**

위 다운로드 URL을 브라우저에 넣으면 CSV가 바로 받아진다. 받은 파일과 이 CSV를 비교하면
된다 (FRED 원본이 이미 `observation_date,UNRATE` / `YYYY-MM-01` 형식이라 가공이 없음).

**③ FRED를 거치지 않고 원출처(BLS)와 대조 — 가장 강한 검증**

FRED가 BLS 수치를 잘못 옮겼을 가능성까지 배제하려면 BLS 원본 뉴스 릴리스와 맞춰본다.
아래는 실제로 대조해 일치를 확인한 사례다.

| 대조 대상 | BLS 원문 | 이 CSV | 일치 |
| --- | --- | --- | --- |
| 2020-03 | "the unemployment rate rose to **4.4 percent**" | 4.4 | O |
| 2020-02 | 위 릴리스가 "0.9%p 상승"이라 명시 → 3.5% | 3.5 | O |

출처: BLS, THE EMPLOYMENT SITUATION — MARCH 2020 (2020-04-03 발표)
<https://www.bls.gov/news.release/archives/empsit_04032020.pdf>

다른 달을 확인하려면 BLS 아카이브에서 해당 월 릴리스를 찾으면 된다:
<https://www.bls.gov/bls/news-release/empsit.htm>

> 참고: bls.gov는 자동 수집 도구의 접근을 차단(403)하므로 **브라우저로 직접 열어야** 한다.

## 알아둘 것 두 가지

**발표 시차** — `observation_date`는 발표일이 아니라 **기준월**이다. 실제 발표는 기준월이
끝난 뒤 다음 달 첫 금요일(월말 +3~9일)이다. 예: 2020년 3월치는 2020-04-03 발표.
따라서 `issue_d` 당월 값을 그대로 쓰면 심사 시점에 알 수 없던 정보를 쓰는 셈이 된다.
**이 파일에는 시차를 적용하지 않았다** — lag은 거시지표 결합 단계에서 팀이 일괄 결정한다.

**개정(revision)** — FRED 값은 최신 개정치이지 당시 실시간 발표값이 아니다. 실제로
2020년 4월 실업률은 발표 당시 **14.7%** 로 보도됐으나 이 파일에는 **14.8%** 로 들어 있다.
BLS가 계절조정계수·인구통제를 사후 갱신하기 때문이다. 옛날 뉴스 기사 숫자와 소수점
단위로 안 맞는다면 오류가 아니라 개정 때문일 수 있으니 이 점을 먼저 확인할 것.
