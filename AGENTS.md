# Team Project — Lending Club 신용평가 / Sharpe Ratio 최적화

서울대학교 핀테크 전문가 과정 "통계, 데이터 사이언스" 팀 프로젝트. Lending Club 대출 데이터로
부도확률 예측 신용평가모형을 구축하고, 이를 이용한 대출 승인/거절 전략의 **Sharpe Ratio를 극대화**하는
것이 목표다.

> 이 파일이 AI 코딩 도구(Claude Code, Codex, Cursor, Gemini CLI 등)가 참조하는 프로젝트 규칙의 단일 원본이다.
> `CLAUDE.md`/`GEMINI.md`는 이 파일을 그대로 가리키는 포인터 파일(`@AGENTS.md`)이므로 내용을 이원화하지 말고
> 이 파일(및 `src/*/AGENTS.md`)만 수정한다.
> 이 파일은 **300줄 이하**를 유지한다 — 규칙의 존재와 포인터는 여기에, 세부·근거 수치는
> `src/*/AGENTS.md`와 리포트에 둔다(같은 수치를 두 곳에 적으면 갱신 때마다 두 곳을 고치게 된다).

## 파이프라인 구조
`data/raw` → `src/preprocessing` → `data/processed` → `src/analysis` → `src/viz` → `outputs`

## 폴더별 규칙 참조
- 전처리 규칙: `src/preprocessing/AGENTS.md`
- 분석 규칙: `src/analysis/AGENTS.md`
- 시각화 규칙: `src/viz/AGENTS.md`
- 아키텍처 개요(Mermaid): `docs/architecture.md`
- 과거 방법론/절대 규칙 원문(참고용): `docs/references/legacy-분석에이전트-CLAUDE.md`
- Git/GitHub 협업 컨벤션: `docs/GIT_CONVENTION.md`

## 문서 권위 순서 — 충돌하면 무엇을 믿는가

문서가 여럿이라 같은 주제가 여러 곳에 적혀 있고, **갱신 시점이 달라 실제로 어긋나 있다.**
어긋난 것을 발견하면 아래 순서로 판단하고, **낮은 쪽을 고친다.**

1. **`outputs/reports/decision_log.md`** — 확정 사항의 단일 원본. 항상 최우선.
2. **`src/*/AGENTS.md`** (폴더 규칙) — 코드를 쓸 때 적용하는 실행 규칙.
3. **`README.md`** — 배경·수업 메모·후보 아이디어. 확정 여부를 판단하는 근거로 쓰지 않는다.
4. **`docs/references/legacy-*`** — 과거 폴더 기준 원문. 경로가 대부분 현존하지 않는다. 방법론만 참고.

> 실제로 이 순서가 필요했던 사례: "부도 시 실현수익률"이 문서 4곳에서 어긋나 있었고,
> `decision_log.md`(-100%)를 기준으로 나머지를 정정했다 (#4 정정 기록, 2026-07-29).

## 현재 진행 단계 (2026-07-31 회의 반영)

코드를 쓰기 전에 **해당 항목이 어느 칸에 있는지** 먼저 확인한다.

| 단계 | 상태 | 근거 |
| --- | --- | --- |
| 변수 사전/사후 라벨 | ✅ **확정** | #1 |
| LC 조건변수(`grade`·`sub_grade`·`int_rate`) 투입 | ✅ **확정 — 투입한다** | #17 ③ |
| 표본 필터 | ✅ **확정 — 만기 + 버퍼 6개월** (723,563건) | #16 |
| 모형 | ✅ **확정 — XGBoost 단일화** (로지스틱 폐기) | #17 ① |
| 대조군 | ✅ **확정 — 전부 승인(approve-all)** | #17 ② |
| 통합 모형 + `term` 피처 | ✅ **확정** (조건부 해제) | #13 ②·#17 |
| 결측치 전부 NaN 유지 | ✅ **확정** (조건부 해제, 대체·더미 없음) | #13 ③·#17 |
| 랜덤 6:2:2 분할 유지 | ✅ **확정** (시간순 분할 기각) | #13 ④ |
| 거시지표 | ✅ **확정 — 미사용** | #15 |
| Rf 기준 | ✅ **확정 — 발행시점 × 만기매칭 `GS3`/`GS5`** | #18 |
| Threshold 반복 횟수 | ✅ **확정 — K=50회** | #18 |
| 재투자 가정 | ✅ **확정 — 잔존기간 매칭 국채** | #18 |
| **실현수익률 구조** | ✅ **확정 — A′** (부도=PD분위 그룹 평균 / 정상상환=건별 계약 현금흐름) | #20 |
| └ 2단계 hurdle 회귀 | ✅ **확정 — 기각** (2단계 회귀 모델 안 만든다) | #20 |
| └ 조기상환 보정 방식 | ✅ **확정 — 건별 실현 현금흐름 반영** (B팀 명세 재현) | #22 ③ |
| └ 국채 금리 기준 | ✅ **확정 — ⓒ발행시점 고정 + 역할 3분리** (`rf`·계약분·실현분 각각 다름) | #22 ① |
| └ 서비스수수료(~1%) 반영 | ✅ **확정 — 0% 미반영** (보고서 한계에 명시) | #22 ② |
| 승인선 랭킹 기준 (`pd` / `E[XR]` / `q_score`) | ✅ **확정 — `q_score`** | #21 ① |
| 확률보정 (isotonic) | ✅ **확정 — 채택** (보정 전/후 PD 역할 분리) | #21 ② |
| 피처에서 `zip_code` 제외 | ✅ **확정 — 제외한다** (짝지은 검증 24/24 우세, Δ Sharpe +0.0122) | #21 ③ |
| 최종 모형 선택 규칙 | ✅ **확정 — K=50 중 Validation Sharpe 최고 모델** (`median_tau` 병기) | #21 ④ |
| 분할 8:2 + 2nd Test·OOF 3-fold | ✅ **확정** (Test 재추출 기각, K=50 유지) | #21 ⑥·#23 |
| 모형 후보 제출 규격 (CV 평균±sd·OOF PD) | ✅ **확정 — 승인** | #21 ⑦ |
| 시각화 규칙 | 🟡 **제안 — 팀 확정 전** (`src/viz/AGENTS.md`, 코드는 동작) | — |

> **2026-07-31 회의로 방법론 미확정이 전부 소진됐다**(#21~#24). 코드는 `plots.py`까지 전부
> 동작하며, 남은 것은 시각화 규칙의 **팀 승인**뿐이다(그림 결과에는 영향 없음).
> 손실값·가정을 **주입받는 형태**로 짜는 구조(#19)는 유지한다 — 가정 변경 시 재학습 없이 재계산한다.

### 실현수익률 — **전부 확정** (구조 A′ + 세부 3건, 2026-07-31)

프로젝트 결과(Sharpe·최적 threshold)를 가장 크게 좌우하는 항목이다. **규칙 전문·공식·근거 수치는
`src/analysis/AGENTS.md`와 `decision_log.md` #18·#20·#22에 있다** — 여기서는 결정과 원칙만 적는다.

- **확정 ① 구조 A′**(#20): 부도분 = PD분위 × term **그룹 평균**, 정상상환분 = `int_rate`·`term`·
  `installment`·발행시점 국채곡선으로 **건별 계약 현금흐름**을 직접 계산 — 알 수 있는 쪽을 그룹
  평균으로 뭉개지 않는다. **구조 B(2단계 hurdle 회귀)는 기각** — #5 정리 2·#6·#7의 2단계 내용은
  기각된 대안의 검토 기록이다. `int_rate`는 피처(#17 ③)이자 현금흐름 입력이다(#18과 모순 아님).
- **확정 ② 재투자 가정**(#18): 매달 상환액을 **잔존기간 매칭 국채**에 재투자한다(0% 아님). 정상·부도
  **동일 적용** — "정상상환은 `int_rate` 그대로"(#4)는 폐기됐다. 이 가정은 `R`을 평균 +107.5bp
  올리므로 **헤드라인은 Δ Sharpe(모형 − approve-all)** 로 보고하고 0% 재투자를 민감도로 병기한다.
- **확정 ③ 세부 3건**(#22 — 옛 "미확정 3건"): ① 조기상환 = **건별 실현 현금흐름 반영**
  (보정폭 36m +1.06%p / 60m +2.24%p, `var_정상` 칸별 추정) ② 국채 금리 = **ⓒ발행시점 고정** —
  단 **역할 3분리**(`rf`=GS3/GS5 · 계약분 재투자=ⓒ · 실현분 재투자·역할인=GS1M 실제경로, #22 표)
  ③ 서비스수수료 = **0% 미반영**. 실측 회수율은 Charged Off 평균 **-45.19%**로, 옛 베이스라인
  -100%는 부도손실을 크게 과대평가한다.
- **코드 지침**: 손실값·가정을 상수로 박지 말고 **주입받는 형태**(`ReturnAssumptions`)로 짠다.
  가정을 바꾸면 통계표·threshold만 재계산하면 되고 **모형 재학습은 불필요**하다(#19). 단, **재투자
  가정 스위치(국채/0%)는 칸별 통계표까지 다시 만든다** — 산출물 파일명에 가정 라벨이 붙는 이유다(#18·#20).
- ⚠️ 옛 문서의 **"부도 시 0"** 은 회수액 0 = 수익률 **-100%**라는 뜻이다(수익률 0% ≠ -100%).
  2026-07-29에 -100%로 통일했고(#4 정정 기록), `docs/references/legacy-*`에는 옛 표기가 보존돼 있다.

## 실행 환경

- **의존성**: 저장소 루트 `requirements.txt` (`pip install -r requirements.txt`).
- ⚠️ **macOS 시스템 `python3`에는 scikit-learn·xgboost가 없다.** 분석 스크립트가 ImportError로 죽는다.
  `outputs/reports/`의 수치는 **conda base(Python 3.13, pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.3.0)** 에서 산출됐다.
- **경로 규칙**: 모든 스크립트는 `Path(__file__).resolve().parents[2]`로 저장소 루트를 잡는다.
  어느 디렉터리에서 실행해도 동작해야 하며, CWD에 의존하는 상대 파일명을 쓰지 않는다.
- **설정 읽기**: 경로·분할비율·seed는 `config/config.yaml`이 단일 출처다.
  직접 파싱하지 말고 `src/utils/config.py`의 `load_config()`를 쓴다 (`python src/utils/config.py`로 확인 가능).
  미확정(null) 항목은 `cfg.require("...")`로 읽어 **임의의 기본값 대신 예외로 중단**시킨다.

## 데이터 규모 — 어느 파일로 작업하는가

| 파일 | 행 | 컬럼 | 용도 |
| --- | ---: | ---: | --- |
| `data/raw/lending_club_2020_train.csv` | 1,755,295 | 141 | **학습·1차 평가의 입력.** 1.2GB, git 미추적 — 팀 공유 채널에서 받는다 |
| `data/raw/lending_club_2020_test_2nd.csv` | 1,170,198 | 141 | **2nd Test(최종 외부 평가) 전용** — 필터 통과 481,833건(#23). 학습·탐색에 쓰면 규칙 위반 |
| `data/processed/lending_club_2020_train_sample_9000.csv` | 9,000 | 141 | 위 파일의 표본. **분석에 쓰지 않는다** — 과거 산출물 재현용으로만 남긴다 |
| `data/raw/lending_club.csv` | 10,000 | 11 | 조교님 실습용 축약본 — 본 분석에 쓰지 않는다 |
| `data/raw/LCDataDictionary.xlsx` | — | — | LC 공식 데이터 사전(원본). 팀 라벨은 `variable_dictionary_byGJ.xlsx` 쪽을 본다 |

> **모든 분석은 원본 전수 1,755,295행으로 한다** (2026-07-29 팀 결정).
> 표본은 결측 × 발행연도 × 타겟을 3중 교차하면 셀이 **n=61까지 말라 부호가 뒤집힌다**(실측).
> 규칙을 적용할 대상이 전수이므로 근거도 같은 분포에서 나와야 한다.
>
> ⚠️ 전수로 작업한다는 건 **모두가 1.2GB 원본을 직접 연다**는 뜻이다.
> 아래 「원본 데이터 취급」을 먼저 읽는다.
>
> 기존 문서에 남아 있는 9,000건 기준 수치는 출처가 표기돼 있다 — **전수 수치와 섞어 인용하지 않는다.**

## 원본 데이터 취급 — 읽기 전용

원본은 **git에 없다.** 한 번 덮어써지면 되돌릴 방법이 없고, 팀 공유 채널에서 다시 받아야 한다.
전수 작업으로 전환하면서 모두가 원본을 직접 열게 되므로 아래를 지킨다.

- **`data/raw/`의 어떤 파일도 수정·삭제·이동하지 않는다.** 정렬, 컬럼 추가, 인코딩 변경, 재저장 전부 금지다.
- **원본 경로로 쓰기 모드를 열지 않는다.** `to_csv`·`open(..., 'w')`·`to_parquet`의 출력 경로가
  `data/raw/` 아래로 가는 코드는 작성하지 않는다. 파생 결과는 `data/processed/`나 `outputs/`에 쓴다.
- **인플레이스 연산 자체는 무해하나 저장 경로가 위험하다.** `df.fillna(..., inplace=True)`는 메모리상
  변경이라 원본과 무관하지만, 그 뒤 **같은 경로로 저장하면 원본이 사라진다.** 저장 직전에 경로를 확인한다.
- **엑셀·Numbers로 원본 CSV를 열지 않는다.** 무심코 저장하면 인코딩·날짜 서식이 바뀌고,
  행 수 한계(1,048,576행)에 걸려 **175만 행이 잘린 채 저장된다.**
  내용을 보려면 `head`나 `pd.read_csv(..., nrows=...)`를 쓴다.
- 원본을 읽는 스크립트는 `Path(__file__).resolve().parents[2]` 기준으로 경로를 잡고 **읽기 전용으로만** 연다.

> 원본이 훼손된 것 같으면 **즉시 팀에 알린다.** 조용히 다시 받아 덮으면 어느 시점부터 숫자가
> 어긋났는지 추적할 수 없게 된다.

## 자료 위치 색인

무엇이 어디에 기록돼 있는지 찾는 표. **규칙이 아니라 길잡이다.**
새 파일을 만들 때는 아래 명명 규칙을 따르면 색인을 고치지 않아도 찾을 수 있다.

### 먼저 볼 것 — 프로젝트 맥락

| 찾는 것 | 위치 |
| --- | --- |
| **팀이 내린 모든 의사결정과 근거** (가장 중요) | `outputs/reports/decision_log.md` |
| 최종 보고서 목차(안) | `outputs/reports/decision_log.md` 상단 |
| 회의에서 결정할 안건(선택지·근거·권고안) | `outputs/reports/meeting_YYYYMMDD.md` |
| 아직 확정되지 않은 미결 사항 | 각 문서의 "팀 협의 필요"·"미확정" 표기 |

> `decision_log.md`를 먼저 읽지 않으면 결정된 사항을 다시 논의하거나 미확정을 확정처럼 다루게 된다.

### 주제별 방법론 문서 (`outputs/reports/`)

| 주제 | 문서 |
| --- | --- |
| ~~IRR·실현수익률 정의~~ → **과거 검토 자료**. 현행 정의는 `decision_log.md` #18 | `irr_vs_int_rate_definition.md` |
| 거시경제지표 선택 검토 (배경·탐색분석·미결 판단) | `macro_indicator_selection.md` |
| 거시지표 개별 수집 기록 | `macro_{지표}.md` — 출처·검증·발표시차·개정 이슈 |
| 변수 전처리 방식 검증 (결측 티어 T0~T3, seasoning 편향) | `preprocessing_validation_kgj.md` |
| 팀원 4인 검증문서 교차검증 (모형 구성·결측 처리 결론) | `preprocessing_crosscheck_kgj.md` |
| **B팀 핸드오프 — 실현수익률 계산** (산출물 3종·1순위 조기상환 보정·작업 0/7~10) | `handoff_teamb_realized_return.md` |
| **OOF 파이프라인 진단 4종** (`q_score` 채택 근거·A′ 건별 계산 실효성·분위 경계 이전·**확률보정**) | `oof_diagnostics_kgj.md` |
| **Sharpe threshold 탐색** (랭킹 기준 3종 비교·Δ의 재투자 가정 안정성·절대 Sharpe 병기 근거) | `sharpe_threshold_kgj.md` |
| **PD 모형 후보 교차검증** (팀원 3인 스펙 6종 비교·**`zip_code` 제외 제안**·후보 제출 규격안) | `model_comparison_kgj.md` |
| **실현수익률 개정안** (이슈 #23 — **부분 채택**: 현금흐름 재구성·역할인 채택, decision #22 ③) | `realized_return_treasury_reinvestment_methodology.md` |
| **초과수익 선별 개정안** (이슈 #23 — **등가중 헤드라인+금액가중 병기·재추출 기각**, decision #23) | `excess_return_sharpe_selection_methodology.md` |
| **#23 개정안 교차검토** (일치 6·채택권고 3·팀결정 3 — **회의 처리 완료**, decision #22~#23) | `methodology_23_review_kgj.md` |
| **계산식 참고자료** (실현·초과수익률·`q_score`·Sharpe 전 과정 수식 — 보고서 작성용, 유명곤) | `return_excess_sharpe_report_writing_notes.md` |
| 국채 GS1M 수집 기록 (재투자·역할인 계수용 — **`rf`가 아니다**) | `treasury_gs1m.md` |
| **절대 Sharpe 0.2068 진단** (이슈 #33 — 하락분해: 조기상환 보정 효과·건전성 점검 5종·개선안 3계층) | `sharpe_level_diagnosis_kgj.md` |
| **최종 결과 — 8:2 K=50·2nd Test** (이슈 #32 — **최종 수치 단일 원본**: Δ Sharpe +0.0893·승자 seed 26·그림 6종) | `final_result_kgj.md` |
| **수업 제출용 최종 보고서** (7장 구성 + 부록 — 수치는 `final_result_kgj.md`에서 인용) | `final_report.md` |
| **최종 보고서 v3** (`REPORT_CONVENTION.md` 규칙 8 적용판 — 8장 구성. 4장에 정의 사슬 집약, 6.3 벤치마크·오라클 신설. **현행 제출 원고**) | `final_report_v3.md` |
| **부록 C 별책 — 전체 구현 코드** (`src/` 32개 파일 8,360줄 전문. **생성물이니 손으로 고치지 말 것** — `export_code_appendix.py`로 재생성, `--check`로 대조) | `final_report_code_appendix.md` |
| **최종 제출물 PDF 2종** (1조_최종보고서·1조_전체코드 — 제출 파일명 그대로. 본책은 `scripts/build_report_pdf.py`가 생성) | `최종제출물/` |

> 이슈 #23 문서 2건은 2026-07-31 회의에서 처리됐다 — A 부분 채택 · B 등가중 헤드라인 · C 기각(#22 ③·#23). **채택된 항목만 구현 근거로 쓴다.**

> 파일명 끝의 이니셜(`_kgj` 등)은 **작성자 표기**다 — 같은 주제를 팀원별로 각자 검증한
> 문서가 여러 개 존재할 수 있다 (`docs/GIT_CONVENTION.md`의 파일명 규칙).
> 팀원 3인의 검증 문서(`preprocessing_review_rsh.md`, `preprocessing_review_ymg.md`,
> `missing_value_analysis.md`)는 아직 각자 브랜치에만 있고 `main`에 병합되지 않았다.

### 데이터 (`data/processed/`)

| 파일 | 내용 |
| --- | --- |
| `lending_club_2020_train_sample_9000.csv` | 대출 표본 9,000건 — **분석에 쓰지 않음**(과거 산출물 재현용) |
| `split_manifest_6_2_2_seed20260730.csv.gz` | **6:2:2 분할 정의**(`id`→split, 2.35MB, **현행** — seed는 `config.yaml`이 결정). 원본이 git에 없어도 팀원 전원이 동일 분할을 쓰게 하는 단일 원본 — `split_from_manifest()`로 읽는다. 옛 `…seed42` 파일은 보존만 하고 쓰지 않는다 |
| `split_manifest_7_3_seed20260730.csv.gz` · `split_manifest_8_2_seed20260730.csv.gz` | **7:3·8:2 분할 정의**(#30·이슈 #32) — Test 칸이 없다(최종 평가는 2nd Test 파일, decision #23). 8:2가 현행 체계다 |
| `shared/` (parquet 4종 + `columns.json` + `README_공유데이터.md`) | **팀 공유용 전처리 데이터셋**(git 미추적, 71.7MB) — `export_shared_dataset.py`가 생성, `load_shared()`로 읽는다. **사후변수는 `*_outcome.parquet`로 분리**(피처에 합치면 누수). CSV 변환 금지(category dtype 깨짐) |
| `variable_dictionary_byGJ.xlsx` | 변수 사전 + 사전/사후 라벨 (**단일 원본**) |
| `us_treasury_GS3_GS5_monthly_*.csv` | 무위험수익률 `rf` (Sharpe용, 독립변수 아님) — **출처 카드 없음** |
| `us_treasury_GS1M_monthly_*.csv` | 1개월 국채 — **실현 현금흐름 재투자·역할인 계수 전용(#22 ①), `rf`가 아니다.** 출처 카드 있음 |
| `macro_*_monthly_*.csv` | 거시경제지표 (독립변수 후보) |
| `*.source.md` | **출처 카드** — 짝이 되는 데이터 파일의 출처·체크섬·검증법 |

> 데이터 파일의 출처가 궁금하면 **같은 이름의 `.source.md`를 먼저 본다.**
> 새 외부 데이터를 추가할 때도 이 카드를 함께 만든다 (`docs/macro_indicators_spec.md` 2.8).
> 단, 국채수익률 파일은 팀원이 별도로 수집해 출처 추적이 어려워 카드가 없다 — 찾지 말 것.
> 값의 타당성은 `outputs/reports/macro_yield_spread.md` 7절에서 `GS10`/`GS2`와 교차 검증했다.

### 코드

코드는 성격이 셋으로 나뉜다. **섞어서 다루면 규칙을 잘못 적용한다.**

**① 수집·생성** — 데이터를 만들어 `data/processed/`에 넣는다

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/preprocessing/fetch_macro_*.py` (4개) | 거시지표 수집 (다운로드+검증+저장) | 동작 |
| `src/preprocessing/fetch_treasury_gs1m.py` | GS1M 수집 (실현 현금흐름 재투자·역할인 계수, #22 ①) | 동작 |
| `src/preprocessing/label_pre_post_by_rule.py` | 규칙 기반 사전/사후 라벨링 | 동작 |
| `src/preprocessing/export_split_manifest.py` | **6:2:2 분할 매니페스트** 생성·체크섬 대조(`--verify`) | 동작 |
| `src/preprocessing/export_shared_dataset.py` | 팀 공유 parquet 내보내기 + 읽기 진입점 `load_shared()`/`load_shared_outcome()` | 동작 |

**② 재현·검증** — 문서에 실린 표를 다시 만든다. 문서 수치를 의심할 때 여기부터 돌린다

| 스크립트 | 재현 대상 문서 | 결과 파일 |
| --- | --- | --- |
| `src/analysis/macro_indicator_screening.py` | `macro_indicator_selection.md` | — |
| `src/analysis/preprocessing_validation.py` | `preprocessing_validation_kgj.md` | — |
| `src/analysis/t2_contribution_reassessment.py` | `preprocessing_validation_kgj.md` 7절 (#12) | `outputs/t2_contribution_by_sample.csv` |
| `src/analysis/term_split_comparison.py` | `preprocessing_crosscheck_kgj.md` (#13 ②) | `outputs/term_split_comparison.csv` |
| `src/analysis/missing_scheme_comparison.py` | `preprocessing_crosscheck_kgj.md` (#13 ③) | `outputs/missing_scheme_comparison.csv` |
| `src/analysis/realized_return_spec_check.py` | `decision_log.md` #20 · 이슈 #15 (탈락 캐스케이드·계산 가능 건수) | `outputs/realized_return_spec_check_cascade.csv`, `outputs/realized_return_spec_check_R_by_status_term.csv` |
| `src/analysis/realized_return_sensitivity.py` | `decision_log.md` #18 (**재투자 가정 +107.5bp**) | `outputs/realized_return_sensitivity.csv` |
| `src/analysis/excluded_audit.py` | `final_report.md` 6.1 · `decision_log.md` #24 ⑤ (계산 제외 건 감사 — train 1,210·2nd Test 836 전건 사유 집계) | `outputs/realized_return_excluded_audit.csv` |
| `src/analysis/oracle_benchmarks.py` | `final_report_v3.md` 6.3 (**벤치마크·오라클 비교** — 전부 국채·전부 승인·모형·상태 Oracle·실현 XR>0 Oracle. 완전예지 오라클 1.4294 대 사후 최적 τ 0.2081은 **다른 개념**) | `outputs/oracle_benchmarks_8_2.csv` |
| `src/analysis/hpr_realized_return.py` | `final_report.md` 4.1·6.4 (**참고논문 HPR 연율화 관례 대조** — 전수 723,563건) | `outputs/hpr_realized_return_summary.csv` (건별 `_full.csv`는 미추적) |
| `src/analysis/auc_sample_filter_comparison.py` | `oof_diagnostics_kgj.md` (**AUC 0.71대 = #16 필터 이전 값**) | `outputs/auc_sample_filter_comparison.csv` |
| `src/analysis/model_comparison.py` (`--only` 6블록) | `model_comparison_kgj.md` (**팀원 3인 스펙 비교·`zip_code` 근거**) | `outputs/model_comparison_*.csv` (8종) |

> ⚠️ **예외 — `model_comparison.py`는 `config.yaml`의 매니페스트 분할을 쓴다**(질문이 "팀 표준
> 분할에서 어느 스펙이 나은가"라서). Test는 열지 않는다. `--only stability`는 수 시간이라 기본에서 빠져 있다.

> **탐색·검증용**이라 `config.yaml`의 6:2:2를 따르지 않는다(자체 분할·seed — 의도된 차이, 상세는
> `src/analysis/AGENTS.md`). AUC는 상대 비교 전용이고 여기 나오는 0.68대는 Lean 스펙 값 —
> **최종 모형 성능으로 인용 금지**(#13 ⑤). 대부분 1.2GB 원본을 읽어 수 분~수십 분 걸린다.
> ⚠️ **실현수익률 기준 구현은 본 파이프라인 `realized_return.py`**(B팀 인계 `handoff_teamb_*`).
> `hpr_realized_return.py`는 **논문 관례 재현**이라 값이 달라도 정상 — 0% 재투자 민감도(`reinvest="cash"`)와 다른 질문이니 섞어 인용하지 않는다.

**③ 본 파이프라인** — 여기에 "규칙"이 그대로 적용된다

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/preprocessing/loader.py` | 원본 로딩·표본 필터(**723,563건 검증 내장**)·피처 컬럼 선택 | 동작 |
| `src/preprocessing/preprocessor.py` | dtype 정리(문자열 수치 복원)·랜덤 6:2:2 층화분할 | 동작 |
| `src/analysis/model.py` | XGBoost PD 모형·K-fold OOF·**isotonic 확률보정**·PD 분위 경계 | 동작 |
| `src/analysis/realized_return.py` | 구조 A′ 실현수익률·`E[XR]`·`Var[XR]`·`q_score` (#22 확정 가정) | 동작 |
| `src/analysis/realized_return_cashflow.py` | **건별 실현 현금흐름** — B팀 명세 재현(조기상환·GS1M 재투자·역할인, `verify_against_teamb()`) | 동작 |
| `src/analysis/oof_diagnostics.py` | 진단 C-1/C-2/C-3/**C-4** — 설계 선택 실측 검증 | 동작 |
| `src/analysis/sharpe_optimizer.py` | threshold 탐색·랭킹 기준 3종 비교·**K=50 본실행**(`--repeat 0-49`) | 동작 |
| `src/analysis/final_evaluation.py` | **Test 1회 평가** — 이긴 모델 재현·고정 τ* 적용·최종 Sharpe 확정 | 동작 |
| `src/analysis/second_test_evaluation.py` | **2nd Test 외부 표본 평가** — 승자 모델·τ* 고정 적용(decision #23) | 동작 |
| `src/viz/plots.py` | 그림 6종 생성 — 산출 CSV만 읽는다(재계산 금지, `src/viz/AGENTS.md`) | 동작 |

> ⚠️ `realized_return.py`의 가정(국채 ⓒ · 수수료 0% · 조기상환 반영)은 #22로 확정됐고 여전히
> `ReturnAssumptions`로 **주입받는다** — 산출물 파일명의 `label()`로 어느 가정인지 추적된다.
> ⚠️ **PD는 역할별로 값이 다르다**(#21 ②·진단 C-4): 분위 경계·배정과 승인선 점수는 **보정 전** PD,
> `E[XR]`·`Var[XR]`의 `p̂`만 isotonic **보정 후** PD. 근거 수치는 `src/analysis/AGENTS.md`.

**공통 유틸**

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/utils/config.py` | `config.yaml` 로더 (경로·분할·seed 단일 출처) | 동작 |
| `src/utils/logger.py` | 공통 로깅 | 동작 (**현재 아무도 쓰지 않음**) |
| `src/utils/export_code_appendix.py` | 부록 C **별책 생성** — `src/` 전체 → `outputs/reports/`. `--check`로 어긋남 대조 | 동작 |
| `scripts/build_report_pdf.py` | **제출용 보고서 PDF 조판** — `final_report_v3.md` → `share/` (표지·목차 쪽번호·러닝 헤더·LaTeX 수식). `src/` 밖인 이유: 분석 코드가 아니라 부록 C 집계에서 제외 | 동작 |

> ⚠️ **변수 사전 시트의 `is_pre_approval`이 확정 사항(#1·#17 ③)과 어긋나 있다** — 그대로 믿으면
> LC 조건변수 8개가 빠져 Lean 스펙이 된다. `loader.py`의 `PRE_APPROVAL_OVERRIDES`가 보정 중이며
> **시트 개정은 열린 실행 항목**이다(`preprocessing_crosscheck_kgj.md` 10절).

### 규격·컨벤션

| 문서 | 내용 |
| --- | --- |
| `docs/macro_indicators_spec.md` | 외부 데이터 수집 규격 (기간·컬럼·검증·출처 카드) |
| `docs/GIT_CONVENTION.md` | 브랜치·커밋·PR 규칙, 파일명 이니셜 규칙 |
| `docs/REPORT_CONVENTION.md` | **제출용 보고서 작성 규칙** — 내부 기록(결정 서사·이슈 번호·논쟁 방어)을 심사자 대상 서술로 번역하는 규칙 7종 + 점검 체크리스트 |
| `docs/architecture.md` | 파이프라인 구조 (Mermaid) |
| `config/config.yaml` | 경로·분할 비율·seed 공통 설정 (`src/utils/config.py`로 읽는다) |
| `requirements.txt` | 실행 환경 버전 고정 |
| `notebooks/` | 실습 템플릿·사전 탐색 (참고용, 산출물 아님). ⚠️ `LendingClub_실습_v2.ipynb`는 **교육용 baseline** — rf 고정 5%, 정상=`int_rate`, 부도=0을 쓴다. 셋 다 현행 방법론이 아니다(#18). **방법론 근거로 인용 금지**, 코드 구조만 참고 |
| `share/` | 팀 공유용 PDF 등 — **git 미추적**(`.gitignore`), 로컬 전용 |

## 공통 원칙
- **`data/raw/`는 절대 수정하지 않는다 — 읽기 전용이다.** 상세는 「원본 데이터 취급」 절.
- 원본 대용량 CSV(`data/raw/` 내 대출 원본 데이터, ~1.2GB)는 git에 커밋하지 않는다 (`.gitignore` 참고). 팀원 공유는 별도 채널(Google Drive 등)로 한다.
- 승인/거절 판단 기준은 **Sharpe Ratio 극대화**이며, accuracy·AUC 등 일반 분류 지표로 대체하지 않는다.
  - 예외: 처리 방식 간 **상대 비교**에는 AUC를 써도 된다. 금지되는 것은 AUC로 승인/거절 기준을 정하는 것이다.
- Test set으로 모형을 재조정하지 않는다 — Train/Validation으로 확정한 모형·threshold를 그대로 적용해 검증만 한다.
- 무거운 처리는 스크립트로 작성 후 결과만 요약해서 보고한다.
- 출력 위치: 리포트는 `outputs/reports/`, 차트는 `outputs/figures/`, **재현 스크립트가 내는 집계 CSV는 `outputs/` 바로 아래**에 둔다.
- 새 산출물을 만들면 위 "자료 위치 색인"에 한 줄 추가하고, 확정 사항이면 `decision_log.md`에 근거와 함께 남긴다.
