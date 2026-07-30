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

## 현재 진행 단계 (2026-07-29 회의 반영)

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
| └ 조기상환 보정 방식 | ❌ **미확정** — B팀 1순위 | #20 |
| └ 국채 금리 ⓐ실제경로/ⓑ상수/ⓒ발행시점 고정 | ❌ 미확정 (ⓒ 권고, 잠정 진행 가능) | #18·#20 |
| └ 서비스수수료(~1%) 반영 | ❌ 미확정 (잠정 0%로 진행 가능) | #20 |
| 승인선 랭킹 기준 (`pd` / `E[XR]` / `q_score`) | ⏳ **근거 확보 — 팀 확정 대기** (`q_score` 우세) | #5·#20 |
| 확률보정 (isotonic) | ⏳ **구현·검증 완료 — 팀 확정 대기** | C-4 |
| 시각화 규칙 | ❌ 미착수 | — |

> **A팀은 미확정 항목을 기다릴 필요가 없다.** 분류 모델의 타깃은 이진 `loan_status`이므로
> 수익률 정의가 학습에 개입하지 않는다 — 정의는 threshold·Sharpe 단계에서만 들어온다(#19).
> 단, 손실값은 **주입받는 형태**로 구현해야 확정 시 재학습 없이 threshold만 다시 탐색할 수 있다.

### 실현수익률 — **구조는 확정(A′), 세부 3건 미확정** (B팀 담당)

프로젝트 결과(Sharpe·최적 threshold)를 가장 크게 좌우하는 항목이다. **규칙 전문·공식·근거 수치는
`src/analysis/AGENTS.md`와 `decision_log.md` #18·#20에 있다** — 여기서는 상태와 원칙만 적는다.

- **확정 ① 구조 A′**(#20): 부도분 = PD분위 × term **그룹 평균**, 정상상환분 = `int_rate`·`term`·
  `installment`·발행시점 국채곡선으로 **건별 계약 현금흐름**을 직접 계산 — 알 수 있는 쪽을 그룹
  평균으로 뭉개지 않는다. **구조 B(2단계 hurdle 회귀)는 기각** — #5 정리 2·#6·#7의 2단계 내용은
  기각된 대안의 검토 기록이다. `int_rate`는 피처(#17 ③)이자 현금흐름 입력이다(#18과 모순 아님).
- **확정 ② 재투자 가정**(#18): 매달 상환액을 **잔존기간 매칭 국채**에 재투자한다(0% 아님). 정상·부도
  **동일 적용** — "정상상환은 `int_rate` 그대로"(#4)는 폐기됐다. 이 가정은 `R`을 평균 +107.5bp
  올리므로 **헤드라인은 Δ Sharpe(모형 − approve-all)** 로 보고하고 0% 재투자를 민감도로 병기한다.
- **미확정 3건**(전부 잠정값으로 진행 가능 — A팀을 막지 않는다): ① **조기상환 보정**(B팀 1순위 —
  없으면 정상상환 `R` 과대추정·`var_정상 = 0`) ② 국채 금리 기준(**ⓒ발행시점 고정 권고**)
  ③ 서비스수수료 ~1%(잠정 0%). 실측 회수율은 Charged Off 평균 **-45.19%**로, 옛 베이스라인
  -100%는 부도손실을 크게 과대평가한다.
- **코드 지침**: 손실값·가정을 상수로 박지 말고 **주입받는 형태**(`ReturnAssumptions`)로 짠다.
  확정 시 통계표·threshold만 재계산하면 되고 **모형 재학습은 불필요**하다(#19). 단, **재투자 가정
  스위치(국채/0%)는 칸별 통계표까지 다시 만든다** — 산출물 파일명에 가정 라벨이 붙는 이유다(#18·#20).
- ⚠️ 옛 문서의 **"부도 시 0"** 은 회수액 0 = 수익률 **-100%**라는 뜻이다(수익률 0% = 원금 전액
  회수와 전혀 다르다). 2026-07-29에 -100%로 통일했고(#4 정정 기록), `docs/references/legacy-*`에는
  옛 표기가 과거 원문으로 보존돼 있다.

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
| `data/raw/lending_club_2020_train.csv` | 1,755,295 | 141 | **본 분석의 유일한 입력.** 1.2GB, git 미추적 — 팀 공유 채널에서 받는다 |
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

> `decision_log.md`를 먼저 읽지 않으면 이미 결정된 사항을 다시 논의하거나,
> 미확정 사항을 확정된 것처럼 다루게 된다.

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

> 파일명 끝의 이니셜(`_kgj` 등)은 **작성자 표기**다 — 같은 주제를 팀원별로 각자 검증한
> 문서가 여러 개 존재할 수 있다 (`docs/GIT_CONVENTION.md`의 파일명 규칙).
> 팀원 3인의 검증 문서(`preprocessing_review_rsh.md`, `preprocessing_review_ymg.md`,
> `missing_value_analysis.md`)는 아직 각자 브랜치에만 있고 `main`에 병합되지 않았다.

### 데이터 (`data/processed/`)

| 파일 | 내용 |
| --- | --- |
| `lending_club_2020_train_sample_9000.csv` | 대출 표본 9,000건 — **분석에 쓰지 않음**(과거 산출물 재현용) |
| `split_manifest_6_2_2_seed20260730.csv.gz` | **6:2:2 분할 정의**(`id`→split, 2.35MB, **현행** — seed는 `config.yaml`이 결정). 원본이 git에 없어도 팀원 전원이 동일 분할을 쓰게 하는 단일 원본 — `split_from_manifest()`로 읽는다. 옛 `…seed42` 파일은 보존만 하고 쓰지 않는다 |
| `shared/` (parquet 4종 + `columns.json` + `README_공유데이터.md`) | **팀 공유용 전처리 데이터셋**(git 미추적, 71.7MB) — `export_shared_dataset.py`가 생성, `load_shared()`로 읽는다. **사후변수는 `*_outcome.parquet`로 분리**(피처에 합치면 누수). CSV 변환 금지(category dtype 깨짐) |
| `variable_dictionary_byGJ.xlsx` | 변수 사전 + 사전/사후 라벨 (**단일 원본**) |
| `us_treasury_GS3_GS5_monthly_*.csv` | 무위험수익률 (Sharpe용, 독립변수 아님) — **출처 카드 없음** |
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
| `src/analysis/auc_sample_filter_comparison.py` | `oof_diagnostics_kgj.md` (**AUC 0.71대 = #16 필터 이전 값**) | `outputs/auc_sample_filter_comparison.csv` |

> **탐색·검증용**이라 `config.yaml`의 6:2:2를 따르지 않는다(자체 분할·자체 seed — 의도된 차이.
> 성격 구분 상세는 `src/analysis/AGENTS.md`). AUC는 상대 비교 전용이며, 여기서 나오는 0.68대는
> Lean 스펙 값 — **최종 모형 성능으로 인용 금지**(#13 ⑤). 대부분 1.2GB 원본을 읽어 수 분~수십 분 걸린다.
> 실현수익률 2종은 모형 학습이 없고, `realized_return_sensitivity.py`는 #18 재투자 가정 검증 전용이다.
> **본 파이프라인의 실현수익률 구현은 `src/analysis/realized_return.py`** — B팀 작업의 출발점은
> 이쪽이다 (`outputs/reports/handoff_teamb_realized_return.md`).

**③ 본 파이프라인** — 여기에 "규칙"이 그대로 적용된다

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/preprocessing/loader.py` | 원본 로딩·표본 필터(**723,563건 검증 내장**)·피처 컬럼 선택 | 동작 |
| `src/preprocessing/preprocessor.py` | dtype 정리(문자열 수치 복원)·랜덤 6:2:2 층화분할 | 동작 |
| `src/analysis/model.py` | XGBoost PD 모형·K-fold OOF·**isotonic 확률보정**·PD 분위 경계 | 동작 |
| `src/analysis/realized_return.py` | 구조 A′ 실현수익률·`E[XR]`·`Var[XR]`·`q_score` | 동작 (**잠정 가정**) |
| `src/analysis/oof_diagnostics.py` | 진단 C-1/C-2/C-3/**C-4** — 설계 선택 실측 검증 | 동작 |
| `src/analysis/sharpe_optimizer.py` | threshold 탐색·랭킹 기준 3종 비교 | 동작 |
| `src/viz/plots.py` | 차트 생성 | **뼈대(TODO)** |

> ⚠️ `realized_return.py`는 미확정 3건을 **잠정값**(ⓒ발행시점 고정 · 수수료 0% · 조기상환 보정 0)으로
> 고정하되, 가정은 전부 `ReturnAssumptions`로 **주입받는다** — 산출물 파일명에 `label()`이 붙어
> 어느 가정인지 추적된다. 산출물 목록·현행 라벨은 `src/analysis/AGENTS.md` 참고.
> ⚠️ **PD에는 두 역할이 있고 서로 다른 값을 쓴다**(진단 C-4): 분위 경계·배정과 승인선 점수는
> **보정 전** PD, `E[XR]`·`Var[XR]`의 `p̂`만 isotonic **보정 후** PD. 근거 수치는
> `src/analysis/AGENTS.md`와 `oof_diagnostics_kgj.md`·`sharpe_threshold_kgj.md`에 있다.
> `decision_log.md` 「부수 결정」의 *"isotonic은 단조변환이라 분위 경계가 바뀌지 않는다"* 는
> **틀렸다** — 정정이 필요하다(팀 확인 후).

**공통 유틸**

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/utils/config.py` | `config.yaml` 로더 (경로·분할·seed 단일 출처) | 동작 |
| `src/utils/logger.py` | 공통 로깅 | 동작 (**현재 아무도 쓰지 않음**) |

> 남은 뼈대(`plots.py`)는 방법론 확정 전이라 의도적으로 비워둔 것이다. 채우기 전에
> `decision_log.md`와 위 「현재 진행 단계」에서 확정 여부를 먼저 확인한다.
> ⚠️ **변수 사전 시트의 `is_pre_approval`이 확정 사항(#1·#17 ③)과 어긋나 있다** — 시트를 그대로
> 믿으면 LC 조건변수(`grade`·`sub_grade`·`int_rate` 등 8개)가 빠져 Lean 스펙이 된다.
> `loader.py`의 `PRE_APPROVAL_OVERRIDES`가 코드에서 보정 중이며 **시트 개정은 열린 실행 항목**이다
> (`preprocessing_crosscheck_kgj.md` 10절).

### 규격·컨벤션

| 문서 | 내용 |
| --- | --- |
| `docs/macro_indicators_spec.md` | 외부 데이터 수집 규격 (기간·컬럼·검증·출처 카드) |
| `docs/GIT_CONVENTION.md` | 브랜치·커밋·PR 규칙, 파일명 이니셜 규칙 |
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
