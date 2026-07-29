# Team Project — Lending Club 신용평가 / Sharpe Ratio 최적화

서울대학교 핀테크 전문가 과정 "통계, 데이터 사이언스" 팀 프로젝트. Lending Club 대출 데이터로
부도확률 예측 신용평가모형을 구축하고, 이를 이용한 대출 승인/거절 전략의 **Sharpe Ratio를 극대화**하는
것이 목표다.

> 이 파일이 AI 코딩 도구(Claude Code, Codex, Cursor, Gemini CLI 등)가 참조하는 프로젝트 규칙의 단일 원본이다.
> `CLAUDE.md`/`GEMINI.md`는 이 파일을 그대로 가리키는 포인터 파일(`@AGENTS.md`)이므로 내용을 이원화하지 말고
> 이 파일(및 `src/*/AGENTS.md`)만 수정한다.

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

## 현재 진행 단계 (2026-07-29 기준)

코드를 쓰기 전에 **해당 항목이 어느 칸에 있는지** 먼저 확인한다.
"조건부 확정"을 확정으로 오인하면 전제가 뒤집힐 때 작업이 통째로 원복된다.

| 단계 | 상태 | 근거 |
| --- | --- | --- |
| 변수 사전/사후 라벨 | ✅ **확정** | #1 |
| 거시지표 4종 수집 | ✅ **확정**(수집 완료) | #10 |
| 거시지표 **결합** 방식 | ❌ 미확정 (팀 협의 5건) | #11 |
| 부도 시 실현수익률 정의 | ❌ 미확정 | #4·#5 |
| 종속변수·2단계 hurdle 구조 | ❌ 제안 단계 | #5·#6 |
| 모형 후보 (로지스틱 vs GBM) | ❌ 비교 진행 중 | #3·#7 |
| 표본 필터 (만기 기준) | ⚠️ 방향 지지 / 방식 재협의 | #12 |
| 통합 모형 + `term` 피처 | ⚠️ **조건부 확정** (전제: XGBoost 단일화) | #13 ② |
| 결측치 전부 NaN 유지 | ⚠️ **조건부 확정** (전제: XGBoost 단일화 + 만기 필터) | #13 ③ |
| 랜덤 6:2:2 분할 유지 | ✅ **확정** (시간순 분할 기각) | #13 ④ |
| Rf 기준 시점 | ❌ 미확정 → `config.yaml`에서 `null` | — |
| 시각화 규칙 | ❌ 미착수 | — |

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
| `data/raw/lending_club_2020_train.csv` | 1,755,295 | 141 | **원본 전수.** 1.2GB, git 미추적 — 팀 공유 채널에서 받는다 |
| `data/processed/lending_club_2020_train_sample_9000.csv` | 9,000 | 141 | 위 파일의 표본. 빠른 검증·탐색용 |
| `data/raw/lending_club.csv` | 10,000 | 11 | 조교님 실습용 축약본 — 본 분석에 쓰지 않는다 |
| `data/raw/LCDataDictionary.xlsx` | — | — | LC 공식 데이터 사전(원본). 팀 라벨은 `variable_dictionary_byGJ.xlsx` 쪽을 본다 |

> 표본 9,000건으로 먼저 돌려 보고 전수로 확정하는 흐름이다.
> 규칙 문서의 수치는 출처를 반드시 밝힌다 — 표본과 전수는 값이 다르다.

## 자료 위치 색인

무엇이 어디에 기록돼 있는지 찾는 표. **규칙이 아니라 길잡이다.**
새 파일을 만들 때는 아래 명명 규칙을 따르면 색인을 고치지 않아도 찾을 수 있다.

### 먼저 볼 것 — 프로젝트 맥락

| 찾는 것 | 위치 |
| --- | --- |
| **팀이 내린 모든 의사결정과 근거** (가장 중요) | `outputs/reports/decision_log.md` |
| 최종 보고서 목차(안) | `outputs/reports/decision_log.md` 상단 |
| 아직 확정되지 않은 미결 사항 | 각 문서의 "팀 협의 필요"·"미확정" 표기 |

> `decision_log.md`를 먼저 읽지 않으면 이미 결정된 사항을 다시 논의하거나,
> 미확정 사항을 확정된 것처럼 다루게 된다.

### 주제별 방법론 문서 (`outputs/reports/`)

| 주제 | 문서 |
| --- | --- |
| IRR·실현수익률 정의 (부도 시 손실 처리) | `irr_vs_int_rate_definition.md` |
| 거시경제지표 선택 검토 (배경·탐색분석·미결 판단) | `macro_indicator_selection.md` |
| 거시지표 개별 수집 기록 | `macro_{지표}.md` — 출처·검증·발표시차·개정 이슈 |
| 변수 전처리 방식 검증 (결측 티어 T0~T3, seasoning 편향) | `preprocessing_validation_kgj.md` |
| 팀원 4인 검증문서 교차검증 (모형 구성·결측 처리 결론) | `preprocessing_crosscheck_kgj.md` |

> 파일명 끝의 이니셜(`_kgj` 등)은 **작성자 표기**다 — 같은 주제를 팀원별로 각자 검증한
> 문서가 여러 개 존재할 수 있다 (`docs/GIT_CONVENTION.md`의 파일명 규칙).
> 팀원 3인의 검증 문서(`preprocessing_review_rsh.md`, `preprocessing_review_ymg.md`,
> `missing_value_analysis.md`)는 아직 각자 브랜치에만 있고 `main`에 병합되지 않았다.

### 데이터 (`data/processed/`)

| 파일 | 내용 |
| --- | --- |
| `lending_club_2020_train_sample_9000.csv` | 대출 표본 9,000건 (검증·탐색용) |
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

**② 재현·검증** — 문서에 실린 표를 다시 만든다. 문서 수치를 의심할 때 여기부터 돌린다

| 스크립트 | 재현 대상 문서 | 결과 파일 |
| --- | --- | --- |
| `src/analysis/macro_indicator_screening.py` | `macro_indicator_selection.md` | — |
| `src/analysis/preprocessing_validation.py` | `preprocessing_validation_kgj.md` | — |
| `src/analysis/t2_contribution_reassessment.py` | `preprocessing_validation_kgj.md` 7절 (#12) | `outputs/t2_contribution_by_sample.csv` |
| `src/analysis/term_split_comparison.py` | `preprocessing_crosscheck_kgj.md` (#13 ②) | `outputs/term_split_comparison.csv` |
| `src/analysis/missing_scheme_comparison.py` | `preprocessing_crosscheck_kgj.md` (#13 ③) | `outputs/missing_scheme_comparison.csv` |

> 이들은 **탐색·검증용**이라 `config.yaml`의 6:2:2를 따르지 않고 자체 2분할·자체 seed 루프를 쓴다(의도된 차이).
> AUC를 쓰지만 **승인/거절 기준을 정하는 데 쓰지 않으므로** Sharpe 규칙과 충돌하지 않는다.
> ⚠️ 이 스크립트들이 내는 AUC(0.68대)는 LC 조건변수를 뺀 Lean 스펙 값 — **최종 모형 성능으로 인용 금지**(#13 ⑤).
> 대부분 `data/raw/lending_club_2020_train.csv`(1.2GB, git 미추적)를 입력으로 받으며 실행에 수 분~수십 분 걸린다.

**③ 본 파이프라인** — 아직 뼈대. 여기에 "규칙"이 그대로 적용된다

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/preprocessing/loader.py`, `preprocessor.py` | 로딩·전처리 파이프라인 | **뼈대(TODO)** |
| `src/analysis/model.py`, `sharpe_optimizer.py` | 모형 학습·threshold 탐색 | **뼈대(TODO)** |
| `src/viz/plots.py` | 차트 생성 | **뼈대(TODO)** |

**공통 유틸**

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/utils/config.py` | `config.yaml` 로더 (경로·분할·seed 단일 출처) | 동작 |
| `src/utils/logger.py` | 공통 로깅 | 동작 (**현재 아무도 쓰지 않음**) |

> 뼈대 파일은 팀이 방법론을 확정하기 전이라 의도적으로 비워둔 것이다.
> 채우기 전에 `decision_log.md`에서 해당 항목이 확정됐는지 — 그리고 위 "현재 진행 단계"에서
> **조건부 확정이 아닌지** — 먼저 확인한다.

### 규격·컨벤션

| 문서 | 내용 |
| --- | --- |
| `docs/macro_indicators_spec.md` | 외부 데이터 수집 규격 (기간·컬럼·검증·출처 카드) |
| `docs/GIT_CONVENTION.md` | 브랜치·커밋·PR 규칙, 파일명 이니셜 규칙 |
| `docs/architecture.md` | 파이프라인 구조 (Mermaid) |
| `config/config.yaml` | 경로·분할 비율·seed 공통 설정 (`src/utils/config.py`로 읽는다) |
| `requirements.txt` | 실행 환경 버전 고정 |
| `notebooks/` | 실습 템플릿·사전 탐색 (참고용, 산출물 아님) |
| `share/` | 팀 공유용 PDF 등 — **git 미추적**(`.gitignore`), 로컬 전용 |

## 공통 원칙
- `data/raw/`는 절대 수정하지 않는다.
- 원본 대용량 CSV(`data/raw/` 내 대출 원본 데이터, ~1.2GB)는 git에 커밋하지 않는다 (`.gitignore` 참고). 팀원 공유는 별도 채널(Google Drive 등)로 한다.
- 승인/거절 판단 기준은 **Sharpe Ratio 극대화**이며, accuracy·AUC 등 일반 분류 지표로 대체하지 않는다.
  - 예외: 처리 방식 간 **상대 비교**에는 AUC를 써도 된다. 금지되는 것은 AUC로 승인/거절 기준을 정하는 것이다.
- Test set으로 모형을 재조정하지 않는다 — Train/Validation으로 확정한 모형·threshold를 그대로 적용해 검증만 한다.
- 무거운 처리는 스크립트로 작성 후 결과만 요약해서 보고한다.
- 출력 위치: 리포트는 `outputs/reports/`, 차트는 `outputs/figures/`, **재현 스크립트가 내는 집계 CSV는 `outputs/` 바로 아래**에 둔다.
- 새 산출물을 만들면 위 "자료 위치 색인"에 한 줄 추가하고, 확정 사항이면 `decision_log.md`에 근거와 함께 남긴다.
