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

| 위치 | 내용 | 상태 |
| --- | --- | --- |
| `src/preprocessing/fetch_macro_*.py` | 거시지표 수집 (다운로드+검증+저장) | 동작 |
| `src/preprocessing/label_pre_post_by_rule.py` | 규칙 기반 사전/사후 라벨링 | 동작 |
| `src/analysis/macro_indicator_screening.py` | 거시지표 탐색 분석 (문서 표 재생성) | 동작 |
| `src/preprocessing/loader.py`, `preprocessor.py` | 로딩·전처리 파이프라인 | **뼈대(TODO)** |
| `src/analysis/model.py`, `sharpe_optimizer.py` | 모형 학습·threshold 탐색 | **뼈대(TODO)** |
| `src/viz/plots.py` | 차트 생성 | **뼈대(TODO)** |
| `src/utils/logger.py` | 공통 로깅 | 동작 |

> 뼈대 파일은 팀이 방법론을 확정하기 전이라 의도적으로 비워둔 것이다.
> 채우기 전에 `decision_log.md`에서 해당 항목이 확정됐는지 먼저 확인한다.

### 규격·컨벤션

| 문서 | 내용 |
| --- | --- |
| `docs/macro_indicators_spec.md` | 외부 데이터 수집 규격 (기간·컬럼·검증·출처 카드) |
| `docs/GIT_CONVENTION.md` | 브랜치·커밋·PR 규칙 |
| `docs/architecture.md` | 파이프라인 구조 (Mermaid) |
| `config/config.yaml` | 경로·분할 비율 등 공통 설정 |
| `notebooks/` | 실습 템플릿·사전 탐색 (참고용, 산출물 아님) |

## 공통 원칙
- `data/raw/`는 절대 수정하지 않는다.
- 원본 대용량 CSV(`data/raw/` 내 대출 원본 데이터, ~1.2GB)는 git에 커밋하지 않는다 (`.gitignore` 참고). 팀원 공유는 별도 채널(Google Drive 등)로 한다.
- 승인/거절 판단 기준은 **Sharpe Ratio 극대화**이며, accuracy·AUC 등 일반 분류 지표로 대체하지 않는다.
- Test set으로 모형을 재조정하지 않는다 — Train/Validation으로 확정한 모형·threshold를 그대로 적용해 검증만 한다.
- 무거운 처리는 스크립트로 작성 후 결과만 요약해서 보고한다.
- 출력 리포트는 `outputs/reports/`에, 차트는 `outputs/figures/`에 저장한다.
