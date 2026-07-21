# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **⚠️ 아키텍처 변경 안내 (2026-07-21):** 이 문서는 `분석 에이전트`가 별도 작업 루트였던 옛 구조 기준으로 작성됐습니다. 그 폴더는 이후 삭제되었고, `Team Project` 루트 자체가 `data/raw`, `data/processed`, `src/{preprocessing,analysis,viz,utils}`, `notebooks`, `outputs`, `docs` 파이프라인 구조로 재편되었습니다. 아래 "절대 규칙"·"방법론"·"Sharpe Ratio 정의" 등은 여전히 유효한 팀 합의 사항이라 참고용으로 보존하지만, "아키텍처" 섹션의 경로 정보는 옛 구조 기준이라 최신화가 필요합니다 (아래 갱신 표 참조).

## 절대 규칙

- 승인/거절 **threshold는 오직 Sharpe Ratio 극대화**로 결정한다. accuracy, AUC 등 일반 분류 지표로 정하지 않는다.
- **거절한 대출을 표본에서 제외하지 않는다.** 거절 건은 국채(무위험자산) 투자로 간주해 수익률을 무위험수익률로 대체한다.
- Sharpe Ratio 표준편차는 **표본표준편차(ddof=1)**를 사용한다.
- **Test set으로는 모형을 재조정하지 않는다.** Train(60%)으로 확정한 모형과 Validation으로 확정한 threshold를 그대로 적용해 검증만 한다.
- **사전/사후(pre/post-approval) 변수 구분은 아직 미확정이다.** 현재 시점에는 이 구분을 모델링에 적용하지 말고, 팀에서 별도로 확정한 뒤 반영한다.
- `data/processed/v_desc_unified.xlsx`(구 경로: `../Team Project/변수 데이터/v_desc_unified.xlsx`)의 라벨 값을 **현재 상태 그대로 신뢰하지 않는다** (수정이 필요한 초안 상태).
- `int_rate`는 실현수익률 계산(목표/평가용)에는 사용하되, 사후 확정 변수이므로 **부도확률 예측 피처로 사용하지 않는다.**
- 원본 데이터(`lending_club_2020_train.csv`, ~1.2GB)를 이 폴더로 이관하더라도 **git 추적/커밋 대상에서 제외한다.**

## 아키텍처 (레거시 — 현재는 아래 새 구조로 대체됨)

~~이 폴더(`분석 에이전트`)는 프로젝트의 새 작업 루트다.~~ 이 서술은 더 이상 유효하지 않다.
`분석 에이전트` 폴더는 삭제되었고, `Team Project` 저장소 루트 자체가 데이터 파이프라인 구조
(`data/raw` → `src/preprocessing` → `data/processed` → `src/analysis` → `src/viz` → `outputs`)로
재편되었다. 아래 표는 옛 경로가 새 구조에서 어디로 이동했는지 정리한 것이다.

### 데이터 위치 (현재 구조 기준 갱신)

| 파일 | 현재 위치 | 설명 |
|---|---|---|
| 실제 원본 데이터 (175만 행, 150+ 컬럼) | `data/raw/lending_club_2020_train.csv` | Lending Club 2020 원본. 용량 큼(~1.2GB) — `.gitignore`로 git 추적/커밋 대상에서 제외됨. |
| 변수 사전/사후 라벨 (초안, 수정 필요) | `data/processed/v_desc_unified.xlsx` | 3개 AI 라벨 다수결 초안. 아직 확정 아님 — 수정 후 재확정 예정. |
| LC 공식 데이터 사전 | `data/raw/LCDataDictionary.xlsx` | 각 컬럼의 공식 정의. |
| 초기 탐색 노트북 | `notebooks/analysis_in_advance.ipynb` | `funded_amnt` vs `funded_amnt_inv` 등 초기 EDA. |
| **방법론 참고(연습용) 노트북** | `notebooks/LendingClub_실습_v2.ipynb` | 단순화된 샘플 데이터(1만 행, 11컬럼)로 전체 파이프라인(분할→모형→threshold→Sharpe 검증→30-seed 안정성 체크)을 시연. 실제 프로젝트 파이프라인의 구조적 템플릿으로 사용할 것 — 단, leakage 방지 기준은 위 절대 규칙으로 대체 적용. |
| 변수 사전 빌드 스크립트 | `src/preprocessing/build_v_desc_check.py`, `src/preprocessing/build_v_desc_unified.py` | `v_desc_unified.xlsx` 생성/검증 로직. |

### 기술 스택

참고 노트북(`LendingClub_실습_v2.ipynb`) 기준: `pandas`, `numpy`, `matplotlib`,
`scikit-learn` (`sklearn.linear_model.LogisticRegression`, `sklearn.model_selection.train_test_split`).

> 폴더 구조 트리, 가상환경/의존성 관리 방식(requirements.txt 등)은 추후 보충.

## 빌드/테스트

> 추후 보충 — 개발환경 설정, 노트북 실행 방법, `build_v_desc_check.py` / `build_v_desc_unified.py` 실행법(및 로직 재검토), 테스트 유무 등.

## 도메인 컨텍스트

서울대학교 핀테크 전문가 과정 "통계 데이터사이언스" 과목 팀 프로젝트. Lending Club 대출 데이터로
신용평가모형(부도확률 예측)을 구축하고, 이를 이용한 대출 승인/거절 전략의 **Sharpe Ratio를 극대화**하는
것이 목표다.

### 방법론

1. **분할**: 전체 데이터를 Train 60% / Validation 20% / Test 20%로 분할한다.
   (구현상으로는 `train_test_split`을 두 번 적용: 전체→80/20으로 Test 분리, 나머지 80%를 다시
   75/25로 Train/Validation 분리 = 결과적으로 60/20/20.)
2. **Train**: 부도확률(`bad`=1)을 예측하는 신용평가모형을 학습한다.
3. **Validation**: 승인/거절 임계점(threshold)을 결정한다 (Sharpe Ratio 극대화 기준 — 절대 규칙 참조).
4. **Test**: 최종 모형과 확정된 threshold로 Sharpe Ratio를 검증한다.

### Sharpe Ratio 정의

```
Sharpe = (포트폴리오 수익률 평균 - 국채수익률) / 포트폴리오 수익률의 표준편차
```

- 승인(예측=정상)한 대출: 실현수익률 = 부도 시 0, 정상상환 시 `int_rate`.
- 거절한 대출: 무위험수익률로 대체 (절대 규칙 참조).
- 참고 구현은 `sharpe_ratio(y_true, y_pred, X, risk_free)` 함수 형태 (아래 참고 노트북 참조).

### 데이터 누출(leakage) 개념 — 기준 자체는 아직 미확정

실제 신용평가모형은 궁극적으로 **대출 신청 시점에 이미 알 수 있는 변수(사전/pre-approval)만** 피처로
사용해야 한다. Lending Club이 심사 후에 결정하거나 대출 실행 이후에 발생하는 변수(사후/post-approval,
예: `grade`, `sub_grade`, `int_rate`, `installment`, `issue_d`, `loan_status`, `total_pymnt`류,
`hardship_*`, `settlement_*`, `last_pymnt_*` 등)를 피처로 쓰면 누출이다. 다만 이 구분 자체가 아직
팀에서 확정되지 않았으므로 현재는 모델링에 적용하지 않는다 (절대 규칙 참조).

### 참고 노트북 파이프라인 요약 (`LendingClub_실습_v2.ipynb`)

- 모형: `LogisticRegression(solver='newton-cholesky')`.
- Threshold 탐색: `np.linspace(0.01, 0.99, 99)` 전 구간을 순회하며 validation Sharpe가 최대인 지점 선택.
- Test 평가 전, threshold 확정 후 **Train 전체(60%)로 모형을 재학습**한 뒤 Test에 적용한다.
- 베이스라인 비교: "전부 승인" 전략의 Sharpe와 모형 전략의 Sharpe를 비교.
- 안정성 검증: `random_state` 0~29로 30회 반복하여 모형 Sharpe가 베이스라인을 이기는 비율을 확인.
- 무위험수익률(`rf`)은 연습 노트북에서 5% 고정값으로 가정 — 실제 프로젝트에서 국채수익률을 어떻게
  반영할지(고정값 vs 대출 발행 시점 기준 실제 국채수익률)는 팀에서 확정 필요.

## 코딩 컨벤션

> 추후 보충 — 파일명 규칙, 커밋 메시지 컨벤션, 시드 고정 규칙, 주석 언어, 결과물 저장 경로 규칙 등.
