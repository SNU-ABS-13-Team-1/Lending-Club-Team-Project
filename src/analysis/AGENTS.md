# 분석 규칙

## 입력/출력
- 입력: `data/processed/` 전처리 완료 데이터
- 출력: `outputs/reports/` (모델·통계 결과), 필요 시 중간 산출물

## 규칙
- 승인/거절 threshold는 **오직 Sharpe Ratio 극대화**로 결정한다. accuracy, AUC 등 일반 분류 지표로 결정하지 않는다.
  - 단, **탐색·검증 목적의 상대 비교**(전처리 방식 A vs B 중 무엇이 나은지)에는 AUC를 써도 된다. 금지되는 것은 AUC로 **승인/거절 기준을 정하는 것**이다. 기존 비교 스크립트 4종이 이 예외에 해당한다.
- Sharpe Ratio 정의: `(포트폴리오 수익률 평균 − 무위험수익률) / 포트폴리오 수익률의 표본표준편차(ddof=1)`.
  - 승인한 대출의 실현수익률: 정상상환 시 `int_rate`, 부도 시 **-100%**(원금 전액 손실) — **현재 베이스라인**.
    - ⚠️ **부도 시 실현수익률의 최종 정의는 팀 확정 전이다.** ① -100% 유지(PD 단독 threshold) ② 부도 건별 IRR 근사 ③ grade×term 평균 대체 — 세 갈래가 열려 있다(`decision_log.md` #4 "남은 판단"·#5). 서비스수수료(~1%) 반영 방식도 별도 결정 사항.
    - 따라서 **부도 손실값을 코드에 상수로 박지 않는다.** 실현수익률 계산은 손실 정의를 **주입받는 형태**(파라미터/전략 함수)로 짜서, 세 안을 갈아 끼우며 Sharpe를 비교할 수 있게 한다.
    - #4가 확정한 것은 **IRR을 쓸 경우의 계산 방법**(3단계 현금흐름 재구성, 단순 ×12 연율화)이지 IRR을 쓴다는 선택 자체가 아니다 — 혼동하지 말 것.
    - ⚠️ "부도 시 0"이라는 옛 표기를 만나면 **회수액 0(= 수익률 -100%)** 로 읽는다. 수익률 0%(원금 전액 회수)가 아니다 — 이 혼동이 문서 4곳에 퍼져 있었고 2026-07-29에 정정했다(#4 정정 기록).
  - 거절한 대출의 실현수익률: 무위험수익률로 대체.
- 데이터 분할: Train 60% / Validation 20% / Test 20% (`train_test_split`을 두 번 적용 — 전체→80/20으로 Test 분리 후, 남은 80%를 75/25로 Train/Validation 분리).
  - 비율·seed는 **`config/config.yaml`이 단일 출처**다. 값을 스크립트에 직접 써 넣지 말고 `src/utils/config.py`의 `load_config()`로 읽는다.
  - 랜덤 분할을 유지한다 — 시간순 분할은 `decision_log.md` #13 ④에서 실측으로 기각됐다(Train의 T2 관측률이 0%가 됨). 필요하면 보조 진단으로만 돌린다.
- Train으로 부도확률 예측 모형을 학습하고, Validation으로 threshold를 확정한다.
- **Test set으로 모형을 재조정하지 않는다.** Train으로 학습한 모형과 Validation으로 확정한 threshold를 그대로 적용해 검증만 한다.
- Threshold 확정 후에는 Train 전체(60%)로 모형을 재학습한 뒤 Test에 적용한다.
- **사전/사후(pre/post-approval) 변수 구분을 피처 선택에 적용한다** (`decision_log.md` #1 확정). 투자자 관점이므로 `grade`·`sub_grade`·`int_rate`·`installment`·`funded_amnt`·`funded_amnt_inv`·`issue_d`·`initial_list_status`는 **사전 변수**다. 라벨 원본은 `data/processed/variable_dictionary_byGJ.xlsx`의 `is_pre_approval`.
  - 단, 위 LC 조건변수를 실제 피처로 **투입할지**는 별개 판단이며 아직 미확정이다(LC 등급을 그대로 베끼는 문제). 포함 여부에 따라 AUC가 0.68 ↔ 0.71대로 갈린다.
- 무위험수익률(`rf`)은 참고 노트북에서 5% 고정값으로 가정했으나, 실제 프로젝트에서 고정값을 쓸지 대출 발행 시점 기준 실제 국채수익률을 쓸지는 팀 확정 필요.
  - 확정 전까지 `config/config.yaml`의 `risk_free_rate.value`는 `null`이다. `cfg.require("risk_free_rate.value")`로 읽어 **미확정이면 예외로 중단**시킨다 — 임의의 기본값이 결과에 섞이지 않게 한다.
- 안정성 검증 시 `random_state` 0~29로 30회 반복해 모형 Sharpe가 "전부 승인" 베이스라인을 이기는 비율을 확인한다.
- 모형 구성은 **통합 모형 + `term` 피처 투입**이다 — 36m/60m 분리 모형은 `decision_log.md` #13 ②에서 실측 기각(60m에서 AUC −0.00564, 5/5 seed). 단 성능표는 term별로 나눠 보고한다.
  - ⚠️ 이는 **PD 모형**을 나누지 않는다는 뜻이며, IRR·Sharpe 계산식에는 term별 현금흐름 기간이 반드시 들어간다(#4).
  - 이 결론은 **XGBoost 단일화**를 전제로 한다. 모형 후보는 아직 확정 전이므로(#3 진행 중), 로지스틱이 되살아나면 #13의 결측 관련 결론이 전부 원복된다.

## 이 폴더 스크립트의 성격 구분
`src/analysis/`에는 성격이 다른 두 종류가 섞여 있다. 혼동하면 규칙을 잘못 적용한다.

| 종류 | 파일 | 성격 |
| --- | --- | --- |
| **탐색·검증** (현재 전부 이쪽) | `preprocessing_validation.py`, `missing_scheme_comparison.py`, `term_split_comparison.py`, `t2_contribution_reassessment.py`, `macro_indicator_screening.py` | 문서의 표를 재생성하는 재현 스크립트. 자체 2분할·자체 seed 루프를 쓰며 `config.yaml`을 따르지 않는다(의도된 차이). AUC는 상대 비교용. |
| **본 파이프라인** (아직 뼈대) | `model.py`, `sharpe_optimizer.py` | 위 "규칙" 절이 그대로 적용되는 대상. `config.yaml`을 반드시 경유한다. |

> 인용 주의: 비교 스크립트가 내는 AUC(0.68대)는 LC 조건변수를 뺀 Lean 스펙 값이다.
> **최종 모형 성능으로 인용하면 안 된다** (`decision_log.md` #13 ⑤).

## 참고
- **확정 사항의 원본은 `outputs/reports/decision_log.md`다.** 이 문서와 어긋나면 decision_log가 우선이고, 이 문서를 고친다.
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md` (⚠️ 과거 폴더 구조 기준이라 문서 내 경로 상당수가 현존하지 않는다 — 방법론만 참고할 것)
- 파이프라인 구조적 템플릿: `notebooks/LendingClub_실습_v2.ipynb`
- 실행 환경: 저장소 루트 `requirements.txt` (scikit-learn·xgboost 필요 — 시스템 python에는 없다)
