# 분석 규칙

## 입력/출력
- 입력: `data/processed/` 전처리 완료 데이터
- 출력: `outputs/reports/` (모델·통계 결과), 필요 시 중간 산출물

## 규칙
- **모형은 XGBoost 하나다** (`decision_log.md` #17 ①, 2026-07-29 회의 확정). 로지스틱 회귀는 폐기됐다.
- **대조군은 전부 승인(approve-all) 전략**이다 (#17 ②). 성과 보고의 헤드라인은 절대 Sharpe가 아니라
  **Δ Sharpe = (모형 전략) − (approve-all)** 이다 — 재투자 가정이 양쪽을 똑같이 밀어올리므로
  절대값은 가정 효과를 포함한다(#18).
- 승인/거절 threshold는 **오직 Sharpe Ratio 극대화**로 결정한다. accuracy, AUC 등 일반 분류 지표로 결정하지 않는다.
  - 단, **탐색·검증 목적의 상대 비교**(전처리 방식 A vs B 중 무엇이 나은지)에는 AUC를 써도 된다. 금지되는 것은 AUC로 **승인/거절 기준을 정하는 것**이다. 기존 비교 스크립트 4종이 이 예외에 해당한다.
- **표본은 만기 + 버퍼 6개월 필터를 적용한 723,563건**이다 (#16). `issue_d + term ≤ 2020-04`.
- Sharpe Ratio 정의: `(포트폴리오 수익률 평균 − 무위험수익률) / 포트폴리오 수익률의 표본표준편차(ddof=1)`.
  - **재투자 가정(확정, #18)**: 매달 받는 상환액을 **잔존기간에 맞춘 국채**에 재투자한다.
    `R = (W/P)^(12/T) − 1`, `W = Σ CFₘ·F(m,T)`. `m`월 수령액은 잔존기간 `T−m` 만기 금리로 굴린다
    (대출 만기 금리를 일괄 적용하면 짧은 잔존기간에 긴 만기 금리를 주게 되어 무위험이 아니다).
    **정상상환·부도 양쪽에 동일 적용**하므로 정상상환에 `int_rate`를 그대로 쓰지 않는다.
  - ⚠️ **부도 시 실현수익률의 최종 정의는 여전히 팀 확정 전이다** (B팀 담당, #19).
    구조 A(PD 분위별 그룹 평균, 권고) vs B(2단계 hurdle 건별 회귀), 국채 금리 기준(ⓒ발행시점 고정 권고),
    서비스수수료(~1%) 반영 방식이 남아 있다.
    - 따라서 **부도 손실값을 코드에 상수로 박지 않는다.** 실현수익률 계산은 손실 정의를 **주입받는 형태**
      (파라미터/전략 함수)로 짜서, 안을 갈아 끼우며 Sharpe를 비교할 수 있게 한다.
    - **분류 모델은 이 정의를 기다리지 않고 만들 수 있다** — 타깃이 이진 `loan_status`라 손실 정의가
      학습에 개입하지 않는다. 정의 확정 시 **threshold만 다시 탐색**하면 되고 재학습은 불필요하다(#19).
    - ⚠️ "부도 시 0"이라는 옛 표기를 만나면 **회수액 0(= 수익률 -100%)** 로 읽는다. 수익률 0%(원금 전액 회수)가 아니다 — 이 혼동이 문서 4곳에 퍼져 있었고 2026-07-29에 정정했다(#4 정정 기록).
  - 거절한 대출의 실현수익률: 무위험수익률로 대체.
- 데이터 분할: Train 60% / Validation 20% / Test 20% (`train_test_split`을 두 번 적용 — 전체→80/20으로 Test 분리 후, 남은 80%를 75/25로 Train/Validation 분리).
  - 비율·seed는 **`config/config.yaml`이 단일 출처**다. 값을 스크립트에 직접 써 넣지 말고 `src/utils/config.py`의 `load_config()`로 읽는다.
  - 랜덤 분할을 유지한다 — 시간순 분할은 `decision_log.md` #13 ④에서 실측으로 기각됐다(Train의 T2 관측률이 0%가 됨). 필요하면 보조 진단으로만 돌린다.
- Train으로 부도확률 예측 모형을 학습하고, Validation으로 threshold를 확정한다.
- **Test set으로 모형을 재조정하지 않는다.** Train으로 학습한 모형과 Validation으로 확정한 threshold를 그대로 적용해 검증만 한다.
- Threshold 확정 후에는 Train 전체(60%)로 모형을 재학습한 뒤 Test에 적용한다.
- **사전/사후(pre/post-approval) 변수 구분을 피처 선택에 적용한다** (`decision_log.md` #1 확정). 투자자 관점이므로 `grade`·`sub_grade`·`int_rate`·`installment`·`funded_amnt`·`funded_amnt_inv`·`issue_d`·`initial_list_status`는 **사전 변수**다. 라벨 원본은 `data/processed/variable_dictionary_byGJ.xlsx`의 `is_pre_approval`.
  - **LC 조건변수(`grade`·`sub_grade`·`int_rate`)는 피처로 투입한다** (#17 ③, 2026-07-29 회의 확정).
    ⚠️ 이에 따라 **최종 모형의 AUC 기준은 0.71대**다. 기존 문서의 0.68대는 조건변수를 뺀 Lean 스펙 값이므로 섞어 인용하지 않는다.
- **무위험수익률(Rf) — 참고논문 방식 확정** (#18): 거절한 대출의 자본은 **발행시점(`issue_d`)에 대출 만기와
  만기를 맞춘 미국채**(3년물/5년물)에 투자했다고 가정한다. 데이터는
  `data/processed/us_treasury_GS3_GS5_monthly_2007-06_to_2020-09.csv`.
  - 고정 상수가 아니므로 `config/config.yaml`의 `risk_free_rate.value`는 **`null`을 유지**하고,
    `issue_d` × `term` 매칭 로직으로 처리한다. `value`를 읽어 쓰는 코드를 작성하지 않는다.
- **Threshold 확정 절차**: 랜덤 6:2:2 분할을 **K=50회 반복**하고 `τ*`의 분포(평균·표준편차)를 본다 (#18).
- 안정성 검증 시 `random_state` 0~29로 30회 반복해 모형 Sharpe가 "전부 승인" 베이스라인을 이기는 비율을 확인한다.
- 모형 구성은 **통합 모형 + `term` 피처 투입**이다 — 36m/60m 분리 모형은 `decision_log.md` #13 ②에서 실측 기각(60m에서 AUC −0.00564, 5/5 seed). 단 성능표는 term별로 나눠 보고한다.
  - ⚠️ 이는 **PD 모형**을 나누지 않는다는 뜻이며, IRR·Sharpe 계산식에는 term별 현금흐름 기간이 반드시 들어간다(#4).
  - XGBoost 단일화가 확정됐으므로(#17 ①) 이 결론은 **조건부가 아니라 확정**이다.
- **결측 처리는 한 줄로 확정됐다** (#13 ③·#17): **모든 결측을 NaN 그대로 투입한다. 대체하지 않고 더미도 만들지 않는다.**
  표준화·로그변환·구간화·캡핑도 하지 않는다 — 트리는 값의 순서만 쓴다.

## 이 폴더 스크립트의 성격 구분
`src/analysis/`에는 성격이 다른 두 종류가 섞여 있다. 혼동하면 규칙을 잘못 적용한다.

| 종류 | 파일 | 성격 |
| --- | --- | --- |
| **탐색·검증** (현재 전부 이쪽) | `preprocessing_validation.py`, `missing_scheme_comparison.py`, `term_split_comparison.py`, `t2_contribution_reassessment.py`, `macro_indicator_screening.py` | 문서의 표를 재생성하는 재현 스크립트. 자체 2분할·자체 seed 루프를 쓰며 `config.yaml`을 따르지 않는다(의도된 차이). AUC는 상대 비교용. |
| **본 파이프라인** (아직 뼈대) | `model.py`, `sharpe_optimizer.py` | 위 "규칙" 절이 그대로 적용되는 대상. `config.yaml`을 반드시 경유한다. |

> 인용 주의: 비교 스크립트가 내는 AUC(0.68대)는 LC 조건변수를 뺀 Lean 스펙 값이다.
> **최종 모형 성능으로 인용하면 안 된다** (`decision_log.md` #13 ⑤).
> 회의에서 조건변수 투입이 확정됐으므로(#17 ③) **최종 모형 기준은 0.71대**다.
>
> `macro_indicator_screening.py`는 거시지표 미사용 확정(#15)으로 **과거 산출물 재현 전용**이 됐다.

## 참고
- **확정 사항의 원본은 `outputs/reports/decision_log.md`다.** 이 문서와 어긋나면 decision_log가 우선이고, 이 문서를 고친다.
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md` (⚠️ 과거 폴더 구조 기준이라 문서 내 경로 상당수가 현존하지 않는다 — 방법론만 참고할 것)
- 파이프라인 구조적 템플릿: `notebooks/LendingClub_실습_v2.ipynb`
- 실행 환경: 저장소 루트 `requirements.txt` (scikit-learn·xgboost 필요 — 시스템 python에는 없다)
