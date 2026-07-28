# Lending Club 변수 전처리 기준 검증 (ymg)

## 문서 목적

`lending_club_변수분류_류성환.xlsx_v2.numbers`의 변수 분류와 전처리 가이드를 전체
`lending_club_2020_train.csv`에 대조하고, `loan_status` 부도확률을 예측하는 XGBoost
모델에 맞게 수정·보완할 사항을 정리한다.

이 문서는 기존 변수 사전인 `data/processed/variable_dictionary_byGJ.xlsx`를 대체하는
확정본이 아니다. 시간순 분할, `Default` 처리 등은 팀 합의가 필요한 제안으로 구분한다.

## 1. 첨부 파일의 컬럼 정보 및 가이드 핵심 요약

### 1.1 시트와 변수 분류

첨부 Numbers 문서는 다음 4개 시트로 구성된다.

| 시트 | 역할 |
| --- | --- |
| `요약` | 변수 그룹, 결측 티어, 제외 기준 및 전처리 방향 요약 |
| `유지변수_84` | 예측변수 83개와 타겟 `loan_status` 1개 |
| `제외변수_57` | 누수·구조적 결측·저분산·식별자 등의 제외 근거 |
| `현금흐름표_구성` | IRR·회수율 계산에 사용하는 사후 현금흐름 변수 |

원본 CSV의 141개 컬럼은 다음과 같이 빠짐없이 분류된다.

| 구분 | 컬럼 수 | 설명 |
| --- | ---: | --- |
| 예측변수 X | 83 | A 22개, B 40개, C 20개, `issue_d` 1개 |
| 타겟 y | 1 | `loan_status` |
| 제외변수 | 57 | 사후 누수 31, 공동신청 관련 15, 저분산 6, 자유텍스트·중복 3, 식별자·무정보 2 |
| 합계 | 141 | 전체 CSV 컬럼 수와 일치 |

결측 티어는 T0 28개, T1 34개, T2 14개, T3 6개로 분류되어 있다. 예측 시점에
알 수 없는 `total_pymnt`, `recoveries`, `last_pymnt_d`, 사후 FICO 등의 변수를
부도확률 X에서 제외한 것은 타당하다. 이 변수들은 사후 현금흐름·실현수익률 계산에만
사용해야 한다.

### 1.2 전체 데이터 대조 결과

전체 CSV는 파서상 1,755,295행이지만, 1행은 `id`에 설명문만 들어 있는 대출 외 메타행이다.
따라서 실제 대출은 1,755,294건이며, 메타행을 제거한 뒤 타입 변환을 수행해야 한다.

주요 `loan_status` 분포는 다음과 같다.

| 상태 | 건수 | 주 분석 처리 |
| --- | ---: | --- |
| Fully Paid | 898,522 | `y=0` |
| 정책미충족 Fully Paid | 1,223 | `y=0` |
| Charged Off | 217,366 | `y=1` |
| 정책미충족 Charged Off | 460 | `y=1` |
| Default | 268 | 주 분석 제외, 민감도 분석에서 `y=1` |
| Current | 618,688 | 결과 미확정으로 제외 |
| Late·Grace·Issued | 18,767 | 결과 미확정으로 제외 |

정책미충족 별칭을 본 상태와 합친 strict 타겟은 다음과 같다.

- 정상: 899,745건
- 부도: 217,826건
- 합계: 1,117,571건
- 부도율: 19.491%

`Default` 268건을 부도에 포함하는 민감도 표본은 1,117,839건이며 부도율은 19.510%다.
전체 비율 차이는 작지만 타겟 정의의 일관성을 위해 팀 결정이 필요하다.

### 1.3 첨부 가이드에서 바로잡을 사항

1. **FICO 구간 폭**
   - 첨부 설명의 30점과 달리 strict 표본에서
     `fico_range_high - fico_range_low`는 1,117,380건이 4점, 191건이 5점이다.
   - `fico_avg` 파생은 가능하지만 30점이라는 설명은 삭제해야 한다.
2. **공동신청 변수**
   - `Joint App`은 126,357건으로 전체 실제 대출의 7.20%다.
   - 규모가 작다고 단정해 15개 전용 변수를 자동 제외하기보다 포함/제외 challenger를 비교한다.
3. **희소 사건 변수**
   - `num_tl_120dpd_2m`, `num_tl_30dpd`, `delinq_amnt`, `acc_now_delinq`의 양수 사건은
     각각 777~5,406건이다.
   - 완전 상수와 달리 희소하지만 강한 위험 신호일 수 있으므로 자동 제외하지 않는다.
4. **T2/T3 결측**
   - strict 표본에서 11개 T2 동시수집 블록의 결측률은 약 45.59%다.
   - 결측이 발행연도와 수집제도 변화를 대리하는지 시간 외 검증이 필요하다.
5. **저장소 결정과의 정합성**
   - 모호한 `D_X` 대신 `is_missing`, `is_observed`, `has_event`처럼 방향이 명확한 이름을 쓴다.
   - 거시 금리차는 월별 `GS10-GS2`, 무위험수익률은 월별 `GS3`·`GS5`를 사용한다.
   - HPR 연율화 선택지는 기존 decision log에서 기각되었으므로 전처리 가이드에서 제거한다.

## 2. XGBoost에 맞게 수정·보완할 전처리

### 2.1 결측치

XGBoost의 tree booster는 수치형 `NaN`의 기본 분기 방향을 학습할 수 있다. 따라서 일괄
중앙값·최빈값 대체를 기본 단계로 두지 않는다.

| 결측 유형 | 권장 기본선 | 비교할 challenger |
| --- | --- | --- |
| T1 일반 결측 | 수치형 `NaN` 유지 | train 중앙값 대체 |
| T2 구조적·수집시기 결측 | `NaN` + 선택적 `is_collected` | 명시적 결측 플래그 포함/제외 |
| T3 사건없음형 | `NaN` + 의미가 검증된 `has_event` | 기존 masked 방식 |
| 범주형 결측 | 고정된 `Missing` 범주 또는 native categorical 정책 | 희소 OHE |

결측을 숫자 0으로 일괄 변환하지 않는다. 0이 실제 관측값인 변수에서 의미가 섞일 수 있기
때문이다. 결측 플래그는 발행연도·수집제도 프록시 여부를 함께 점검한다.

### 2.2 인코딩

범주형은 다음 두 파이프라인을 비교한다.

1. pandas `category`, `enable_categorical=True`, `tree_method="hist"`를 사용하는
   XGBoost native categorical
2. Train에서만 범주를 학습하고 미지 범주를 처리하는 sparse One-Hot Encoding

`addr_state`는 51개 범주여서 native categorical 또는 OHE로 처리할 수 있다. Target
Encoding은 기본안에서 제외하고, 필요할 경우 Train 내부 out-of-fold 방식만 허용한다.
동일 행의 타겟을 인코딩에 직접 사용하면 누수가 발생한다.

Native categorical을 채택하면 학습·추론의 category vocabulary와 dtype을 함께
버전 관리하고 모델은 JSON 또는 UBJSON으로 저장한다.

### 2.3 스케일링·로그변환·구간화

- XGBoost tree booster에는 StandardScaler·MinMaxScaler를 적용하지 않는다.
- 금액형 로그변환과 winsorization은 필수가 아니다. 오류값만 정정한 원본을 기본선으로 둔다.
- count 변수의 0/1/2+ 구간화나 이진화보다 원 count를 우선 보존한다.
- 로그값·이진 플래그는 원본과 비교해 시간 외 Validation Sharpe와 확률보정을 개선할 때만 채택한다.
- 로지스틱 회귀 등 선형 baseline은 별도 스케일링 파이프라인으로 관리한다.

### 2.4 타입·단위·파생변수

- `int_rate`, `revol_util`: `%`를 제거해 숫자로 변환한다. 수익률 계산용 `int_rate`는
  별도로 소수율로 변환한다.
- `term`: `36 months`, `60 months`를 고정 범주 또는 36/60 숫자로 변환한다.
- `emp_length`: 순서형 숫자+결측 플래그와 native category를 비교한다.
- `earliest_cr_line`: `issue_d`와의 차이인 `credit_history_months`를 만들고 미래 날짜와
  음수를 오류로 검사한다.
- `fico_avg=(fico_range_low+fico_range_high)/2`를 후보로 만들되 원본 두 열 제거 여부는
  ablation으로 정한다.
- `installment/annual_inc`, `loan_amnt/annual_inc` 등의 비율은 분모 0과 결측을 명시적으로
  처리하고 경제적 의미가 있을 때만 추가한다.

`dti` 최대 999, `revol_util` 최대 366.6, `bc_util` 최대 318.2, `il_util` 최대 558이
관측되므로 일괄 절단하기보다 도메인 오류 여부를 확인하고 원본/상한처리 민감도 분석을 한다.

### 2.5 중복·공선성·피처 사양

`loan_amnt`와 `funded_amnt`의 상관계수는 약 0.999686,
`funded_amnt`와 `funded_amnt_inv`는 약 0.999136이다. 트리 모델에서는 다중공선성을
이유로 반드시 하나만 남길 필요는 없지만, 계산량과 importance 해석을 위해 다음 사양을
비교한다.

- **Full 모델:** `grade`, `sub_grade`, `int_rate`, `funded_amnt` 등 투자자가 발행 시점에
  확인할 수 있는 Lending Club 조건변수 포함
- **Lean 모델:** LC 조건변수를 제외해 자체 신청자 정보의 추가 설명력 측정
- **Joint challenger:** 공동신청 전용 15개 변수 포함
- **Rare-event challenger:** 희소 사건 변수 4개 포함

최종 피처 사양은 Validation의 Sharpe와 확률보정 품질을 기준으로 고른다.

## 3. 타겟·전체 데이터 특성을 고려한 분할 및 학습 전략

### 3.1 결과 미확정과 seasoning 편향

발행연도별 strict 결과 확정 비중은 다음과 같이 감소한다.

| 발행연도 | 결과 확정 비중 |
| ---: | ---: |
| 2016 | 92.77% |
| 2017 | 70.81% |
| 2018 | 39.70% |
| 2019 | 14.56% |
| 2020 | 2.92% |

최신 코호트에서 Fully Paid·Charged Off만 남기면 빨리 완납하거나 빨리 부도난 대출이
과대표집된다. 무작위 stratified split은 동일한 발행환경과 수집제도 패턴을 Train과
Test에 섞으므로 실제 미래 성능을 낙관적으로 평가할 수 있다.

### 3.2 권장 분할

파일의 마지막 발행월을 관측 종료 2020-09로 가정한 최소 성숙도 조건은 다음과 같다.

- 36개월 대출: `issue_d <= 2017-09`
- 60개월 대출: `issue_d <= 2015-09`

이 조건을 strict 타겟에 적용하면 814,621건, 부도 138,337건, 부도율 16.98%가 남는다.
실제 charge-off 인식 지연을 반영하려면 관측 종료일을 확인한 뒤 추가 버퍼를 검토한다.

월 코호트를 쪼개지 않는 약 60/20/20 시간순 분할 결과는 다음과 같다.

| Partition | 발행기간 | 건수 | 부도율 | 용도 |
| --- | --- | ---: | ---: | --- |
| Train | ~2015-11 | 483,626 | 17.54% | 학습·내부 튜닝 |
| Validation | 2015-12~2016-09 | 165,067 | 16.04% | 모델·Sharpe threshold 선택 |
| Test | 2016-10~2017-09 | 165,928 | 16.29% | 최종 1회 검증 |

term별 성숙도 cutoff 때문에 후기 Test는 36개월 대출 중심이 될 수 있다. 36개월·60개월을
별도 모델로 학습하거나 최소한 term별 성능표를 반드시 보고한다.

현재 저장소의 stratified random 60/20/20은 기존 결과 재현과 디버깅용 baseline으로
유지할 수 있지만 시간 일반화 검증을 대체하지 않는다. 시간순 분할을 공식 규칙으로 바꾸려면
`src/analysis/AGENTS.md`와 decision log를 팀 합의 후 함께 수정해야 한다.

### 3.3 XGBoost 학습·검증 순서

1. Train 내부를 시간순 fit/early-stop 구간으로 다시 나누어 하이퍼파라미터와
   `best_iteration`을 정한다.
2. 필요하면 Train 내부 out-of-fold 예측으로 sigmoid 또는 isotonic calibration을 학습한다.
3. 고정된 모델과 calibrator로 공식 Validation의 PD를 예측한다.
4. 공식 Validation에서는 승인 threshold를 Sharpe Ratio로 선택한다.
5. 모델·calibrator·threshold를 고정한 뒤 Test에 한 번만 적용한다.

권장 시작점은 `objective="binary:logistic"`, `tree_method="hist"`,
`n_estimators`를 충분히 크게 둔 early stopping 방식이다. 부도율 19.49%는 극단적
불균형이 아니므로 확률모형 기본선은 `scale_pos_weight=1`로 둔다. 정상/부도 비율인
약 4.13은 challenger로만 비교하고, 가중 모델을 쓰면 calibration curve, Brier score,
logloss로 재보정을 확인한다.

ROC-AUC와 Average Precision은 진단 지표로 사용한다. 최종 모델과 승인 정책은 accuracy가
아니라 Validation Sharpe로 선택한다.

### 3.4 저장·재현 항목

다음 항목을 모델과 함께 저장한다.

- Train/Validation/Test 인덱스와 월 경계
- 원본 CSV SHA-256:
  `52bc9369518d409e8742808e8b93e8472fca73a5df0f84042450a33c482aad14`
- XGBoost·pandas·scikit-learn 버전
- category vocabulary와 미지 범주 정책
- 전체 하이퍼파라미터와 `best_iteration`
- calibrator와 Sharpe 승인 threshold
- Full/Lean/Joint/Rare-event ablation 결과

원본 CSV와 첨부 Numbers 파일은 Git에 커밋하지 않는다.

## 팀 협의 필요 사항

1. `Default` 268건을 주 분석의 부도로 포함할지
2. 실제 데이터 snapshot 종료일과 term별 성숙도 버퍼
3. 최종상태 분류를 유지할지, 동일 성과관측기간의 horizon target을 만들지
4. 36개월·60개월을 하나의 모델로 학습할지
5. 시간순 분할을 프로젝트 공식 60/20/20 규칙으로 채택할지
6. Full 모델을 주 모델로 두고 Lean 모델을 해석용 challenger로 둘지

## 참고자료

- XGBoost FAQ — Missing values:
  <https://xgboost.readthedocs.io/en/stable/faq.html>
- XGBoost categorical data:
  <https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html>
- XGBoost parameters:
  <https://xgboost.readthedocs.io/en/stable/parameter.html>
- scikit-learn probability calibration:
  <https://scikit-learn.org/stable/modules/calibration.html>
- 저장소 결정 기록: `outputs/reports/decision_log.md`

