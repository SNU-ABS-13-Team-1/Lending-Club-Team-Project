# 전처리 규칙

## 입력/출력
- 입력
  - `data/raw/`의 원본 대출 데이터 CSV (원본, 절대 수정 금지)
  - `data/raw/`의 공식 데이터 사전(Data Dictionary) Excel
  - 외부 공개 통계(FRED 등) — 스크립트가 직접 다운로드한다. 손으로 받아 넣지 않는다.
- 출력 (모두 `data/processed/`)
  - `*.xlsx` — 변수 사전/라벨 산출물
  - `*.csv` — 외부에서 수집한 시계열(거시경제지표 등)
  - `*.source.md` — **출처 카드.** 데이터 파일과 같은 이름으로 반드시 함께 만든다.
    담을 내용과 형식은 `docs/macro_indicators_spec.md` 2.8 참고 (출처·시리즈 ID·받은 날짜·
    SHA-256·다운로드 URL·검증 방법·발표시차·개정 주의사항).

## 외부 데이터 수집 규칙
- 수집은 `src/preprocessing/fetch_*.py` 스크립트로만 한다 — 손으로 CSV를 편집하지 않는다.
  스크립트 하나가 다운로드 → 검증 → 저장 → 기술통계 출력까지 수행한다.
- 검증은 스크립트에 내장하고, 실패하면 **파일을 쓰지 않고 예외로 중단**한다.
  행 수·기간·주기 연속성·결측뿐 아니라 **알려진 사건과 값을 대조**하는 검증을 포함한다.
- 기대값은 "지금 받으면 나오는 값" 기준으로 적는다 — 옛 뉴스의 당시 발표값을 쓰면
  개정된 정상 데이터가 실패 판정을 받는다 (`docs/macro_indicators_spec.md` 2.9 참고).
- 발표 시차(publication lag)가 있는 지표는 **시차를 적용하지 않은 원시 시계열로 저장**하고,
  실제 발표 시차를 조사해 리포트에 기록한다. lag 적용 여부는 결합 단계에서 일괄 결정한다.
- 수집 경위·검증 결과는 `outputs/reports/macro_{지표}.md` 형식의 리포트로 남긴다.

## 규칙
- 거절된 대출을 표본에서 제외하지 않는다. 거절 건은 무위험자산(국채) 투자로 간주해 실현수익률을 무위험수익률로 대체한다 — 승인 건만 남기는 필터링 금지.
- 행 필터링(샘플링): `loan_status`가 `Current`, `Late (16-30 days)`, `Late (31-120 days)`인 행은 표본에서 제외한다. `Does not meet the credit policy. Status:Fully Paid`/`Status:Charged Off`는 각각 `Fully Paid`/`Charged Off`와 동일하게 취급한다 (배경/근거: `README.md`의 "샘플링(행 필터링) 확정 사항" 참고).
  - **만기 기준 필터를 함께 적용한다 — 확정** (`decision_log.md` #16, 2026-07-29 회의): `issue_d + term ≤ 2020-04` (만기 + 버퍼 6개월). 스냅샷(2020-10) 시점에 만기가 도래하지 않은 완결건은 조기부도가 과대표집돼 부도율이 8.5%p 높기 때문이다. 결과는 **723,563건**, 구간 내 `Current` 0.06%.

### ⚠️ 두 모집단을 구분한다 — 건수가 다른 것은 오류가 아니다

| 모집단 | 건수 | 무엇에 쓰나 |
| --- | ---: | --- |
| **PD 모형 학습 모집단** | **723,563** | 위 필터를 통과한 완결건 전부. 부도확률 예측 모형의 학습·검증 대상 |
| **실현수익률 산출 모집단** | **721,809** | 위에서 현금흐름으로 `R`을 계산할 수 있는 건. 계산 불가 **1,754건**을 제외한 값 |

두 숫자가 다르다고 어느 한쪽이 틀린 게 아니다. **제외된 1,754건은 버리지 말고 사유와 함께 감사(audit) 테이블로 보존**한다 — 나중에 "왜 건수가 줄었나"에 답할 수 있어야 한다.
- **사전/사후(pre/post-approval) 변수 라벨은 확정됐다** (`decision_log.md` #1). 원본은 `data/processed/variable_dictionary_byGJ.xlsx`의 `is_pre_approval` 컬럼이며, 이 값을 기준으로 사후 변수를 제외한다.
  - ⚠️ **시트를 그대로 믿으면 안 된다 — 확정 사항과 어긋나 있다.** 시트는 `grade`·`sub_grade`를
    미라벨(NaN)로, `int_rate`·`installment`·`funded_amnt`·`funded_amnt_inv`·`issue_d`·
    `initial_list_status`를 **사후(0)** 로 두는데, #1이 이 8개를 사전으로 재분류했고 #17 ③이
    `grade`·`sub_grade`·`int_rate` 투입을 확정했다. **시트 값만 쓰면 LC 조건변수가 전부 빠져
    Lean 스펙(AUC 0.68대)이 된다.**
  - `loader.py`의 `PRE_APPROVAL_OVERRIDES`가 이를 코드에서 보정한다 — 피처 목록은 시트를 직접
    읽지 말고 **`loader.pre_approval_columns()` / `loader.select_feature_columns()`를 경유**한다.
    시트가 개정되면 `verify_dictionary_overrides()`가 알려주고, 그때 오버라이드를 비울 수 있다.
    **시트 개정은 열린 실행 항목**이다(`preprocessing_crosscheck_kgj.md` 10절).
  - 규칙 기반 재라벨링 결과와의 대조는 `label_pre_post_by_rule.py`로 재현한다(151개 중 사후 40 / 사전 111).
- `grade`, `sub_grade`, `int_rate`, `installment`, `funded_amnt`, `funded_amnt_inv`, `issue_d`, `initial_list_status`는 대출 발행(origination) 시점에 함께 확정되는 값으로, 신규 심사 대상 데이터에도 이미 채워져 있어 **사전(pre-approval) 변수로 간주**한다 (근거: 신규 데이터 컬럼이 훈련 데이터와 동일하다는 조교님 확인, `README.md`의 "독립변수(피처) 구성" 참고). **이 변수들은 부도확률 예측 모델의 입력 피처로 투입한다** (2026-07-29 회의 확정, `decision_log.md` #17 ③). "LC 자체 등급을 그대로 베끼는 셈"이라는 우려는 남지만, 투자자는 LC 심사를 통과한 대출에 투자할지만 결정하므로 결정 시점에 관측 가능한 값이다(#1). ⚠️ 기존 문서의 0.68대(조건변수를 뺀 Lean 스펙)와 섞어 인용하지 않는다. ⚠️ 문서에 남아 있는 **"0.71대"는 #16 만기필터 확정 이전 값**이다 — 확정 표본(723,563건) 재측정값은 **0.70대**이며, 같은 조건에서 필터만 빼면 0.7283이 나와 −2.1%p가 필터 효과임이 확인됐다(`outputs/reports/oof_diagnostics_kgj.md`). `decision_log.md` 정정은 팀 확인 후 한다.
  - ⚠️ **`int_rate`를 실현수익률로 그대로 쓰지 않는다.** 예전 규칙("정상상환 시 실현수익률 = `int_rate`")은 폐기됐다(#18) — 재투자 가정이 "매달 상환액을 잔존기간 매칭 국채에 재투자"로 확정되면서 정상상환분도 같은 관례로 재계산해야 한다. `int_rate`를 쓰는 것은 재투자율 12.6%를 가정하는 것과 같아 부도 쪽(국채)과 어긋난다. `int_rate`는 **피처로만** 쓴다.
### 누수(leakage) 방지 — 결과변수를 피처로 넣지 않는다

**실현수익률 계산에 쓰는 변수와 그 산출물은 모델 입력에서 전부 제외한다.** 이걸 넣으면 모형이
"이 대출이 얼마를 갚았는지"를 보고 부도를 맞히게 되어, 승인 시점에 존재하지 않는 정보로 예측하는
셈이 된다.

| 구분 | 변수 | 이유 |
| --- | --- | --- |
| **사후 상환 이력** | `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `out_prncp` 등 | 대출 실행 후에야 값이 생긴다 |
| **실현수익률 산출물** | `R`(계약만기 등가 연율화 수익률), `rf`, `XR`(초과수익률), `W(T)` | 결과변수 자체다 |

- 라벨 원본은 `data/processed/variable_dictionary_byGJ.xlsx`의 `is_pre_approval`이며, **결과 파일을
  모델 입력 데이터로 오인하지 않도록** 파일명·문서에 용도를 명시한다.
- `R`·`rf`·`XR`은 **threshold 탐색과 Sharpe 계산 단계에서 `loan_id`로 결합**해 쓴다. 피처 테이블에
  미리 붙여두지 않는다 — 붙여두면 실수로 학습에 들어간다.

### 결정으로 제외한 피처 — 누수와 구분한다

`loader.EXCLUDED_BY_DECISION`은 **누수도 식별자도 아닌데 실측 결과 모형을 깎아서** 뺀 변수다.
`NON_FEATURE_PRE_APPROVAL`(식별자·자유서술)과 목록을 합치지 않는다 — 저쪽은 "원리상 피처가
아니다", 이쪽은 "돌려보니 깎는다"이므로 사유가 섞이면 나중에 판단 근거를 잃는다.

- 현재 **`zip_code` 1건** (911범주 고카디널리티 → Train 칸별 부도율 암기). 같은 재분할에
  제외 전후를 짝지어 돌린 검증에서 **24 seed 전부, 6조합 모두 제외 쪽 우세**였고 국채·`q_score`
  기준 Δ Sharpe **+0.0122**(sd 0.0020)였다. 근거: `outputs/reports/model_comparison_kgj.md`.
- 본 파이프라인은 기본값(`apply_decisions=True`)으로 제외된 **101개 피처**를 받는다.
  **비교 실험만** `build_feature_table(apply_decisions=False)`로 102개(제외 전)를 받는다 —
  `model_comparison.py`의 `cv`·`stability` 블록이 그 경우다.

### 결과 파일 저장 규칙

- **표본 실행이 전수 결과를 덮어쓰지 않게 한다.** 출력 경로를 분리한다: 전수는 `*_full.csv`,
  표본은 `*_sample.csv`. 같은 파일명을 공유하지 않는다.
- 결과는 성격별로 나눠 저장한다.
  - **모델 결합용**: `id`, `term`, `issue_d`, `funded_amnt`, `R`, `rf`, `XR`
  - **계산 감사용**: 월별 현금흐름, Recovery 시점·금액, `W(T)`, 제외 사유
- 전수 결과 CSV(`*_full.csv`)는 대용량이라 **git에 커밋하지 않는다**(`.gitignore`). 요약 집계표만
  `outputs/` 아래에 커밋하고, 전체 결과는 각자 스크립트로 재생성한다.

- 원본 CSV(대출 원본 데이터, ~1.2GB)는 git 추적 대상에서 제외한다 (`.gitignore` 참고) — 새 원본 파일도 `data/raw/`에만 두고 커밋하지 않는다.
- 변수 사전 빌드/검증은 `src/preprocessing/`의 전용 스크립트를 통해서만 수행하고, 결과는 `data/processed/`에 저장한다.

## 결측치 처리 — NaN 그대로 유지 (**확정**)
`decision_log.md` #13 ③에서 실측으로 정리됐고, **2026-07-29 회의에서 전제 2건이 모두 확정되어
조건부가 해제됐다** — 모형 XGBoost 단일화(#17 ①), 표본 만기+버퍼 6개월(#16).

- **원칙**: T2 pre-2016(미수집)·T2 2016+(계좌 미보유)·T3(사건 없음) **전부 NaN 그대로 투입**한다.
  대체하지 않고 결측 더미도 만들지 않는다. GBM이 결측을 native로 처리하기 때문이다.
- **근거**: 처리 스킴 4종의 AUC 최대 차이가 0.00009로 seed 편차(0.00134)의 1/15이고 부호도 seed마다 갈린다.
  "계좌가 없으면 잔액은 0"이 겨눈 금액·개수형 변수는 2016년 이후 결측률이 0.01%라 채울 결측 자체가 없다.
- **여전히 유효한 금지**: **2015년 이전 구간의 0 대체 금지.** 그 구간은 "계좌가 없다"가 아니라 "안 물어봤다"이므로
  0을 넣으면 옛 차주 전체가 무활동 차주로 왜곡된다.
- **T2 변수 14개는 남긴다** (#12 ④). 없애야 할 것은 T2 *결측더미* 뿐이다.
- **표준화·로그변환·구간화·캡핑도 하지 않는다** — 트리는 값의 순서만 쓴다(#17). `dti` 999 같은 특수값도
  원본 그대로 둔다(캡핑하면 진짜 100인 행과 뭉개져 정보가 준다).
- ⚠️ 이 결론은 **XGBoost 단일화**에 전적으로 의존한다. 로지스틱이 되살아나면 NaN을 못 받으므로
  대체값·더미·표준화가 전부 되살아난다. 현재는 #17 ①로 확정된 상태다.

## 범주형 변수 — `category` dtype (**더미화하지 않는다**)

결측을 NaN으로 두는 것과 같은 이유다 — XGBoost 단일화(#17 ①)로 더미가 필요 없어졌다.

- 문자열 범주형은 pandas **`category` dtype**으로 두고 XGBoost에 `enable_categorical=True`로 넘긴다.
  `pd.get_dummies()`를 쓰지 않는다 — 더미화하면 변수 수가 폭증하고, NaN을 따로 다뤄야 한다.
- `emp_length`는 **순서 있는(ordered) category**로 만든다 (`< 1 year` … `10+ years`).
- 문자열로 저장된 수치·날짜는 되돌린다: `int_rate`·`revol_util`은 `%` 제거 후 실수,
  `issue_d` 등 월 컬럼은 **절대 월 서수**(`year*12 + month`)로 바꾼다.
- ⚠️ `README.md`의 "범주형 변수는 더미화하되…"는 **로지스틱 전제의 과거 검토**다(#17 ① 이전).
  이 절이 현행 규칙이다.

> 컬럼명 표준화 규칙, 파생(비율) 변수 목록, 처리 로그 경로는 아직 미확정 — 추후 보충.

## 분할 공유 — **매니페스트를 쓴다. seed만 믿지 않는다**

원본 CSV는 git에 없으므로(1.2GB) 팀원이 각자 받는다. 그런데 **각자 돌린 분할이 같다는 보장이
없다.** `train_test_split`은 행의 **위치(position)** 를 셔플하므로 원본의 **행 순서가 다르면
완전히 다른 분할**이 나오고, `filter_analysis_sample()`의 723,563건 검증은 **건수만 보므로
그대로 통과한다** — 어긋난 것을 아무도 모른다.

행 순서가 달라지는 경로는 실제로 있다: 원본을 엑셀·Numbers로 열었다 저장(그래서 금지한다),
다른 시점에 받은 파일, 정렬해서 저장한 사본. 그리고 팀원이 각자 다른 seed를 쓰면 **한 사람의
Test 건이 다른 사람의 Train에 들어가** 개인 기준으로는 규칙을 지켰는데 팀 전체로는 **Test가
오염된다.**

**해법: `id → split` 매니페스트를 git으로 공유한다.**

| 파일 | 내용 |
| --- | --- |
| `data/processed/split_manifest_6_2_2_seed20260730.csv.gz` | `id`·`split`·`target` (723,563행, **2.35MB gzip** — 커밋 가능). **현행** — seed는 `config.yaml`의 `random_seed.default`가 결정한다 |
| `data/processed/split_manifest_6_2_2_seed20260730.source.md` | 출처 카드 (seed·비율·건수·부도율·체크섬) |

- **생성**: `python src/preprocessing/export_split_manifest.py` (1회. 이미 만들어져 있다)
- **대조**: `python src/preprocessing/export_split_manifest.py --verify`
  → 자기 원본으로 만든 분할이 매니페스트와 같은지 **SHA-256 체크섬**으로 확인한다.
  현행 체크섬 `26bf46f9…e57c`(seed 20260730). 옛 `…seed42` 매니페스트(체크섬 `cffd9896…4d5c`)는
  보존만 하고 쓰지 않는다. 불일치하면 **매니페스트를 새로 만들지 말고 원인을 먼저 찾는다** —
  이미 그 분할로 낸 산출물이 전부 무효가 된다.
- **사용**: `split_6_2_2()`를 직접 쓰지 말고 **`split_from_manifest(X, y, meta)`** 를 쓴다.
  `id`로 매칭하므로 행 순서·pandas·sklearn 버전과 무관하다.
- ⚠️ `split_from_manifest()`는 각 split을 **`id` 오름차순으로 정렬해** 돌려준다. 분할 집합이
  같아도 **행 순서가 다르면 `StratifiedKFold`의 fold 배정이 달라지기** 때문이다(그것도 위치를
  셔플한다). 정렬하면 OOF fold까지 `(id 집합, seed)`만으로 결정된다.

### ⚠️ Test 격리 — 코드로 잠갔다

- **`split_from_manifest()`는 Test를 기본적으로 반환하지 않는다.** `unlock_test=True`를 명시해야
  나오고, 그때 경고를 출력한다. 실수로 `parts["test"]`를 집는 일을 막는 장치다.
- Test는 모형·threshold가 **전부 확정된 뒤 단 1회** 적용한다(`src/analysis/AGENTS.md`).
  랭킹 기준 3종 비교처럼 **여러 안 중 하나를 고르는 작업에 Test를 쓰면 규칙 위반**이다.
- **K=50 반복(#18)에는 `resplit_train_validation()`을 쓴다.** `split_6_2_2(seed=k)`를 반복에
  쓰면 **Test 구성까지 매번 바뀌어** "Test set은 그대로 고정해두고"(`README.md`)가 깨진다.
  이 함수는 매니페스트의 Train+Validation 풀만 75/25로 다시 가른다.

## 구현 — 어느 함수를 부르는가 (2026-07-30, 커밋 `7e1cb9d`)

위 규칙은 이미 코드로 구현돼 있다. **다시 짜지 말고 아래를 호출한다.**

| 모듈 | 담당 | 주요 함수 |
| --- | --- | --- |
| `loader.py` | 원본 로딩 · 표본 필터 · 피처 컬럼 선택 | `load_raw_loans()`, `filter_analysis_sample()`(**723,563건 검증 내장**), `normalize_loan_status()`, `parse_term_months()`, `make_target()`, `pre_approval_columns()`, `select_feature_columns()`, `validate_schema()` |
| `preprocessor.py` | dtype 정리 · 피처 테이블 · 분할 | `coerce_dtypes()`, `build_feature_table()`, **`split_from_manifest()`**(권장), `resplit_train_validation()`(K=50 반복용), `load_split_manifest()`, `split_6_2_2()`(매니페스트 **생성 전용**) |
| `export_split_manifest.py` | 분할 매니페스트 생성·체크섬 대조 | `build_manifest()`, `split_checksum()`, `main()` (`--verify`) |

> ⚠️ **`split_6_2_2()`를 분석 코드에서 직접 부르지 않는다.** 이 함수는 매니페스트를 **만들 때만**
> 쓴다. 분석은 `split_from_manifest()`를 경유해야 팀원 간 분할이 일치하고 Test가 잠긴다.

- `filter_analysis_sample(df, verify=True)`는 결과가 **723,563건이 아니면 예외로 중단**한다(#16).
  건수가 바뀌었다면 필터를 고치기 전에 **왜 바뀌었는지부터** 확인한다.
- `META_COLUMNS`(`id`·`issue_d`·`term`·`funded_amnt`·`int_rate`)는 피처가 아니라 **결합·현금흐름
  계산용**으로 따로 들고 간다. `int_rate`만 예외적으로 피처이기도 하다(#17 ③).
- 두 모듈은 **파일을 쓰지 않는다** — 결과는 in-memory로 `src/analysis/`에 넘긴다.
  중간 산출물을 저장하게 되면 위 「결과 파일 저장 규칙」의 `*_full.csv` / `*_sample.csv` 분리를 따른다.

## 실행
- 실행 환경은 저장소 루트 `requirements.txt` 참고. 시스템 python에는 scikit-learn·xgboost가 없다.
- 모든 스크립트는 **저장소 루트 기준 절대경로**(`Path(__file__).resolve().parents[2]`)로 파일을 찾는다.
  어느 디렉터리에서 실행해도 동작해야 하며, CWD에 의존하는 상대 파일명을 쓰지 않는다.
- 경로·분할비율·seed가 필요하면 `src/utils/config.py`의 `load_config()`를 쓴다 (`config/config.yaml`이 단일 출처).

## 참고
- **확정 사항의 원본은 `outputs/reports/decision_log.md`다.** 이 문서와 어긋나면 decision_log가 우선이고, 이 문서를 고친다.
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md` (구 `분석 에이전트` 폴더의 절대 규칙)
  - ⚠️ 과거 폴더 구조 기준이라 문서에 적힌 경로(`v_desc_unified.xlsx`, `build_v_desc_*.py` 등) 상당수가 **현존하지 않는다.** 방법론만 참고하고 파일 경로는 믿지 말 것.
