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
- `data/processed/`의 변수 사전/라벨 산출물에 담긴 사전/사후(pre/post-approval) 변수 라벨은 아직 팀에서 확정하지 않은 초안이다. 현재 상태 그대로 신뢰해 피처를 자동 필터링하지 않는다.
- `grade`, `sub_grade`, `int_rate`, `installment`, `funded_amnt`, `funded_amnt_inv`, `issue_d`, `initial_list_status`는 대출 발행(origination) 시점에 함께 확정되는 값으로, 신규 심사 대상 데이터에도 이미 채워져 있어 **사전(pre-approval) 변수로 간주**한다 (근거: 신규 데이터 컬럼이 훈련 데이터와 동일하다는 조교님 확인, `README.md`의 "독립변수(피처) 구성" 참고). `int_rate`는 실현수익률 계산에도 그대로 쓴다. 단, 이 변수들을 실제로 부도확률 예측 모델의 입력 피처로 포함할지(예: LC 자체 등급을 그대로 베끼는 셈이 되지 않는지)는 별도 모델링 판단이 필요하며 아직 팀에서 확정하지 않았다.
- 원본 CSV(대출 원본 데이터, ~1.2GB)는 git 추적 대상에서 제외한다 (`.gitignore` 참고) — 새 원본 파일도 `data/raw/`에만 두고 커밋하지 않는다.
- 변수 사전 빌드/검증은 `src/preprocessing/`의 전용 스크립트를 통해서만 수행하고, 결과는 `data/processed/`에 저장한다.

> 결측치 처리 방식, 컬럼명 표준화 규칙, 처리 로그 경로 등은 아직 팀에서 확정하지 않음 — 추후 보충.

## 참고
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md` (구 `분석 에이전트` 폴더의 절대 규칙)
