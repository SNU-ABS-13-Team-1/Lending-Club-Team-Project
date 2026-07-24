# 분석 규칙

## 입력/출력
- 입력: `data/processed/` 전처리 완료 데이터
- 출력: `outputs/reports/` (모델·통계 결과), 필요 시 중간 산출물

## 규칙
- 승인/거절 threshold는 **오직 Sharpe Ratio 극대화**로 결정한다. accuracy, AUC 등 일반 분류 지표로 결정하지 않는다.
- Sharpe Ratio 정의: `(포트폴리오 수익률 평균 − 무위험수익률) / 포트폴리오 수익률의 표본표준편차(ddof=1)`.
  - 승인한 대출의 실현수익률: 부도 시 0, 정상상환 시 `int_rate`.
  - 거절한 대출의 실현수익률: 무위험수익률로 대체.
- 데이터 분할: Train 60% / Validation 20% / Test 20% (`train_test_split`을 두 번 적용 — 전체→80/20으로 Test 분리 후, 남은 80%를 75/25로 Train/Validation 분리).
- Train으로 부도확률 예측 모형을 학습하고, Validation으로 threshold를 확정한다.
- **Test set으로 모형을 재조정하지 않는다.** Train으로 학습한 모형과 Validation으로 확정한 threshold를 그대로 적용해 검증만 한다.
- Threshold 확정 후에는 Train 전체(60%)로 모형을 재학습한 뒤 Test에 적용한다.
- 사전/사후(pre/post-approval) 변수 구분은 아직 미확정이므로, 현재는 이 구분을 피처 선택에 적용하지 않는다. 팀에서 확정 후 반영.
- 무위험수익률(`rf`)은 참고 노트북에서 5% 고정값으로 가정했으나, 실제 프로젝트에서 고정값을 쓸지 대출 발행 시점 기준 실제 국채수익률을 쓸지는 팀 확정 필요.
- 안정성 검증 시 `random_state` 0~29로 30회 반복해 모형 Sharpe가 "전부 승인" 베이스라인을 이기는 비율을 확인한다.

## 참고
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md`
- 파이프라인 구조적 템플릿: `notebooks/LendingClub_실습_v2.ipynb`
