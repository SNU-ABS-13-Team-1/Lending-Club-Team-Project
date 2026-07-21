# 전처리 규칙

## 입력/출력
- 입력: `data/raw/lending_club_2020_train.csv` (원본, 절대 수정 금지), `data/raw/LCDataDictionary.xlsx` (LC 공식 데이터 사전)
- 출력: `data/processed/*.xlsx` (변수 사전/라벨 등 전처리 산출물)

## 규칙
- 거절된 대출을 표본에서 제외하지 않는다. 거절 건은 무위험자산(국채) 투자로 간주해 실현수익률을 무위험수익률로 대체한다 — 승인 건만 남기는 필터링 금지.
- `data/processed/v_desc_unified.xlsx`의 사전/사후(pre/post-approval) 변수 라벨은 아직 팀에서 확정하지 않은 초안이다. 현재 상태 그대로 신뢰해 피처를 자동 필터링하지 않는다.
- `int_rate`는 실현수익률 계산용으로는 보존하되, 부도확률 예측 모델의 입력 피처 목록에서는 제외한다 (사후 확정 변수).
- 원본 CSV(`lending_club_2020_train.csv`, ~1.2GB)는 git 추적 대상에서 제외한다 (`.gitignore` 참고) — 새 원본 파일도 `data/raw/`에만 두고 커밋하지 않는다.
- 변수 사전 빌드/검증은 `build_v_desc_check.py`, `build_v_desc_unified.py`를 통해서만 수행하고, 결과는 `data/processed/`에 저장한다.

> 결측치 처리 방식, 컬럼명 표준화 규칙, 처리 로그 경로 등은 아직 팀에서 확정하지 않음 — 추후 보충.

## 참고
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md` (구 `분석 에이전트` 폴더의 절대 규칙)
