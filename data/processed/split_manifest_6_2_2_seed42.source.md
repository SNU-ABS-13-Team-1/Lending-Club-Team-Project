# split_manifest_6_2_2_seed42.csv.gz — 출처 카드

## 무엇인가
Lending Club 분석 표본 **723,563건**(`decision_log.md` #16)의
**Train/Validation/Test 6:2:2 분할 정의**다. `id`와 `split`, 그리고 대조용 `target`을 담는다.

## 왜 있는가
seed만 공유하면 분할이 재현되지 않을 수 있다 — `train_test_split`은 **행의 위치**를 셔플하므로
원본 CSV의 행 순서가 다르면 다른 분할이 나오고, 723,563건 검증은 건수만 보므로 **조용히
어긋난다.** 이 파일은 `id` 기준이라 행 순서·라이브러리 버전과 무관하다.

## 어떻게 만들었나
- 생성 스크립트: `src/preprocessing/export_split_manifest.py`
- seed: **42** (`config/config.yaml`의 `random_seed.default`)
- 비율: `config/config.yaml`의 `split` (train 0.6 / validation 0.2 / test 0.2)
- 방법: `train_test_split`을 두 번 — 전체를 80/20으로 갈라 Test를 떼고, 남은 80%를 75/25로
  Train/Validation으로 나눈다. 두 단계 모두 `stratify`를 건다(부도율 16.2%).
- 입력: `data/raw/lending_club_2020_train.csv` → `filter_analysis_sample()` (만기 + 버퍼 6개월)

## 검증
| split | 건수 | 비율(%) | 부도율(%) |
| --- | ---: | ---: | ---: |
| train | 434,137 | 60.0 | 16.2131 |
| validation | 144,713 | 20.0 | 16.2135 |
| test | 144,713 | 20.0 | 16.2128 |

- **분할 체크섬(SHA-256)**: `cffd98962bee6428b0e97e5febb5001b1d787d58eebf2f50bcec4fb0d8b54d5c`
  → 팀원끼리 같은 분할을 쓰는지 이 값으로 대조한다.
    `python src/preprocessing/export_split_manifest.py --verify`
- 파일 SHA-256: `efc2a502ee21d31b6583e1dfb1518c9d8c84075af11c85990e7c501ca7defeb6`
- 부도율이 세 split에서 소수점 둘째 자리까지 맞는지 확인한다(`stratify` 정상 동작 근거).

## 쓰는 법
```python
from preprocessing.preprocessor import build_feature_table, split_from_manifest

X, y, meta = build_feature_table()
parts = split_from_manifest(X, y, meta)          # train·validation만 돌려준다
X_tr, y_tr, meta_tr = parts["train"]
```
`split_6_2_2()`(seed 기반)를 직접 쓰지 말고 이 함수를 쓴다.

## ⚠️ Test 취급
- **Test는 기본적으로 반환되지 않는다.** `split_from_manifest(..., unlock_test=True)`로
  명시해야 나오고, 그때 경고를 출력한다.
- Test는 **모형·threshold가 전부 확정된 뒤 단 1회** 적용한다
  (`src/analysis/AGENTS.md` "Test set으로 모형을 재조정하지 않는다").
- 랭킹 기준 3종 비교처럼 **여러 안을 고르는 작업에 Test를 쓰면 규칙 위반**이다 —
  그 비교는 Validation에서 끝낸다.
