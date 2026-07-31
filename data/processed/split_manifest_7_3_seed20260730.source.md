# split_manifest_7_3_seed20260730.csv.gz — 출처 카드

## 무엇인가
Lending Club 분석 표본 **723,563건**(`decision_log.md` #16)의
**Train/Validation 7:3 분할 정의**다. `id`와 `split`, 그리고 대조용 `target`을 담는다.

## 왜 있는가
seed만 공유하면 분할이 재현되지 않을 수 있다 — `train_test_split`은 **행의 위치**를 셔플하므로
원본 CSV의 행 순서가 다르면 다른 분할이 나오고, 723,563건 검증은 건수만 보므로 **조용히
어긋난다.** 이 파일은 `id` 기준이라 행 순서·라이브러리 버전과 무관하다.

## 어떻게 만들었나
- 생성 스크립트: `src/preprocessing/export_split_manifest.py --scheme 7_3`
- seed: **20260730** (`config/config.yaml`의 `random_seed.default`)
- 비율: train 0.7 / validation 0.3 — **Test 칸이 없다**
- 방법: `train_test_split` 한 번. `stratify`를 건다(부도율 16.2%).
- **최종 Test는 별도 파일**이다 — `data/raw/lending_club_2020_test_2nd.csv`
  (481,833건, train 원본과 `id` 교집합 0건). `decision_log.md` #30.
- 입력: `data/raw/lending_club_2020_train.csv` → `filter_analysis_sample()` (만기 + 버퍼 6개월)

## 검증
| split | 건수 | 비율(%) | 부도율(%) |
| --- | ---: | ---: | ---: |
| train | 506,494 | 70.0 | 16.213 |
| validation | 217,069 | 30.0 | 16.2133 |

- **분할 체크섬(SHA-256)**: `532b9cd86c7dc1bb031ce82ba7e0347e4fbf5e3e850b299beb8ba788b9707e7d`
  → 팀원끼리 같은 분할을 쓰는지 이 값으로 대조한다.
    `python src/preprocessing/export_split_manifest.py --scheme 7_3 --verify`
- 파일 SHA-256: `889909c6e5927e4e4145682cb5904c5b844f72b8123ace68178269b689d4f1cd`
- 부도율이 세 split에서 소수점 둘째 자리까지 맞는지 확인한다(`stratify` 정상 동작 근거).

## 쓰는 법
```python
from preprocessing.preprocessor import build_feature_table, split_from_manifest

X, y, meta = build_feature_table()
parts = split_from_manifest(X, y, meta, scheme="7_3")
X_tr, y_tr, meta_tr = parts["train"]
```
`split_6_2_2()`/`split_7_3()`(seed 기반)를 직접 쓰지 말고 이 함수를 쓴다.

## ⚠️ Test 취급
- 이 매니페스트에는 **Test 칸이 없다.** `unlock_test=True`를 주면 예외가 난다.
- 최종 평가는 `loader.second_test_path()`의 2nd Test로 **단 1회** 한다.
- K=50 반복은 `resplit_train_validation(..., scheme="7_3")`을 쓴다 — 표본 전량을
  70/30으로 다시 가르며, Test는 애초에 이 표본 밖이라 흔들리지 않는다.
