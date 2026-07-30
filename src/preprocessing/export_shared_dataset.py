"""팀 공유용 전처리 데이터셋 내보내기 — Slack에 올릴 parquet 4종을 만든다.

## 왜 필요한가

팀원 전원이 원본 1.2GB를 갖고 있지만, **각자 전처리를 돌리면 결과가 갈린다.**
표본 필터·dtype 파싱·분할이 조금씩 어긋나도 건수 검증(723,563건)은 통과하므로
조용히 다른 데이터로 작업하게 된다. 전처리까지 끝난 파일을 한 번 만들어 공유하면
그 경로가 막힌다.

## 왜 CSV가 아니라 parquet인가

`preprocessor.coerce_dtypes()`는 object를 전부 `category`로, `emp_length`를 **순서형**
category로, 월 날짜를 `Int64`(결측 보존)로 만든다. CSV로 내리면 이게 전부 사라진다.

특히 위험한 건 **파일을 Test/비Test로 쪼갤 때**다. 팀원이 CSV를 받아 다시
`astype("category")`하면 **자기 파일에 등장한 값만** 카테고리가 되므로, Test에만 있는
희귀 범주가 비Test 파일에 없으면 **같은 문자열이 두 파일에서 다른 정수 코드**를 받는다.
XGBoost(`enable_categorical=True`)는 코드로 학습하니 최종 Test 적용 때 모델이 다른 값을
보게 된다 — 에러 없이 조용히 틀린다.

그래서 이 스크립트는 **전체 표본에서 category dtype을 확정한 뒤 잘라서** 저장한다.
pandas는 슬라이싱해도 미사용 카테고리를 지우지 않으므로 두 파일의 코드가 일치한다.
parquet은 pandas 메타데이터로 category·ordered·`Int64`를 그대로 보존한다.

## 산출물 (`data/processed/shared/`)

| 파일 | 내용 |
| --- | --- |
| `trainval_features.parquet` | 비Test 80% — 피처 + `target` + meta + `split`(train/validation) |
| `test_features.parquet` | Test 20% — **최종 1회 평가 전까지 열지 않는다** |
| `trainval_outcome.parquet` | 비Test 사후변수 (실현수익률·조기상환 보정용) |
| `test_outcome.parquet` | Test 사후변수 |
| `columns.json` | 어느 열이 피처/meta/타깃인지 — 읽을 때 `X`를 복원하는 근거 |
| `README_공유데이터.md` | 사용법·체크섬·Test 취급 규칙 |

⚠️ **사후변수(`outcome`)를 피처 파일에 합치지 않는다** — 합치면 누수다
(`src/preprocessing/AGENTS.md` 「누수 방지」). `id`로 조인해서 threshold·Sharpe 단계에서만 쓴다.

실행: `PYTHONPATH=src python src/preprocessing/export_shared_dataset.py`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    from preprocessing.preprocessor import META_COLUMNS, build_feature_table, load_split_manifest
    from utils.config import load_config, repo_root
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.preprocessor import META_COLUMNS, build_feature_table, load_split_manifest
    from utils.config import load_config, repo_root


SHARED_DIRNAME = "shared"

# 사후(post-approval) 컬럼 — `realized_return.load_outcome_frame()`이 쓰는 것과 같은 목록이다.
OUTCOME_COLUMNS = (
    "id", "loan_status", "term", "issue_d", "funded_amnt", "installment", "int_rate",
    "total_pymnt", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "last_pymnt_d",
)


def shared_dir() -> Path:
    d = load_config().paths.data_processed / SHARED_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_shared_table(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """피처 + 타깃 + meta + `split` 한 장. **자르기 전** 상태다.

    category dtype을 여기서 확정하는 것이 핵심이다 — 자른 뒤에 각자 만들면 코드가 어긋난다.
    """
    X, y, meta = build_feature_table()

    shared = X.copy()
    # meta 중 X에 없는 열만 덧붙인다. 겹치는 열(`issue_d`·`term`·`int_rate`·`funded_amnt`)은
    # 같은 파싱을 거쳐 값이 동일하므로 중복 저장할 이유가 없다.
    meta_only = [c for c in META_COLUMNS if c in meta.columns and c not in shared.columns]
    for col in meta_only:
        shared[col] = meta[col]

    shared["target"] = y.astype("int8")

    manifest = load_split_manifest(seed)
    assigned = meta["id"].astype(str).map(manifest)
    if assigned.isna().any():
        raise RuntimeError(
            f"매니페스트에 없는 id가 {int(assigned.isna().sum()):,}건이다 — "
            "표본 필터가 매니페스트 생성 시점과 다르다(#16)."
        )
    shared["split"] = assigned.astype("category")

    layout = {
        "features": list(X.columns),
        "meta_only": meta_only,
        "target": "target",
        "split": "split",
        "outcome": list(OUTCOME_COLUMNS),
    }
    return shared, layout


def load_outcome_table() -> pd.DataFrame:
    """사후변수 테이블 — 원본에서 표본 필터만 적용해 그대로 가져온다.

    파싱하지 않고 원본 형태로 둔다. `realized_return.py`가 자기 방식으로 파싱하므로
    여기서 미리 손대면 두 곳에 파싱 규칙이 생긴다.
    """
    from preprocessing.loader import filter_analysis_sample, load_raw_loans

    raw = load_raw_loans(usecols=list(OUTCOME_COLUMNS))
    return filter_analysis_sample(raw)


def export(seed: int | None = None) -> dict[str, Path]:
    out = shared_dir()

    print("표본 구성 중... (원본 1.2GB 로딩 — 수 분 걸린다)")
    shared, layout = build_shared_table(seed)
    print(f"  피처 {len(layout['features'])}개 + meta {len(layout['meta_only'])}개, 총 {len(shared):,}건")

    print("사후변수 로딩 중...")
    outcome = load_outcome_table()
    outcome_by_id = outcome.set_index(outcome["id"].astype(str))

    is_test = shared["split"] == "test"
    written: dict[str, Path] = {}

    for name, mask in (("trainval", ~is_test), ("test", is_test)):
        part = shared.loc[mask]
        path = out / f"{name}_features.parquet"
        part.to_parquet(path, index=False, compression="zstd")
        written[f"{name}_features"] = path

        ids = part["id"].astype(str)
        opath = out / f"{name}_outcome.parquet"
        outcome_by_id.loc[ids].reset_index(drop=True).to_parquet(
            opath, index=False, compression="zstd"
        )
        written[f"{name}_outcome"] = opath

        print(f"  {name:9s} {len(part):>8,}건  부도율 {part['target'].mean():.4%}")

    cols_path = out / "columns.json"
    cols_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    written["columns"] = cols_path

    written["readme"] = write_readme(out, shared, layout, written, seed)
    return written


def write_readme(
    out: Path, shared: pd.DataFrame, layout: dict, written: dict[str, Path], seed: int | None
) -> Path:
    cfg = load_config()
    seed = cfg.random_seed.default if seed is None else seed
    counts = shared["split"].value_counts()

    def row(key: str) -> str:
        p = written[key]
        return (f"| `{p.name}` | {p.stat().st_size / 1024**2:.1f} MB | "
                f"`{sha256_of(p)[:16]}…` |")

    path = out / "README_공유데이터.md"
    path.write_text(
        f"""# 팀 공유 데이터셋 — 전처리 완료본

원본 1.2GB를 각자 전처리하면 결과가 갈리므로, **전처리까지 끝낸 파일**을 공유한다.
받은 뒤 `filter_analysis_sample()`을 다시 돌리지 않는다 — 이미 적용돼 있다.

- 표본: **{len(shared):,}건** (만기 + 버퍼 6개월, `decision_log.md` #16)
- 분할: 6:2:2, seed **{seed}** — train {counts.get('train', 0):,} /
  validation {counts.get('validation', 0):,} / test {counts.get('test', 0):,}
- 생성 스크립트: `src/preprocessing/export_shared_dataset.py`

## 파일

| 파일 | 크기 | SHA-256 (앞 16자) |
| --- | ---: | --- |
{row('trainval_features')}
{row('trainval_outcome')}
{row('test_features')}
{row('test_outcome')}
{row('columns')}

## ⚠️ Test 파일은 열지 않는다

`test_*.parquet` 2종은 **모형·threshold가 전부 확정된 뒤 단 1회** 적용한다
(`src/analysis/AGENTS.md`). 랭킹 기준을 고르거나 threshold를 조정하는 데 쓰면 규칙 위반이다 —
그 비교는 Validation에서 끝낸다.

작업은 `trainval_*.parquet`만으로 한다.

## 쓰는 법

```python
from preprocessing.export_shared_dataset import load_shared

X, y, meta, split = load_shared("trainval")

X_tr,  y_tr  = X[split == "train"],      y[split == "train"]
X_val, y_val = X[split == "validation"], y[split == "validation"]
```

`split` 컬럼은 **기준 1회 실행(base run)** 용이다. K=50 안정성 검증처럼 Train/Validation을
매번 다시 나누는 작업은 이 컬럼을 무시하고 비Test 80% 전체를 풀로 쓴다
(`resplit_train_validation()`과 같은 취급).

## 사후변수 (`*_outcome.parquet`)

`total_pymnt`·`recoveries`·`last_pymnt_d` 등 **결과변수**다. 실현수익률·조기상환 보정
(B팀 1순위, #20)에 쓴다.

⚠️ **피처에 합치지 않는다 — 누수다.** `id`로 조인해 threshold·Sharpe 단계에서만 쓴다.

## dtype 주의

범주형은 `category` dtype이고, **카테고리 목록은 전체 표본 기준으로 확정**돼 있다
(trainval/test 파일의 코드가 일치한다). CSV로 변환해 다시 읽으면 이 보장이 깨지므로
**parquet 그대로 쓴다.**
""",
        encoding="utf-8",
    )
    return path


def load_shared(part: str = "trainval") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """공유 parquet → `(X, y, meta, split)`.

    `columns.json`으로 피처/meta를 갈라 돌려주므로 **`X`에 `target`·`split`·`id`가 섞이지
    않는다.** 원본 1.2GB 없이 파이프라인에 바로 꽂을 수 있는 진입점이다.

    `part="test"`는 최종 1회 평가 전까지 쓰지 않는다.
    """
    out = shared_dir()
    layout = json.loads((out / "columns.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(out / f"{part}_features.parquet")

    if part == "test":
        print(f"⚠️  Test set을 열었다 ({len(df):,}건). 모형·threshold 확정 후 **1회만** 쓴다.")

    X = df[layout["features"]]
    y = df[layout["target"]]
    meta = df[[c for c in META_COLUMNS if c in df.columns]]
    return X, y, meta, df[layout["split"]]


def load_shared_outcome(part: str = "trainval") -> pd.DataFrame:
    """공유 사후변수 테이블. `id`로 `load_shared()` 결과와 조인해 쓴다."""
    return pd.read_parquet(shared_dir() / f"{part}_outcome.parquet")


def main() -> None:
    import sys

    seed = None
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])

    written = export(seed)

    print("\n산출물")
    total = 0.0
    for key, path in written.items():
        mb = path.stat().st_size / 1024**2
        total += mb
        print(f"  {path.relative_to(repo_root())}  ({mb:.1f} MB)")
    print(f"\n  합계 {total:.1f} MB")
    print("\n→ git 미추적이다(.gitignore). Slack 등 팀 채널로 공유한다.")
    print("  받는 사람은 README_공유데이터.md의 SHA-256으로 같은 파일인지 대조한다.")


if __name__ == "__main__":
    main()
