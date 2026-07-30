"""6:2:2 분할 매니페스트 생성 — 팀원 전원이 **동일한 분할**을 쓰게 만든다.

## 왜 필요한가 — seed만으로는 부족하다

`split_6_2_2()`는 `config.yaml`의 seed(42)와 비율로 `train_test_split`을 두 번 돌린다.
같은 입력이면 같은 결과가 나오지만, **"같은 입력"의 조건이 생각보다 까다롭다.**

`train_test_split`은 행의 **위치(position)** 를 셔플한다. 따라서 원본 CSV의 **행 순서가 다르면
완전히 다른 분할**이 나온다. 그런데 `filter_analysis_sample()`의 723,563건 검증은 **건수만
보므로 그대로 통과한다** — 어긋난 것을 아무도 모른 채 진행된다.

행 순서가 달라지는 경로는 실제로 있다.

- 원본을 엑셀·Numbers로 열었다 저장 (`AGENTS.md`가 금지하는 이유 중 하나)
- 팀 공유 채널에서 다른 시점에 받은 파일
- 정렬해서 저장한 사본

그리고 팀원이 각자 다른 seed로 돌리면 **한 사람의 Test 건이 다른 사람의 Train에 들어간다.**
개인 기준으로는 규칙을 지켰는데 **팀 전체로 보면 Test가 오염된다.**

## 해법 — `id` 기준 매니페스트

`id`(대출 고유번호)는 표본 723,563건에서 **전부 고유하고 결측이 없다**(실측).
그래서 `id → split` 표를 한 번 만들어 git에 올리면

- **행 순서와 무관하다** — 위치가 아니라 `id`로 매칭한다
- **pandas·sklearn 버전과 무관하다** — 난수를 다시 뽑지 않는다
- **원본 1.2GB 없이도 공유된다** — 매니페스트만 있으면 "어느 건이 Test인지" 알 수 있다
- **감사 가능하다** — 언제 어떤 seed로 만든 분할인지 기록이 남는다

⚠️ 원본은 여전히 각자 받아야 한다. 매니페스트는 **분할 정의**만 공유하는 것이다.

실행: `/opt/anaconda3/bin/python src/preprocessing/export_split_manifest.py`
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

try:
    from preprocessing.preprocessor import build_feature_table, split_6_2_2
    from utils.config import load_config, repo_root
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.preprocessor import build_feature_table, split_6_2_2
    from utils.config import load_config, repo_root


MANIFEST_STEM = "split_manifest_6_2_2"


def manifest_path(seed: int) -> Path:
    """`data/processed/split_manifest_6_2_2_seed42.csv.gz`.

    seed를 파일명에 남긴다 — 다른 seed로 만든 분할이 같은 파일을 덮어쓰면 어느 것이
    쓰였는지 추적할 수 없다.
    """
    return load_config().paths.data_processed / f"{MANIFEST_STEM}_seed{seed}.csv.gz"


def build_manifest(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """`(매니페스트, 요약)`. 매니페스트는 `id`·`split` 두 열이다.

    `id`를 문자열로 둔다 — 정수로 캐스팅하면 선행 0이 사라지거나 dtype이 환경마다 달라진다.
    """
    cfg = load_config()
    seed = cfg.random_seed.default if seed is None else seed

    X, y, meta = build_feature_table()
    parts = split_6_2_2(X, y, meta, seed=seed)

    rows = []
    for name in ("train", "validation", "test"):
        _, y_part, meta_part = parts[name]
        rows.append(
            pd.DataFrame({"id": meta_part["id"].astype(str), "split": name, "target": y_part})
        )
    manifest = pd.concat(rows).sort_values("id", kind="mergesort").reset_index(drop=True)

    if manifest["id"].duplicated().any():
        raise RuntimeError("id가 중복됐다 — 매니페스트를 쓸 수 없다.")
    if len(manifest) != len(X):
        raise RuntimeError(f"매니페스트 {len(manifest):,}건 ≠ 표본 {len(X):,}건")

    summary = {
        "seed": seed,
        "n_total": len(manifest),
        "checksum": split_checksum(manifest),
        "shares": (manifest["split"].value_counts(normalize=True) * 100).round(3).to_dict(),
        "counts": manifest["split"].value_counts().to_dict(),
        "default_rate": (manifest.groupby("split")["target"].mean() * 100).round(4).to_dict(),
    }
    return manifest, summary


def split_checksum(manifest: pd.DataFrame) -> str:
    """분할 정의의 SHA-256. **행 순서에 무관하도록** `id`로 정렬한 뒤 계산한다.

    팀원끼리 "같은 분할을 쓰고 있는가"를 이 한 줄로 대조할 수 있다.
    """
    ordered = manifest.sort_values("id", kind="mergesort")
    payload = "\n".join(f"{i},{s}" for i, s in zip(ordered["id"], ordered["split"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_source_card(path: Path, summary: dict) -> Path:
    """짝이 되는 `.source.md` 출처 카드 (`docs/macro_indicators_spec.md` 2.8 규칙)."""
    card = path.with_suffix("").with_suffix(".source.md")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    counts, shares, dr = summary["counts"], summary["shares"], summary["default_rate"]

    card.write_text(
        f"""# {path.name} — 출처 카드

## 무엇인가
Lending Club 분석 표본 **{summary['n_total']:,}건**(`decision_log.md` #16)의
**Train/Validation/Test 6:2:2 분할 정의**다. `id`와 `split`, 그리고 대조용 `target`을 담는다.

## 왜 있는가
seed만 공유하면 분할이 재현되지 않을 수 있다 — `train_test_split`은 **행의 위치**를 셔플하므로
원본 CSV의 행 순서가 다르면 다른 분할이 나오고, 723,563건 검증은 건수만 보므로 **조용히
어긋난다.** 이 파일은 `id` 기준이라 행 순서·라이브러리 버전과 무관하다.

## 어떻게 만들었나
- 생성 스크립트: `src/preprocessing/export_split_manifest.py`
- seed: **{summary['seed']}** (`config/config.yaml`의 `random_seed.default`)
- 비율: `config/config.yaml`의 `split` (train 0.6 / validation 0.2 / test 0.2)
- 방법: `train_test_split`을 두 번 — 전체를 80/20으로 갈라 Test를 떼고, 남은 80%를 75/25로
  Train/Validation으로 나눈다. 두 단계 모두 `stratify`를 건다(부도율 16.2%).
- 입력: `data/raw/lending_club_2020_train.csv` → `filter_analysis_sample()` (만기 + 버퍼 6개월)

## 검증
| split | 건수 | 비율(%) | 부도율(%) |
| --- | ---: | ---: | ---: |
| train | {counts.get('train', 0):,} | {shares.get('train', 0)} | {dr.get('train', 0)} |
| validation | {counts.get('validation', 0):,} | {shares.get('validation', 0)} | {dr.get('validation', 0)} |
| test | {counts.get('test', 0):,} | {shares.get('test', 0)} | {dr.get('test', 0)} |

- **분할 체크섬(SHA-256)**: `{summary['checksum']}`
  → 팀원끼리 같은 분할을 쓰는지 이 값으로 대조한다.
    `python src/preprocessing/export_split_manifest.py --verify`
- 파일 SHA-256: `{sha}`
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
""",
        encoding="utf-8",
    )
    return card


def parse_seed_arg(argv: list[str]) -> int | None:
    """`--seed 20260730` 형태를 읽는다. 없으면 `None`(= config의 기본 seed).

    seed를 CLI로 받는 이유는 **새 분할을 뽑을 때 기존 파일을 덮지 않기 위해서**다 —
    파일명에 seed가 들어가므로 다른 seed는 다른 파일이 되고, 어느 분할로 낸 산출물인지
    추적된다.
    """
    if "--seed" not in argv:
        return None
    i = argv.index("--seed")
    if i + 1 >= len(argv):
        raise SystemExit("--seed 뒤에 값을 적어라. 예: --seed 20260730")
    return int(argv[i + 1])


def main() -> None:
    import sys

    verify_only = "--verify" in sys.argv
    seed = parse_seed_arg(sys.argv)

    print("표본 구성 중... (원본 1.2GB 로딩 — 수 분 걸린다)")
    manifest, summary = build_manifest(seed=seed)

    print(f"\n[분할 요약] seed={summary['seed']}  총 {summary['n_total']:,}건")
    for name in ("train", "validation", "test"):
        print(f"  {name:11s} {summary['counts'].get(name, 0):>8,}건  "
              f"{summary['shares'].get(name, 0):>7}%  부도율 {summary['default_rate'].get(name, 0)}%")
    print(f"\n  분할 체크섬 {summary['checksum']}")

    path = manifest_path(summary["seed"])
    if verify_only:
        if not path.exists():
            raise SystemExit(f"매니페스트가 없다: {path}\n먼저 --verify 없이 실행해 생성하라.")
        existing = pd.read_csv(path, dtype={"id": str})
        old = split_checksum(existing)
        same = old == summary["checksum"]
        print(f"  기존 파일 체크섬 {old}")
        print(f"\n  판정: {'✅ 일치 — 같은 분할을 쓰고 있다.' if same else '❌ 불일치!'}")
        if not same:
            raise SystemExit(
                "분할이 어긋났다. 원본 CSV의 행 순서나 표본 필터가 달라졌을 수 있다.\n"
                "매니페스트를 새로 만들지 말고 **먼저 원인을 찾아라** — 이미 이 분할로 낸\n"
                "산출물이 전부 무효가 된다."
            )
        return

    manifest.to_csv(path, index=False, compression="gzip")
    card = write_source_card(path, summary)
    size_mb = path.stat().st_size / 1024**2
    print(f"\n산출물")
    print(f"  {path.relative_to(repo_root())}  ({size_mb:.2f} MB, gzip)")
    print(f"  {card.relative_to(repo_root())}")
    print("\n→ 이 두 파일을 git에 커밋하면 팀원 전원이 동일한 분할을 쓴다.")
    print("  팀원은 원본을 각자 받은 뒤 `--verify`로 자기 분할이 같은지 확인한다.")


if __name__ == "__main__":
    main()
