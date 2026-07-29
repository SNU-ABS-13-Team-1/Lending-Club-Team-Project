"""`config/config.yaml` 로더 — 경로·분할비율·seed의 단일 출처.

왜 필요한가:
    분할비율(6:2:2)·seed·경로를 스크립트마다 직접 써 넣으면 값이 두 곳에 존재하게 되고,
    한쪽만 고쳤을 때 조용히 어긋난다. 파이프라인 코드는 이 모듈을 통해서만 읽는다.

쓰는 법:
    from utils.config import load_config, repo_root      # 스크립트 직접 실행 시
    cfg = load_config()
    cfg.split.train, cfg.split.validation, cfg.split.test   # 0.6 / 0.2 / 0.2
    cfg.random_seed.default                                  # 42
    cfg.paths.data_processed / "파일명.csv"                   # 절대경로로 해석됨

    rf = cfg.require("risk_free_rate.value")   # 미확정(null) 항목은 명시적으로 실패시킨다

적용 범위:
    **모형 학습·threshold 결정 등 본 파이프라인 코드**가 대상이다.
    src/analysis/ 의 탐색·검증용 비교 스크립트는 6:2:2가 아닌 자체 분할(2분할)을 쓰는데,
    이는 처리 방식 간 AUC 상대 비교가 목적이라 의도된 차이다 — 그 스크립트들까지
    이 설정에 맞출 필요는 없다 (config/config.yaml 주석 참고).

확인:
    python src/utils/config.py     # 해석된 설정을 출력
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def repo_root() -> Path:
    """저장소 루트 절대경로. 스크립트가 parents[2]를 각자 계산하지 않도록 여기서 제공한다."""
    return REPO_ROOT


@dataclass(frozen=True)
class Paths:
    data_raw: Path
    data_processed: Path
    outputs_reports: Path
    outputs_figures: Path


@dataclass(frozen=True)
class Split:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class RandomSeed:
    default: int
    stability_check_range: tuple[int, int]


@dataclass(frozen=True)
class RiskFreeRate:
    value: float | None


@dataclass(frozen=True)
class Config:
    paths: Paths
    split: Split
    random_seed: RandomSeed
    risk_free_rate: RiskFreeRate
    _raw: dict[str, Any]

    def require(self, dotted_key: str) -> Any:
        """미확정(null) 설정을 조용히 기본값으로 대체하지 않고 명시적으로 실패시킨다.

        예: cfg.require("risk_free_rate.value") — 팀이 Rf를 확정하기 전에는
        RuntimeError를 내서, 임의의 5% 같은 값이 결과에 섞여 들어가는 것을 막는다.
        """
        node: Any = self._raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"config.yaml에 '{dotted_key}' 항목이 없습니다.")
            node = node[part]
        if node is None:
            raise RuntimeError(
                f"config.yaml의 '{dotted_key}'가 아직 미확정(null)입니다. "
                f"팀에서 확정한 뒤 값을 채워야 합니다 — 임의의 기본값으로 대체하지 않는다."
            )
        return node


def load_config(config_path: Path | None = None) -> Config:
    """config/config.yaml을 읽어 검증된 Config로 반환한다.

    paths는 저장소 루트 기준 상대경로로 적혀 있으므로 절대경로로 해석해서 돌려준다.
    """
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    p = raw["paths"]
    paths = Paths(
        data_raw=REPO_ROOT / p["data_raw"],
        data_processed=REPO_ROOT / p["data_processed"],
        outputs_reports=REPO_ROOT / p["outputs_reports"],
        outputs_figures=REPO_ROOT / p["outputs_figures"],
    )

    s = raw["split"]
    split = Split(train=s["train"], validation=s["validation"], test=s["test"])
    total = split.train + split.validation + split.test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split 비율의 합이 1이 아닙니다: {total} ({s})")

    r = raw["random_seed"]
    lo, hi = r["stability_check_range"]
    random_seed = RandomSeed(default=r["default"], stability_check_range=(lo, hi))

    return Config(
        paths=paths,
        split=split,
        random_seed=random_seed,
        risk_free_rate=RiskFreeRate(value=raw["risk_free_rate"]["value"]),
        _raw=raw,
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"저장소 루트: {repo_root()}")
    print("\n[경로] (모두 존재해야 정상)")
    for name, path in vars(cfg.paths).items():
        mark = "O" if path.exists() else "X 없음"
        print(f"  {name:18s} {path.relative_to(REPO_ROOT)}  [{mark}]")
    print("\n[분할]")
    print(f"  train/validation/test = {cfg.split.train}/{cfg.split.validation}/{cfg.split.test}")
    print("\n[seed]")
    print(f"  default={cfg.random_seed.default}  안정성검증범위={cfg.random_seed.stability_check_range}")
    print("\n[무위험수익률]")
    try:
        print(f"  value={cfg.require('risk_free_rate.value')}")
    except RuntimeError as e:
        print(f"  (미확정) {e}")
