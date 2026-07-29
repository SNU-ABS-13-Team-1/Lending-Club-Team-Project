"""T2 블록(14개) 기여도 재측정 — seasoning 표본 필터 변경에 따른 재판정.

preprocessing_validation_kgj.md 5절의 *"T2 변수 자체를 제거하면 AUC −0.00813"* 은
**현행 표본(Current만 제외, T2 관측률 54.4%)** 에서 잰 값이다.
팀이 seasoning 편향 때문에 만기도래 대출만 쓰기로 하면서 T2 관측률이 35.8%로 떨어졌고,
특히 60개월 만기 구간에서는 **0.0%** 가 된다(60m 만기도래 = 2015년 이전 발행 vs
T2 수집 개시 2015-12 — 구조적 배타). 따라서 옛 수치를 그대로 인용할 수 없다.

이 스크립트는 표본 정의를 바꿔가며 **T2 14개 변수를 넣은 모형과 뺀 모형의 AUC 차이**만
측정한다. 판정 기준선은 seed 간 자체 편차이며, 그 이하면 "기여 없음"으로 본다.

표본 필터링은 src/preprocessing/AGENTS.md 규칙(완결건만) 위에 만기 조건을 얹는다:
    만기 = issue_d + term,  버퍼 6개월  ->  만기 <= 2020-04
(스냅샷 2020-10에서 상각 확정까지 4~5개월 걸리므로 버퍼를 둔다)

AUC는 처리 방식 간 상대 비교용이다 — 승인/거절 threshold 결정과 무관하다
(src/analysis/AGENTS.md).

실행:
    python src/analysis/t2_contribution_reassessment.py
    python src/analysis/t2_contribution_reassessment.py --seeds 3   # 빠르게
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_validation import (  # noqa: E402  (같은 폴더의 검증 스크립트와 정의 공유)
    BASE_FEATURES, EMP_LENGTH_MAP, POLICY_PREFIX, RAW_FILE, T2, T3, header,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "outputs" / "t2_contribution_by_sample.csv"

# 만기 + 버퍼 6개월. 스냅샷(last_credit_pull_d 최종월) = 2020-10.
MATURITY_CUTOFF = pd.Period("2020-04", freq="M")


def load_loans() -> pd.DataFrame:
    """원본을 읽어 완결표본을 만들고 만기(issue_d + term)를 계산한다."""
    cols = sorted(set(T2 + T3 + BASE_FEATURES + ["loan_status", "issue_d", "term"]))
    raw = pd.read_csv(RAW_FILE, usecols=cols, low_memory=False)

    status = raw["loan_status"].astype(str).str.replace(POLICY_PREFIX, "", regex=False)
    keep = status.isin(["Fully Paid", "Charged Off"])

    df = raw[keep].copy()
    df["y"] = (status[keep] == "Charged Off").astype(int)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["year"] = df["issue_dt"].dt.year
    df["term_m"] = df["term"].astype(str).str.extract(r"(\d+)").astype(int)
    df["maturity"] = df["issue_dt"].dt.to_period("M") + df["term_m"]

    df["term"] = df["term_m"].astype(float)
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    return df


def slices(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """비교할 표본 정의와 각각에서 볼 스킴.

    주 비교축은 네 표본 전부에서 동일하게 S2(결측더미 없음 — XGBoost 권고안)로 맞춘다.
    S1(시트안)은 '만기+버퍼6m 전체'에서만 추가로 돌려, 결론이 더미 스킴 선택에
    의존하지 않는다는 것만 확인한다.
    """
    matured = df[df["maturity"] <= MATURITY_CUTOFF]
    return {
        "현행(Current만 제외)": (df, ["S2_더미없음"]),
        "만기+버퍼6m 전체": (matured, ["S1_시트안", "S2_더미없음"]),
        "만기+버퍼6m 36m만": (matured[matured["term_m"] == 36], ["S2_더미없음"]),
        "만기+버퍼6m 60m만": (matured[matured["term_m"] == 60], ["S2_더미없음"]),
    }


def design(df: pd.DataFrame, with_t2: bool, dummies: bool) -> pd.DataFrame:
    """설계행렬. with_t2=False면 T2 14개 변수를 통째로 뺀다."""
    X = df[BASE_FEATURES].copy()
    value_cols = (T2 if with_t2 else []) + T3
    for c in value_cols:
        X[c] = df[c]
    if dummies:
        for c in value_cols:
            X["D_" + c] = df[c].isna().astype(int)
    return X


def measure(sample: pd.DataFrame, name: str, n_sample: int, n_seeds: int,
            schemes: list[str]) -> list[dict]:
    """한 표본 정의 안에서 T2 포함/제외 AUC를 재고, 차이를 seed 편차와 대조한다."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    d = sample.sample(n=min(n_sample, len(sample)), random_state=0)
    t2_obs = d["all_util"].notna().mean() * 100
    print(f"\n----- {name} -----")
    print(f"  전체 n={len(sample):,}  부도율 {sample['y'].mean() * 100:.2f}%  "
          f"| 분석 표본 n={len(d):,}  T2 관측률 {t2_obs:.1f}%")

    rows = []
    for scheme in schemes:
        dummies = scheme == "S1_시트안"
        aucs: dict[bool, list[float]] = {True: [], False: []}
        for seed in range(n_seeds):
            for with_t2 in (True, False):
                X = design(d, with_t2=with_t2, dummies=dummies)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, d["y"].values, test_size=0.25, random_state=seed,
                    stratify=d["y"].values)
                model = XGBClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                    n_jobs=8, random_state=seed)
                model.fit(X_tr, y_tr)
                aucs[with_t2].append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

        inc, exc = np.array(aucs[True]), np.array(aucs[False])
        delta, seed_sd = (exc - inc).mean(), np.std(inc)
        rows.append({
            "표본": name, "스킴": scheme, "n": len(d), "T2관측률%": round(t2_obs, 1),
            "AUC_T2포함": round(inc.mean(), 5), "AUC_T2제외": round(exc.mean(), 5),
            "delta": round(delta, 5), "seed편차": round(seed_sd, 5),
            "판정": "기여 있음" if abs(delta) > seed_sd else "잡음 이하",
        })
        print(f"  {scheme:10s} T2포함 {inc.mean():.5f} / T2제외 {exc.mean():.5f} "
              f"| Δ {delta:+.5f} (seed 편차 {seed_sd:.5f}) "
              f"→ {'기여 있음' if abs(delta) > seed_sd else '잡음 이하'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-sample", type=int, default=300_000)
    args = parser.parse_args()

    if not RAW_FILE.exists():
        raise SystemExit(f"원본 파일이 없다: {RAW_FILE}")

    header("T2 블록 기여도 재측정 — 표본 정의별")
    print("Δ = AUC(T2 제외) − AUC(T2 포함). 음수면 T2가 기여하고 있다는 뜻이다.")
    print("판정 기준선 = seed 간 자체 편차.")

    df = load_loans()
    all_rows = []
    for name, (sample, schemes) in slices(df).items():
        all_rows += measure(sample, name, args.n_sample, args.seeds, schemes)

    out = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n\n{out.to_string(index=False)}")
    print(f"\n저장: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
