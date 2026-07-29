"""2016년 이후 잔여 결측을 어떻게 채울 것인가 — NaN 유지 vs 0 대체 vs 전용 더미.

## 왜 재는가

`preprocessing_validation_kgj.md` 4절은 T2 결측을 시점별로 쪼개 처방했다.

    ~2015-11 : 미수집(vintage). 정보가 존재하지 않음  -> 결측더미를 만들지 않는다
    2015-12~ : 진짜 "할부계좌 없음"                    -> 성격별로 나눠 처리

그리고 2016년 이후 구간에 대해 **금액·개수형은 0 대체가 옳다**(계좌가 없으면 잔액도
개설건수도 진짜 0)고 권고했다. 류성환·이지희 문서는 같은 구간에 **중앙값 대체**를 적었다.
교차검증 문서는 이 대립을 "반드시 정할 것"으로 올렸다.

그런데 XGBoost 단일화를 전제하면 중앙값 대체 주장은 근거를 잃는다(NaN을 그대로 받으므로).
남는 질문은 **"0으로 볼 것인가 NaN으로 둘 것인가"** 하나인데, 그 답을 재기 전에
**애초에 0을 채울 대상이 얼마나 있는지**부터 확인해야 한다.

## 무엇을 확인하는가

1단계 — 만기컷 표본의 2016년 이후 발행분에서 **T2 14개의 변수별 결측률**을 센다.
2단계 — 처리 스킴 4개의 AUC를 비교한다. 판정 기준선은 seed 간 자체 편차다.

    N  : NaN 유지                      (권고안 / 기준선)
    Z  : 비율·시점형까지 0으로 채움      (0 대체가 유리한지)
    D  : NaN 유지 + has_no_il_account   (전용 더미가 기여하는지)
    Za : 2016+ 모든 T2 결측을 0으로     (시트가 금지한 방식 — sanity check)

## 결과 (만기+버퍼6m 723,563건, seeds=5, 2026-07-29)

1단계가 결정적이다.

    il_util             2016+ 결측률 14.12%   <- 비율형
    mths_since_rcnt_il  2016+ 결측률  3.14%   <- 시점형
    all_util            2016+ 결측률  0.02%
    나머지 금액·개수형 11개  2016+ 결측률  0.01%   <- 사실상 없다

**"0 대체가 의미상 맞다"고 했던 금액·개수형 변수에는 채울 결측이 남아 있지 않다.**
25만 건 중 약 25건이다. 실제로 결측이 남는 두 변수(`il_util`, `mths_since_rcnt_il`)는
분모가 0이라 정의 자체가 불가능한 경우이므로 0이 틀린 값이다.

2단계도 같은 방향이다.

    스킴                 AUC       N 대비      나쁜 seed
    N  NaN 유지         0.68080   -          -          (seed 편차 0.00134)
    Z  비율형까지 0      0.68074   -0.00006   3/5
    D  NaN + 더미       0.68089   +0.00009   2/5
    Za 2016+ 전부 0     0.68089   +0.00009   2/5

최대 차이 0.00009는 seed 편차의 **1/15**이고 부호도 seed마다 갈린다. 네 스킴이 같다.

## 결론

**전부 NaN으로 두고 아무것도 만들지 않는다.**

- 성능이 같으면 더 단순한 쪽을 택한다. 의미상으로도 `0 / 0`에 0을 넣는 것은 틀렸다.
- `has_no_il_account` 더미도 불필요하다. **`open_act_il`이 이미 피처로 들어가 있어**
  트리가 `open_act_il < 1`로 직접 분기할 수 있기 때문이다 — 같은 정보를 두 번 넣는 셈이다.
- 이는 결측더미 전반을 만들지 않기로 한 판단(`preprocessing_validation_kgj.md` 5절 처방 ①,
  `decision_log.md` #8의 GBM native 결측 처리 원칙)과 같은 결론이다.

정리하면 전제(XGBoost 단일)에서 결측 처리 규칙은 하나로 통일된다.

    T2 pre-2016 -> NaN 그대로 (미수집)
    T2 2016+    -> NaN 그대로 (계좌 미보유 — open_act_il이 이미 그 정보를 담고 있다)
    T3          -> NaN 그대로 (사건 없음)
    파생 더미     -> 만들지 않음

주의: 이 결론은 **XGBoost를 쓴다는 전제**에 의존한다. NaN을 받지 못하는 선형모형을
쓴다면 대체값과 더미가 다시 필요해진다.

AUC는 처리 방식 간 상대 비교용이다 — 승인/거절 threshold 결정과 무관하다
(`src/analysis/AGENTS.md`).

실행:
    python src/analysis/missing_scheme_comparison.py
    python src/analysis/missing_scheme_comparison.py --seeds 3   # 빠르게
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_validation import (  # noqa: E402  (같은 폴더의 검증 스크립트와 정의 공유)
    BASE_FEATURES, EMP_LENGTH_MAP, POLICY_PREFIX, RAW_FILE, T2, T3, header,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "outputs" / "missing_scheme_comparison.csv"

MATURITY_CUTOFF = pd.Period("2020-04", freq="M")

# 2016년 이후 결측이 실제로 남는 두 변수. 분모가 0이라 값이 정의되지 않는 성격이다
# (il_util = 할부잔액/할부한도, mths_since_rcnt_il = 최근 할부계좌 개설 후 경과월).
RATIO_LIKE = ["il_util", "mths_since_rcnt_il"]

# 계좌가 없으면 값도 진짜 0인 성격 — "0 대체가 옳다"고 권고했던 대상.
AMOUNT_LIKE = [v for v in T2 if v not in RATIO_LIKE + ["all_util"]]

FEATURES = BASE_FEATURES + T2 + T3

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
    colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", n_jobs=8,
)


def load_matured() -> pd.DataFrame:
    """완결표본에 만기 조건을 얹고 발행연도를 붙인다."""
    cols = sorted(set(T2 + T3 + BASE_FEATURES + ["loan_status", "issue_d", "term"]))
    raw = pd.read_csv(RAW_FILE, usecols=cols, low_memory=False)

    status = raw["loan_status"].astype(str).str.replace(POLICY_PREFIX, "", regex=False)
    keep = status.isin(["Fully Paid", "Charged Off"])

    df = raw[keep].copy()
    df["y"] = (status[keep] == "Charged Off").astype(int)
    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["year"] = issue_dt.dt.year
    df["term_m"] = df["term"].astype(str).str.extract(r"(\d+)").astype(int)
    df = df[(issue_dt.dt.to_period("M") + df["term_m"]) <= MATURITY_CUTOFF].copy()

    df["term"] = df["term_m"].astype(float)
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """1단계 — 2016년 이후 발행분에서 T2 변수별 결측률. 0을 채울 대상이 있는지 본다."""
    post = df["year"] >= 2016
    rows = []
    for v in T2:
        kind = ("비율형" if "util" in v else
                "시점형" if v in RATIO_LIKE else "금액·개수형")
        rows.append({"변수": v, "성격": kind,
                     "전체_결측률%": round(df[v].isna().mean() * 100, 2),
                     "2016+_결측률%": round(df.loc[post, v].isna().mean() * 100, 2)})
    out = pd.DataFrame(rows).sort_values("2016+_결측률%", ascending=False)
    print(f"\n2016년 이후 발행 {post.sum():,}건 ({post.mean() * 100:.1f}%) 기준\n")
    print(out.to_string(index=False))
    print("\n-> 금액·개수형은 2016년 이후 결측이 사실상 없다. 0을 채울 대상 자체가 없다.")
    return out


def design(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """스킴별 설계행렬. 0 대체는 2016년 이후 발행분에만 적용한다.

    2015년 이전은 '미수집'이라 0을 넣으면 옛 차주 전체를 무활동 차주로 왜곡하므로
    (시트 개정 5번), 어떤 스킴에서도 건드리지 않는다.
    """
    X = df[FEATURES].copy()
    post = (df["year"] >= 2016).values
    if scheme == "Z_비율형까지0":
        for v in RATIO_LIKE:
            X.loc[post & X[v].isna().values, v] = 0
    elif scheme == "D_NaN+더미":
        X["has_no_il_account"] = (df["open_act_il"] == 0).astype(int)
    elif scheme == "Za_2016+전부0":
        for v in T2:
            X.loc[post & X[v].isna().values, v] = 0
    return X


def compare(df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    """2단계 — 스킴별 AUC. 기준선은 N(NaN 유지)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    schemes = ["N_NaN유지", "Z_비율형까지0", "D_NaN+더미", "Za_2016+전부0"]
    scores: dict[str, np.ndarray] = {}
    for scheme in schemes:
        X = design(df, scheme)
        aucs = []
        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, df["y"].values, test_size=0.2, random_state=seed,
                stratify=df["y"].values)
            model = XGBClassifier(**XGB_PARAMS, random_state=seed).fit(X_tr, y_tr)
            aucs.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        scores[scheme] = np.array(aucs)

    base = scores["N_NaN유지"]
    seed_sd = base.std()
    rows = []
    print(f"\n{'스킴':<18}{'AUC 평균':>11}{'N 대비':>11}{'나쁜 seed':>11}  판정")
    print("-" * 68)
    for scheme in schemes:
        a = scores[scheme]
        delta = a.mean() - base.mean()
        worse = int((a < base).sum())
        if scheme == "N_NaN유지":
            verdict = f"기준선 (seed 편차 {seed_sd:.5f})"
            shown_delta, shown_worse = "—", "—"
        else:
            verdict = "차이 없음" if abs(delta) <= seed_sd else "차이 있음"
            shown_delta, shown_worse = f"{delta:+.5f}", f"{worse}/{n_seeds}"
        rows.append({"스킴": scheme, "AUC": round(a.mean(), 5),
                     "N대비delta": round(delta, 5), "나쁜seed": worse,
                     "seed편차": round(seed_sd, 5), "판정": verdict})
        print(f"{scheme:<16}{a.mean():>11.5f}{shown_delta:>11}{shown_worse:>11}  {verdict}")
    print("-" * 68)
    spread = max(abs(scores[s].mean() - base.mean()) for s in schemes)
    print(f"스킴 간 최대 차이 {spread:.5f} = seed 편차의 {spread / seed_sd:.2f}배"
          f" -> {'구분 불가' if spread < seed_sd else '유의'}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="2016+ 잔여 결측 처리 스킴 비교")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    if not RAW_FILE.exists():
        raise SystemExit(f"원본 파일이 없다: {RAW_FILE}")

    header("2016년 이후 잔여 결측 — NaN 유지 vs 0 대체 vs 전용 더미")
    df = load_matured()
    print(f"만기+버퍼6m 표본 {len(df):,}건  부도율 {df['y'].mean() * 100:.2f}%")

    header("1단계 — 0을 채울 대상이 있는가")
    miss = report_missing(df)

    header("2단계 — 스킴별 AUC")
    res = compare(df, args.seeds)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([miss.assign(구분="결측률"), res.assign(구분="AUC")],
              ignore_index=True).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
