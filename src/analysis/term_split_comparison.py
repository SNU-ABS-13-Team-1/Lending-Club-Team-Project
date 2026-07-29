"""36m/60m 분리 모형 vs 통합 모형 — 어느 쪽이 나은가.

## 왜 재는가

만기 컷(`issue_d + term <= 2020-04`)을 적용하면 term 구성이 크게 바뀐다.
60개월 대출은 만기가 2년 늦게 오므로 발행 상한이 2015-04까지로 밀리고
(36개월은 2017-04), 그 2년이 Lending Club 물량 급증기와 겹쳐 60m가 3분의 2 날아간다.

    36개월 : 837,476 -> 623,125건 (잔존 74.4%, 비중 86.1%)
    60개월 : 280,095 -> 100,438건 (잔존 35.9%, 비중 13.9%)

두 상품은 부도율도 다르다(36m 14.61% vs 60m 26.18%). 그래서 `decision_log.md` #12 (5)는
term-vintage 교락의 완화책으로 **36m/60m 분리 모형**을 후보로 올려 뒀다.

이 스크립트는 그 후보를 실측으로 판정한다. **같은 표본·같은 하이퍼파라미터·같은 test set**
위에서 두 방식만 바꿔 비교하므로, 차이가 나면 그것은 분리 여부에서 온 것이다.

    (a) 통합 : 전체로 모형 1개 학습. `term`을 피처로 투입한다
    (b) 분리 : 36m/60m 각각 모형 1개씩. `term`은 그룹 내 상수라 제외한다

## 어떻게 재는가

평가는 **동일한 test set**에서 세 구간으로 나눠 본다. (b)의 전체 AUC는 두 모형의
예측확률을 각 행의 term에 맞춰 이어 붙여 계산한다.

    - 전체        : 두 상품을 한 포트폴리오로 볼 때의 순위매김 성능
    - 36개월 안에서 : 36m 부분집합에서만 잰 성능
    - 60개월 안에서 : 60m 부분집합에서만 잰 성능  <- 분리의 기대 이익이 가장 큰 곳

판정 기준선은 늘 그렇듯 **seed 간 자체 편차**다. 그보다 작으면 잡음으로 본다.
다만 60m는 차이와 편차가 비슷한 크기로 나오므로 **seed별 부호 일관성**도 함께 본다.

## 결과 (seeds=5 기준, 2026-07-29)

    구간            (a) 통합    (b) 분리    차이(b-a)   seed편차   판정
    전체            0.68080    0.68033    -0.00047   0.00150   미세하게 나쁨 (부호 5/5)
    36개월 안에서     0.67197    0.67223    +0.00025   0.00131   차이 없음
    60개월 안에서     0.62783    0.62219    -0.00564   0.00563   뚜렷하게 나쁨 (5/5 seed)

**분리는 이득이 없고, 60개월에서는 오히려 일관되게 나쁘다.**

이유는 표본 크기다. 분리하면 60m 모형이 8만 건으로 학습하는데, 통합하면 58만 건을 본다.
신용위험의 기본 패턴(FICO가 낮으면 위험, dti가 높으면 위험)은 상품 종류와 무관하게
공통이므로 36m 데이터로 배운 것이 60m 예측에도 전이된다. 데이터를 7배 더 본 쪽이 이긴다.

그리고 트리 모형은 필요하면 스스로 나눈다. `term`을 피처로 넣어 두면 두 상품이 정말
다르게 행동하는 지점에서 XGBoost가 알아서 `term < 48`로 분기한다. 사람이 미리 갈라 주면
그 분기를 강제하는 대신 **공통 패턴을 배울 기회를 뺏는** 셈이다.

## 성능 밖의 비용 (수치로 재지 않은 부분 — 문서 8절 참조)

- **threshold 탐색이 2차원이 된다.** 본 과제의 판단 기준은 Sharpe Ratio이고, Sharpe는
  포트폴리오 단위 지표다. 36m/60m를 따로 최적화해 합치면 전체 Sharpe가 최적이라는 보장이
  없으므로 (t36, t60) 격자를 동시에 탐색해야 한다. Validation에서 맞추는 파라미터가
  1개에서 2개로 늘어 과적합 위험도 커진다.
- **변수중요도가 두 세트가 된다.** 특히 60m 모형에서는 T2 14개가 전량 결측이라
  (60m 만기도래 = 2015-04 이전 발행 vs T2 수집 개시 2015-12) 피처가 통째로 죽는다.

## 결론

**통합 모형을 쓴다.** 다만 학습한 뒤 예측값을 term으로 나눠 성능표를 따로 보고한다 —
모형을 나누지 않고도 "36m/60m가 다르다"는 진단은 그대로 얻을 수 있다.

주의: 이것은 **PD 모형**을 나누지 않는다는 뜻이지, 수익률 계산에서 term을 무시한다는
뜻이 아니다. 60개월 대출은 5년간 현금흐름이 나오므로 IRR/Sharpe 계산식에는 만기가
반드시 들어간다 (`decision_log.md` #4).

AUC는 처리 방식 간 상대 비교용이다 — 승인/거절 threshold 결정과 무관하다
(`src/analysis/AGENTS.md`).

실행:
    python src/analysis/term_split_comparison.py
    python src/analysis/term_split_comparison.py --seeds 3   # 빠르게
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_validation import (  # noqa: E402  (같은 폴더의 검증 스크립트와 정의 공유)
    BASE_FEATURES, EMP_LENGTH_MAP, POLICY_PREFIX, RAW_FILE, T2, T3, header,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = REPO_ROOT / "outputs" / "term_split_comparison.csv"

# 만기 + 버퍼 6개월. 스냅샷(last_credit_pull_d 최종월) = 2020-10.
MATURITY_CUTOFF = pd.Period("2020-04", freq="M")

# 결측더미 없는 권고안(preprocessing_validation_kgj.md 5절 S2). 값만 투입하고
# 결측은 XGBoost의 native 분기에 맡긴다.
FEATURES = BASE_FEATURES + T2 + T3

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
    colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", n_jobs=8,
)


def load_matured() -> pd.DataFrame:
    """완결표본에 만기 조건을 얹어 반환한다."""
    cols = sorted(set(T2 + T3 + BASE_FEATURES + ["loan_status", "issue_d", "term"]))
    raw = pd.read_csv(RAW_FILE, usecols=cols, low_memory=False)

    status = raw["loan_status"].astype(str).str.replace(POLICY_PREFIX, "", regex=False)
    keep = status.isin(["Fully Paid", "Charged Off"])

    df = raw[keep].copy()
    df["y"] = (status[keep] == "Charged Off").astype(int)
    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["term_m"] = df["term"].astype(str).str.extract(r"(\d+)").astype(int)
    df = df[(issue_dt.dt.to_period("M") + df["term_m"]) <= MATURITY_CUTOFF].copy()

    df["term"] = df["term_m"].astype(float)
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    return df


def one_seed(df: pd.DataFrame, seed: int) -> dict:
    """한 seed에서 (a)통합과 (b)분리를 같은 test set으로 평가한다."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    train, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["y"])
    tr36, tr60 = train["term_m"] == 36, train["term_m"] == 60
    te36, te60 = test["term_m"] == 36, test["term_m"] == 60

    # (a) 통합 — term을 피처로 투입
    unified = XGBClassifier(**XGB_PARAMS, random_state=seed)
    unified.fit(train[FEATURES], train["y"])
    p_a = unified.predict_proba(test[FEATURES])[:, 1]

    # (b) 분리 — term은 그룹 내 상수이므로 제외
    feats_split = [f for f in FEATURES if f != "term"]
    p_b = np.empty(len(test))
    for mask_tr, mask_te in ((tr36, te36), (tr60, te60)):
        model = XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(train.loc[mask_tr, feats_split], train.loc[mask_tr, "y"])
        p_b[mask_te.values] = model.predict_proba(test.loc[mask_te, feats_split])[:, 1]

    def auc(mask, pred):
        return roc_auc_score(test["y"][mask], pred[mask.values])

    everything = pd.Series(True, index=test.index)
    return {
        "seed": seed,
        "통합_전체": auc(everything, p_a), "분리_전체": auc(everything, p_b),
        "통합_36m": auc(te36, p_a), "분리_36m": auc(te36, p_b),
        "통합_60m": auc(te60, p_a), "분리_60m": auc(te60, p_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="36m/60m 분리 모형 vs 통합 모형")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    if not RAW_FILE.exists():
        raise SystemExit(f"원본 파일이 없다: {RAW_FILE}")

    header("36m/60m 분리 모형 vs 통합 모형")
    print("같은 표본·같은 하이퍼파라미터·같은 test set. 판정 기준선은 seed 간 자체 편차.")

    df = load_matured()
    n36, n60 = (df["term_m"] == 36).sum(), (df["term_m"] == 60).sum()
    print(f"\n만기+버퍼6m 표본 {len(df):,}건  부도율 {df['y'].mean() * 100:.2f}%")
    for label, mask in (("36개월", df["term_m"] == 36), ("60개월", df["term_m"] == 60)):
        sub = df[mask]
        print(f"  {label} {len(sub):>8,}건 ({len(sub) / len(df) * 100:5.1f}%)  "
              f"부도율 {sub['y'].mean() * 100:5.2f}%")
    if min(n36, n60) == 0:
        raise SystemExit("한쪽 term이 비어 있어 비교할 수 없다.")

    res = pd.DataFrame([one_seed(df, s) for s in range(args.seeds)])

    rows = []
    print(f"\n{'구간':<16}{'(a) 통합':>11}{'(b) 분리':>11}{'차이(b-a)':>12}"
          f"{'seed편차':>10}  판정")
    print("-" * 78)
    for label, col_a, col_b in (("전체", "통합_전체", "분리_전체"),
                                ("36개월 안에서", "통합_36m", "분리_36m"),
                                ("60개월 안에서", "통합_60m", "분리_60m")):
        delta = res[col_b].mean() - res[col_a].mean()
        seed_sd = res[col_a].std()
        worse = int((res[col_b] < res[col_a]).sum())
        direction = "나쁨" if delta < 0 else "나음"
        # 크기(잡음 대비)와 부호 일관성을 분리해서 표기한다. 크기가 잡음 이하라도
        # 모든 seed에서 부호가 같으면 "우연"으로 보기 어렵기 때문이다.
        if abs(delta) > seed_sd:
            verdict = f"분리가 뚜렷하게 {direction} ({worse}/{args.seeds} seed)"
        elif worse in (0, args.seeds):
            verdict = f"분리가 미세하게 {direction} (부호 일관 {worse}/{args.seeds}, 크기는 잡음 이하)"
        else:
            verdict = "차이 없음"
        rows.append({"구간": label, "AUC_통합": round(res[col_a].mean(), 5),
                     "AUC_분리": round(res[col_b].mean(), 5), "delta": round(delta, 5),
                     "seed편차": round(seed_sd, 5), "분리가나쁜seed": worse,
                     "판정": verdict})
        print(f"{label:<14}{res[col_a].mean():>11.5f}{res[col_b].mean():>11.5f}"
              f"{delta:>+12.5f}{seed_sd:>10.5f}  {verdict}")
    print("-" * 78)
    print("60개월은 차이와 편차가 비슷하므로 seed별 부호 일관성으로 판정한다.")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nseed별 원자료:\n{res.round(5).to_string(index=False)}")
    print(f"\n저장: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
