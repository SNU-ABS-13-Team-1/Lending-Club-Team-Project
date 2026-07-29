"""변수 전처리 방식(결측치 처리 규칙) 검증 분석.

outputs/reports/preprocessing_validation_kgj.md 에 실린 표를 전부 재생성한다.
문서의 숫자를 검증하거나 시트가 개정된 뒤 다시 계산할 때 이 스크립트를 돌린다.

검증 대상은 data/processed/lending_club_변수분류_류성환.xlsx_v2.numbers 에 정리된
결측 티어(T0~T3) 분류와 티어별 처방이다.

이 스크립트는 **탐색·검증**용이다 — 모형 학습이나 threshold 결정과는 무관하며,
src/analysis/AGENTS.md의 Sharpe Ratio 기반 의사결정 규칙이 적용되는 단계가 아니다.
5절의 AUC 비교는 처리 방식 간 상대 비교가 목적이므로 AUC를 쓰지만, 승인/거절 기준을
정하는 데 쓰지 않는다.

표본 필터링은 src/preprocessing/AGENTS.md 규칙을 따른다:
- loan_status가 Current / Late (16-30 days) / Late (31-120 days)인 행 제외
- "Does not meet the credit policy. Status:*" 는 접두어를 떼고 동일하게 취급

입력은 data/raw/lending_club_2020_train.csv (~1.2GB)다. 이 파일은 용량 때문에
git에 올리지 않으므로(.gitignore 참고) 팀 공유 채널에서 받아 data/raw/ 에 두고 실행한다.

실행:
    python src/analysis/preprocessing_validation.py             # 1~4, 6절 (수 분)
    python src/analysis/preprocessing_validation.py --with-model  # 5절 AUC 비교까지 (수십 분)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = REPO_ROOT / "data" / "raw" / "lending_club_2020_train.csv"

EXCLUDED = ["Current", "Late (16-30 days)", "Late (31-120 days)"]
POLICY_PREFIX = "Does not meet the credit policy. Status:"

# 시트 「결측티어」 분류 — 검증 대상
T2 = [
    "il_util", "mths_since_rcnt_il", "all_util", "inq_fi", "inq_last_12m", "max_bal_bc",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m",
    "open_rv_24m", "total_bal_il", "total_cu_tl",
]
T3 = [
    "mths_since_last_record", "mths_since_recent_bc_dlq", "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq", "mths_since_last_delinq", "mths_since_recent_inq",
]
T1_SAMPLE = [
    "emp_length", "mort_acc", "avg_cur_bal", "mo_sin_old_il_acct",
    "mths_since_recent_bc", "num_sats", "dti", "revol_util",
]

# 5절 AUC 비교에서 공통 통제로 쓰는 결측 없는 기본 피처
BASE_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc", "emp_length", "term",
]
# T3 중 관측값 자체는 신호가 약해 '더미만' 남겨도 되는 변수 (문서 5절 처방 ③).
# mths_since_last_record는 경계선이다 — 2016~2018 구간에서는 스프레드 2.68pp라 '값도 신호'로
# 판정되지만 2016~2020으로 넓히면 1.82pp로 뒤집힌다. 여기서는 공격적인 쪽(더미만)으로 두고,
# 그래도 성능 손실이 잡음 이하임을 9절에서 확인한다.
T3_FLAT = [
    "mths_since_last_record", "mths_since_recent_bc_dlq", "mths_since_last_major_derog",
    "mths_since_recent_revol_delinq", "mths_since_last_delinq",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
    "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_loans(usecols: list[str]) -> pd.DataFrame:
    """원본을 읽어 전처리 규칙대로 필터링하고 부도 라벨·발행연도를 만든다."""
    need = sorted(set(usecols) | {"loan_status", "issue_d"})
    raw = pd.read_csv(RAW_FILE, usecols=need, low_memory=False)

    status = raw["loan_status"].astype(str).str.replace(POLICY_PREFIX, "", regex=False)
    keep = status.isin(["Fully Paid", "Charged Off"])

    df = raw[keep].copy()
    df["y"] = (status[keep] == "Charged Off").astype(int)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["year"] = df["issue_dt"].dt.year
    return df


def missing_vs_default(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    """결측 여부별 부도율과 2-bin IV를 계산한다 (문서 3절 표)."""
    n_bad = int(df["y"].sum())
    n_good = len(df) - n_bad
    rows = []
    for c in cols:
        miss = df[c].isna()
        bad_m, n_m = int(df.loc[miss, "y"].sum()), int(miss.sum())
        good_m = n_m - bad_m

        # 결측 bin과 관측 bin 두 개로 IV를 구한다.
        iv = 0.0
        for bad, good in ((bad_m, good_m), (n_bad - bad_m, n_good - good_m)):
            p_bad, p_good = bad / n_bad, good / n_good
            iv += (p_bad - p_good) * np.log((p_bad + 1e-9) / (p_good + 1e-9))

        rows.append({
            "변수": c,
            "결측률%": miss.mean() * 100,
            "부도율_결측%": df.loc[miss, "y"].mean() * 100 if n_m else np.nan,
            "부도율_관측%": df.loc[~miss, "y"].mean() * 100,
            "차이pp": (df.loc[miss, "y"].mean() - df.loc[~miss, "y"].mean()) * 100,
            "IV": iv,
        })

    table = pd.DataFrame(rows).sort_values("결측률%", ascending=False)
    print(f"\n----- {label} (전체 부도율 {df['y'].mean() * 100:.2f}%) -----")
    print(table.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    return table


def identical_missing_groups(df: pd.DataFrame, cols: list[str]) -> list[list[str]]:
    """결측벡터가 비트 단위로 일치하는 변수들을 묶는다 (문서 5절 처방 ②)."""
    groups: list[list[str]] = []
    for c in cols:
        for g in groups:
            if df[c].isna().equals(df[g[0]].isna()):
                g.append(c)
                break
        else:
            groups.append([c])
    return groups


def section_sample(df: pd.DataFrame) -> None:
    header("1) 표본 구성 — 행 필터링 규칙 적용 후")
    print(f"완결표본 n = {len(df):,}   부도율 {df['y'].mean() * 100:.2f}%")
    print(f"  Charged Off = {int(df['y'].sum()):,}")
    print("\n연도별 건수·부도율:")
    print(df.groupby("year")["y"].agg(["size", "mean"]).round(4).to_string())


def section_missrate_base(df: pd.DataFrame) -> None:
    """시트의 원본 전수 기준 결측률과 모델링 표본 기준을 대조한다 (문서 2절 ①)."""
    header("2) 결측률 기준 비교 — 시트(원본 전수) vs 모델링 표본(완결)")
    cols = T2 + T3 + ["emp_length"]
    full = pd.read_csv(RAW_FILE, usecols=cols, low_memory=False)
    cmp = pd.DataFrame({
        "원본전수%": full[cols].isna().mean() * 100,
        "완결표본%": df[cols].isna().mean() * 100,
    })
    cmp["차이pp"] = cmp["완결표본%"] - cmp["원본전수%"]
    print(cmp.round(2).to_string())


def section_t2_is_vintage(df: pd.DataFrame) -> None:
    """T2 결측더미가 신용 신호인지 발행 시점인지 판정한다 (문서 2절 ②)."""
    header("3) T2 결측더미 = 발행 시점인가")

    print("연도별 T2 결측률(%):")
    print(df.groupby("year")[T2].apply(lambda g: g.isna().mean() * 100).round(1).to_string())

    d_t2 = df["all_util"].isna().astype(int)
    print(f"\nD_T2 ~ 발행연도 상관        : {d_t2.corr(df['year']):+.4f}")
    print(f"D_T2 == (발행연도 < 2016) 일치율: {(d_t2 == (df['year'] < 2016).astype(int)).mean():.4f}")

    y15 = df[df["year"] == 2015]
    print("\n연도 고정(2015년 — 유일한 혼재 연도) 시 잔여 신호:")
    print(y15.groupby(y15["all_util"].isna())["y"].agg(["size", "mean"]).round(4).to_string())

    print("\n2015년 월별 T2 결측 비율 — 수집 개시 시점:")
    print(y15.groupby(y15["issue_dt"].dt.month)["all_util"].apply(
        lambda s: s.isna().mean()).round(3).to_string())


def section_first_observed(df: pd.DataFrame) -> None:
    """T3로 분류된 변수에 수집 블록 성격이 섞여 있는지 본다 (문서 2절 ③)."""
    header("4) 변수별 최초 관측 연월 — 수집 블록 여부")
    for c in T3 + ["il_util"]:
        first = df.loc[df[c].notna(), "issue_dt"].min()
        print(f"  {c:32s} {first.strftime('%Y-%m') if pd.notna(first) else 'NA'}")


def section_missing_meaning(df: pd.DataFrame) -> None:
    """결측이 신용도 신호인지, 연도 고정 후에도 남는지 확인한다 (문서 3절)."""
    header("5) 결측의 의미 — 결측 여부별 부도율")
    missing_vs_default(df, T3, "T3 사건없음형")
    missing_vs_default(df, T2, "T2 구조적 미수집형")
    missing_vs_default(df, T1_SAMPLE, "T1 수집실패형(일부)")

    print("\n----- 연도 고정(2016~2018) 후에도 신호가 남는가 -----")
    fixed = df[df["year"].between(2016, 2018)]
    for c in T3:
        miss = fixed[c].isna()
        gap = (fixed.loc[miss, "y"].mean() - fixed.loc[~miss, "y"].mean()) * 100

        # 관측값 자체가 신호인지 — 4분위 부도율 스프레드로 본다.
        obs = fixed.loc[~miss, [c, "y"]]
        by_q = obs.groupby(pd.qcut(obs[c], 4, duplicates="drop"), observed=True)["y"].mean() * 100
        spread = by_q.max() - by_q.min()
        verdict = "값도 신호" if spread > 2 else "값은 거의 무신호(더미만 남겨도 됨)"
        print(f"  {c:32s} 더미효과 {gap:+6.2f}pp | 값 4분위 스프레드 {spread:5.2f}pp → {verdict}")


def section_post_2015(df: pd.DataFrame) -> None:
    """수집 개시 이후 남는 결측의 원인을 진단한다 (문서 4절)."""
    header("6) 2015-12 수집 개시 이후의 잔여 결측 — 원인 진단")
    post = df[df["year"] >= 2016]
    print(f"2016년 이후 완결표본 n = {len(post):,}, 부도율 {post['y'].mean() * 100:.2f}%\n")

    for c in ["il_util", "mths_since_rcnt_il", "all_util", "total_bal_il"]:
        miss = post[c].isna()
        if not miss.any():
            print(f"  {c:20s} 결측 0.00% — 잔여 결측 없음")
            continue
        gap = (post.loc[miss, "y"].mean() - post.loc[~miss, "y"].mean()) * 100
        no_il = (post.loc[miss, "open_act_il"] == 0).mean() * 100
        print(f"  {c:20s} 결측 {miss.mean() * 100:5.2f}% | 부도율 차이 {gap:+5.2f}pp "
              f"| 결측건 중 open_act_il==0 비율 {no_il:5.2f}%")

    print("\n→ 결측 원인이 '할부계좌 미보유'라면 중앙값 대체가 아니라 해당없음 더미가 맞다.")


def section_redundancy(df: pd.DataFrame) -> None:
    """결측더미를 몇 개까지 줄일 수 있는지 본다 (문서 5절 처방 ②)."""
    header("7) 결측더미 중복도 — 몇 개면 정보 손실이 0인가")
    for cols, label in ((T2, "T2 14개"), (T2 + T3, "T2+T3 20개")):
        groups = identical_missing_groups(df, cols)
        print(f"\n{label} → 서로 다른 결측벡터 {len(groups)}개")
        for i, g in enumerate(groups, 1):
            print(f"  그룹{i} (결측 {df[g[0]].isna().mean() * 100:5.2f}%): {', '.join(g)}")


def section_sheet_facts() -> None:
    """시트에 적힌 실측 수치를 재현한다 (문서 6절)."""
    header("8) 시트 기재 사실 검증")
    cols = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "total_pymnt", "recoveries",
        "loan_status", "policy_code", "pymnt_plan", "out_prncp", "out_prncp_inv",
        "next_pymnt_d", "grade", "sub_grade", "int_rate", "hardship_flag",
        "debt_settlement_flag", "num_tl_120dpd_2m", "num_tl_30dpd", "delinq_amnt",
        "acc_now_delinq", "issue_d",
    ]
    fin = load_loans(cols)
    fin["int_rate"] = (fin["int_rate"].astype(str)
                       .str.replace("%", "", regex=False).str.strip().astype(float))

    print(f"완결표본 n = {len(fin):,}  (시트 기재 1,115,888 — 차이는 '{POLICY_PREFIX}' 포함 여부)")
    print(f"Charged Off = {int(fin['y'].sum()):,}  (시트 기재 217,366)")

    print("\n[상관계수 — 시트 미결사항 ① 근거]")
    grade = fin["grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    sub = (fin["sub_grade"].str[0].map({g: i for i, g in enumerate("ABCDEFG")}) * 5
           + fin["sub_grade"].str[1].astype(int))
    for name, a, b in [
        ("loan_amnt ~ funded_amnt", fin["loan_amnt"], fin["funded_amnt"]),
        ("funded_amnt ~ funded_amnt_inv", fin["funded_amnt"], fin["funded_amnt_inv"]),
        ("grade ~ int_rate", grade, fin["int_rate"]),
        ("sub_grade ~ int_rate", sub, fin["int_rate"]),
        ("grade ~ sub_grade", grade, sub),
    ]:
        print(f"  {name:32s} {a.corr(b):.6f}")

    print("\n[저분산 주장 — 완결표본 기준 0 초과 비율(%)]")
    for c in ["num_tl_120dpd_2m", "num_tl_30dpd", "delinq_amnt", "acc_now_delinq"]:
        print(f"  {c:22s} {(fin[c] > 0).mean() * 100:6.3f}%")
    print(f"  policy_code 고유값 수 {fin['policy_code'].nunique()} / "
          f"pymnt_plan 고유값 {sorted(fin['pymnt_plan'].dropna().unique())}")

    print("\n[사후변수 '사용불가' 주장]")
    for c in ["out_prncp", "out_prncp_inv"]:
        print(f"  {c:15s} 0 초과 비율 {(fin[c] > 0).mean() * 100:.4f}%")
    print(f"  next_pymnt_d    비결측 비율 {fin['next_pymnt_d'].notna().mean() * 100:.4f}%  (시트 0%)")

    print("\n[현금흐름 관련]")
    bad = fin[fin["y"] == 1]
    ret = bad["total_pymnt"] / bad["funded_amnt"] - 1
    print(f"  부도건 recoveries>0     {(bad['recoveries'] > 0).mean() * 100:.2f}%  (시트 72.9%)")
    print(f"  부도건 total_pymnt>0    {(bad['total_pymnt'] > 0).mean() * 100:.2f}%  (시트 99.6%)")
    print(f"  부도건 총수익률 평균     {ret.mean() * 100:.2f}%  (시트 재검증치 -43.7%)")
    print(f"  hardship_flag=Y         {(fin['hardship_flag'] == 'Y').mean() * 100:.4f}%  (시트 0.09%)")
    print(f"  debt_settlement_flag=Y  {(fin['debt_settlement_flag'] == 'Y').mean() * 100:.4f}%  (시트 2.74%)")


def build_design_matrix(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """처리 방식별 설계행렬을 만든다 (문서 5절 표)."""
    X = df[BASE_FEATURES].copy()

    if scheme == "S1_시트안":              # 원본 20 + 결측더미 20
        for c in T2 + T3:
            X[c] = df[c]
            X["D_" + c] = df[c].isna().astype(int)
    elif scheme == "S2_더미없음":          # 원본 20, 결측은 모델 native 처리에 위임
        for c in T2 + T3:
            X[c] = df[c]
    elif scheme == "S3_블록압축":          # T2 더미 14 -> 1
        for c in T2 + T3:
            X[c] = df[c]
        X["D_T2_block"] = df["all_util"].isna().astype(int)
        for c in T3:
            X["D_" + c] = df[c].isna().astype(int)
    elif scheme == "S4_최소":              # 블록압축 + 무신호 T3는 더미만
        for c in T2:
            X[c] = df[c]
        X["D_T2_block"] = df["all_util"].isna().astype(int)
        for c in T3_FLAT:
            X["D_" + c] = df[c].isna().astype(int)
        X["mths_since_recent_inq"] = df["mths_since_recent_inq"]
        X["D_mths_since_recent_inq"] = df["mths_since_recent_inq"].isna().astype(int)
    else:
        raise ValueError(f"알 수 없는 스킴: {scheme}")
    return X


def section_scheme_comparison(df: pd.DataFrame, n_sample: int = 300_000,
                              n_seeds: int = 5) -> None:
    """결측 처리 방식별 성능을 비교한다 (문서 5절).

    AUC는 처리 방식 간 상대 비교 지표로만 쓴다 — 승인/거절 threshold 결정과 무관하다.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    header("9) 결측 처리 방식별 성능 비교")

    d = df.copy()
    d["term"] = d["term"].astype(str).str.extract(r"(\d+)").astype(float)
    d["emp_length"] = d["emp_length"].map(EMP_LENGTH_MAP)
    d = d.sample(n=min(n_sample, len(d)), random_state=0)
    print(f"표본 {len(d):,}건 · seed 0~{n_seeds - 1} 반복\n")

    schemes = ["S1_시트안", "S3_블록압축", "S4_최소", "S2_더미없음"]
    results = {}
    for scheme in schemes:
        X = build_design_matrix(d, scheme)
        aucs = []
        for seed in range(n_seeds):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, d["y"].values, test_size=0.25, random_state=seed, stratify=d["y"].values)
            model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
                n_jobs=8, random_state=seed)
            model.fit(X_tr, y_tr)
            aucs.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        results[scheme] = np.array(aucs)
        print(f"  {scheme:12s} p={X.shape[1]:3d}  AUC {np.mean(aucs):.5f} "
              f"(표준편차 {np.std(aucs):.5f})")

    base = results["S1_시트안"]
    print(f"\n  시트안 대비 차이 (seed 쌍별) — 비교 기준: seed 간 자체 편차 {np.std(base):.5f}")
    for scheme in schemes[1:]:
        print(f"    {scheme:12s} Δ평균 {(results[scheme] - base).mean():+.5f}")

    # Out-of-time: 학습 구간에만 존재하는 결측더미가 실전에서 작동하는지 확인한다.
    print("\n  Out-of-time (학습 ≤2016 → 검증 2017~2019):")
    tr, te = d[d["year"] <= 2016], d[d["year"].between(2017, 2019)]
    print(f"    학습 n={len(tr):,} (T2 결측 {tr['all_util'].isna().mean() * 100:.1f}%) / "
          f"검증 n={len(te):,} (T2 결측 {te['all_util'].isna().mean() * 100:.1f}%)")
    for scheme in schemes:
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
            n_jobs=8, random_state=0)
        model.fit(build_design_matrix(tr, scheme), tr["y"].values)
        auc = roc_auc_score(te["y"].values,
                            model.predict_proba(build_design_matrix(te, scheme))[:, 1])
        print(f"    {scheme:12s} AUC {auc:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true",
                        help="9절 처리 방식별 AUC 비교까지 실행 (수십 분 소요)")
    args = parser.parse_args()

    if not RAW_FILE.exists():
        raise SystemExit(
            f"원본 파일이 없다: {RAW_FILE}\n"
            "용량 때문에 git에 올리지 않는다 — 팀 공유 채널에서 받아 data/raw/ 에 두고 실행한다.")

    cols = T2 + T3 + T1_SAMPLE + BASE_FEATURES + ["open_act_il"]
    df = load_loans(sorted(set(cols)))

    section_sample(df)
    section_missrate_base(df)
    section_t2_is_vintage(df)
    section_first_observed(df)
    section_missing_meaning(df)
    section_post_2015(df)
    section_redundancy(df)
    section_sheet_facts()
    if args.with_model:
        section_scheme_comparison(df)
    else:
        print("\n(9절 AUC 비교는 --with-model 옵션으로 실행한다)")


if __name__ == "__main__":
    main()
