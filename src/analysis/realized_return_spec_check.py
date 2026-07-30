"""실현수익률 계산 명세(팀원 제공)를 원본 전수 데이터에 적용 가능한지 점검한다.

명세의 각 가정·예외처리·수치 주장을 data/raw/lending_club_2020_train.csv (1,755,295행)에
대해 실측한다. 값을 계산하는 것이 목적이 아니라, 계산이 성립하는지/몇 건이 탈락하는지를 센다.

성격: **탐색·검증(재현)** 스크립트다. 본 파이프라인이 아니므로 `config.yaml`을 경유하지 않는다
(`src/analysis/AGENTS.md` 「이 폴더 스크립트의 성격 구분」).

재현 대상: `decision_log.md` #20 / 이슈 #15 코멘트 ④의 **계산 가능 721,809건 · 계산 불가 1,754건**,
`AGENTS.md`의 **Charged Off 평균 실현수익률 -45.19%(중앙값 -49.00%)**.

⚠️ `data/raw/`는 읽기 전용이다 — 이 스크립트는 원본을 읽기만 하고 결과는 `outputs/`에 쓴다.
실행에 수 분 걸리며 scikit-learn 없이 pandas/numpy만 쓴다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/lending_club_2020_train.csv"
TSY = REPO / "data/processed/us_treasury_GS3_GS5_monthly_2007-06_to_2020-09.csv"
OUT = REPO / "outputs"

COLS = [
    "loan_status", "funded_amnt", "term", "issue_d", "last_pymnt_d",
    "total_pymnt", "installment", "last_pymnt_amnt",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "grade", "debt_settlement_flag",
]

def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)

print(f"reading {RAW} ...")
df = pd.read_csv(RAW, usecols=COLS, low_memory=False)
print(f"rows={len(df):,}  cols={len(df.columns)}")

# ---------------------------------------------------------------- 1. 상태
hdr("1. loan_status 분포 및 명세의 포함 대상 건수 검증")
raw_status = df["loan_status"].astype(str)
print("[원자료 loan_status]")
print(raw_status.value_counts(dropna=False).to_string())

PREFIX = "Does not meet the credit policy. Status:"
status = raw_status.str.replace(PREFIX, "", regex=False).str.strip()
df["status"] = status
print("\n[접두사 제거 후]")
print(status.value_counts(dropna=False).to_string())
print(f"\n접두사 보유 행: {raw_status.str.startswith(PREFIX).sum():,}")

n_fp = (status == "Fully Paid").sum()
n_co = (status == "Charged Off").sum()
print(f"\n명세 주장  : Fully Paid 899,745 / Charged Off 217,826 / 합 1,117,571")
print(f"실측       : Fully Paid {n_fp:,} / Charged Off {n_co:,} / 합 {n_fp + n_co:,}")
print(f"차이       : FP {n_fp - 899745:+,} / CO {n_co - 217826:+,} / 합 {n_fp + n_co - 1117571:+,}")

inc = df[status.isin(["Fully Paid", "Charged Off"])].copy()
print(f"\n계산 대상 표본: {len(inc):,}")

# ---------------------------------------------------------------- 2. 항등식
hdr("2. total_pymnt 항등식 (1센트 오차 이내 성립 주장)")
comp = (inc["total_rec_prncp"].fillna(0) + inc["total_rec_int"].fillna(0)
        + inc["total_rec_late_fee"].fillna(0) + inc["recoveries"].fillna(0))
resid = (inc["total_pymnt"].fillna(0) - comp).abs()
print(f"max |오차| = {resid.max():.6f}")
for thr in (0.01, 0.02, 0.10, 1.0):
    print(f"  |오차| > {thr:>5}: {(resid > thr).sum():,}건")
print("\n전수(포함대상 외 포함) 검증:")
comp_all = (df["total_rec_prncp"].fillna(0) + df["total_rec_int"].fillna(0)
            + df["total_rec_late_fee"].fillna(0) + df["recoveries"].fillna(0))
resid_all = (df["total_pymnt"].fillna(0) - comp_all).abs()
print(f"  max |오차| = {resid_all.max():.6f} / >0.01: {(resid_all > 0.01).sum():,}건")

# ---------------------------------------------------------------- 3. term
hdr("3. term 파싱")
print(inc["term"].value_counts(dropna=False).to_string())
T = inc["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
inc["T"] = T
print(f"\n파싱 실패(NaN): {T.isna().sum():,}")
print(f"고유값: {sorted(T.dropna().unique())}")

# ---------------------------------------------------------------- 4. 날짜/K
hdr("4. issue_d / last_pymnt_d 파싱과 K_i")
issue = pd.to_datetime(inc["issue_d"], format="%b-%Y", errors="coerce")
last = pd.to_datetime(inc["last_pymnt_d"], format="%b-%Y", errors="coerce")
print(f"issue_d 파싱 실패: {issue.isna().sum():,}  (원자료 결측 {inc['issue_d'].isna().sum():,})")
print(f"issue_d 범위: {issue.min().date()} ~ {issue.max().date()}")
print(f"last_pymnt_d 파싱 실패: {last.isna().sum():,}")
print(f"last_pymnt_d 범위: {last.min().date()} ~ {last.max().date()}")

print("\n[last_pymnt_d 결측: 상태별]  ※명세는 Charged Off 2,043건 주장")
print(inc.assign(miss=last.isna()).groupby("status")["miss"].agg(["sum", "size"]).to_string())

inc["issue"] = issue
inc["last"] = last
Ktil = (last.dt.year - issue.dt.year) * 12 + (last.dt.month - issue.dt.month)
inc["Ktil"] = Ktil
inc["K"] = Ktil.clip(lower=0)

print("\n[K̃_i (clip 전) 분포]")
print(f"  결측(날짜 결측)  : {Ktil.isna().sum():,}")
print(f"  K̃ < 0 (음수)     : {(Ktil < 0).sum():,}   ← clip으로 0이 되어 은폐됨")
print(f"  K̃ = 0            : {(Ktil == 0).sum():,}")
print(f"  K̃ = 1            : {(Ktil == 1).sum():,}")
print(f"  K̃ >= 2           : {(Ktil >= 2).sum():,}")
if (Ktil < 0).sum():
    print("\n  K̃<0 상태별:")
    print(inc[Ktil < 0].groupby("status").size().to_string())
    print(f"  K̃<0 최소값: {Ktil.min()}")

print("\n[K vs 계약만기 T]  ※K>T = 계약만기 후 납입 (역할인 구간 필요)")
kt = inc.dropna(subset=["K", "T"])
print(f"  K > T          : {(kt['K'] > kt['T']).sum():,}  ({(kt['K'] > kt['T']).mean():.2%})")
print(f"  K > T (Fully Paid) : {((kt['K'] > kt['T']) & (kt['status'] == 'Fully Paid')).sum():,}")
print(f"  K > T (Charged Off): {((kt['K'] > kt['T']) & (kt['status'] == 'Charged Off')).sum():,}")
print(f"  (K-T) 최대: {(kt['K'] - kt['T']).max():.0f}개월")
print("\n  K-T 분위수 (K>T인 건):")
over = (kt["K"] - kt["T"])[kt["K"] > kt["T"]]
print("   ", {q: round(float(over.quantile(q)), 1) for q in (0.5, 0.9, 0.99, 1.0)})

# ---------------------------------------------------------------- 5. 현금흐름 성분
hdr("5. C_regular / C_recovery / L 의 정합성")
C_reg = inc["total_pymnt"].fillna(0) - inc["recoveries"].fillna(0)
C_rec = inc["recoveries"].fillna(0) - inc["collection_recovery_fee"].fillna(0)
L = inc["last_pymnt_amnt"]
inc["C_reg"] = C_reg
inc["C_rec"] = C_rec

print(f"funded_amnt 결측/≤0      : {inc['funded_amnt'].isna().sum():,} / {(inc['funded_amnt'] <= 0).sum():,}")
print(f"total_pymnt 결측         : {inc['total_pymnt'].isna().sum():,}")
print(f"recoveries 결측          : {inc['recoveries'].isna().sum():,}")
print(f"collection_recovery_fee 결측: {inc['collection_recovery_fee'].isna().sum():,}")
print(f"installment 결측         : {inc['installment'].isna().sum():,}")
print(f"last_pymnt_amnt 결측     : {L.isna().sum():,}")
print("\n  last_pymnt_amnt 결측 상태별:")
print(inc.assign(m=L.isna()).groupby("status")["m"].sum().to_string())

print(f"\nC_regular < 0            : {(C_reg < 0).sum():,}")
print(f"C_recovery < 0 (fee>rec) : {(C_rec < 0).sum():,}   ← 순회수액 음수")
print(f"total_pymnt == 0         : {(inc['total_pymnt'].fillna(0) == 0).sum():,}  → R=-100%")
print(inc.assign(z=inc["total_pymnt"].fillna(0) == 0).groupby("status")["z"].sum().to_string())

K2 = inc["K"] >= 2
print(f"\n[K>=2 & L>C_regular] (명세: 기준계산에서 제외)")
bad_L = K2 & (L > C_reg)
print(f"  건수: {bad_L.sum():,}  ({bad_L.sum() / K2.sum():.3%} of K>=2)")
print(inc[bad_L].groupby("status").size().to_string())
print(f"\n  L < 0 : {(L < 0).sum():,}")
print(f"  L == 0: {(L == 0).sum():,}")

# ---------------------------------------------------------------- 6. D_i 진단
hdr("6. D_i 진단지표 (명세: 표본 9,000건에서 0.9~1.1 비율 FP 84.1% / CO 83.3%)")
ok = K2 & (L <= C_reg) & L.notna() & inc["installment"].notna() & (inc["installment"] > 0)
D = ((C_reg - L) / (inc["K"] - 1)) / inc["installment"]
sub = inc[ok].assign(D=D[ok])
print(f"진단 대상: {len(sub):,}건")
for st, g in sub.groupby("status"):
    band = ((g["D"] >= 0.9) & (g["D"] <= 1.1)).mean()
    print(f"  {st:<12} n={len(g):>9,}  0.9<=D<=1.1: {band:.1%}   "
          f"median D={g['D'].median():.3f}  p05={g['D'].quantile(.05):.3f}  p95={g['D'].quantile(.95):.3f}")
print(f"\n  D > 1.5 비율: {(sub['D'] > 1.5).mean():.2%} / D < 0.5 비율: {(sub['D'] < 0.5).mean():.2%}")

# ---------------------------------------------------------------- 7. 국채 커버리지
hdr("7. 국채 월수익률 커버리지 — F_i(a,T) 계산 가능 여부")
tsy = pd.read_csv(TSY, parse_dates=["observation_date"])
t_min, t_max = tsy["observation_date"].min(), tsy["observation_date"].max()
print(f"국채 파일 범위: {t_min.date()} ~ {t_max.date()}  ({len(tsy)}행)")
print(f"결측: GS3 {tsy['GS3'].isna().sum()} / GS5 {tsy['GS5'].isna().sum()}")

need_max_u = np.maximum(inc["T"], inc["K"] + 6)          # 필요한 마지막 달 offset
need_last = inc["issue"] + pd.to_timedelta(0, "D")
need_last = inc["issue"] + need_max_u.map(lambda u: pd.DateOffset(months=int(u)) if pd.notna(u) else pd.NaT)
inc["need_last"] = pd.to_datetime(need_last)
inc["need_first"] = inc["issue"] + pd.DateOffset(months=1)

beyond = inc["need_last"] > t_max
before = inc["need_first"] < t_min
print(f"\n필요 최종월 > 국채 최종월(2020-09): {beyond.sum():,}건  ({beyond.mean():.1%})")
print(f"필요 최초월 < 국채 최초월(2007-06): {before.sum():,}건  ({before.mean():.1%})")
print(f"둘 중 하나라도 벗어남              : {(beyond | before).sum():,}건  ({(beyond | before).mean():.1%})")
print("\n  [term별 커버리지 부족]")
print(inc.assign(beyond=beyond).groupby("T")["beyond"].agg(["sum", "size", "mean"]).to_string())
print("\n  [필요 최종월 초과분 분포]")
gap = ((inc["need_last"] - t_max).dt.days / 30.44)[beyond]
print("   ", {q: round(float(gap.quantile(q)), 1) for q in (0.5, 0.9, 0.99, 1.0)}, "개월 초과")
print("\n  [issue_d 연도별 커버리지 부족 비율]")
print(inc.assign(beyond=beyond, yr=inc["issue"].dt.year).groupby("yr")["beyond"]
      .agg(["sum", "size", "mean"]).to_string())

# ---------------------------------------------------------------- 8. 탈락 캐스케이드
hdr("8. 제외 규칙 누적 적용 — 최종 계산 가능 건수")
n0 = len(inc)
steps = []
m = pd.Series(True, index=inc.index)
def step(name, cond):
    global m
    before_n = m.sum()
    m = m & cond
    steps.append((name, before_n - m.sum(), m.sum()))

step("issue_d 파싱 가능", inc["issue"].notna())
step("term 파싱 가능", inc["T"].notna())
step("last_pymnt_d 존재", inc["last"].notna())
step("K̃ >= 0 (음수 아님)", inc["Ktil"] >= 0)
step("funded_amnt 유효(>0)", inc["funded_amnt"] > 0)
step("K>=2일 때 installment 존재", ~((inc["K"] >= 2) & inc["installment"].isna()))
step("K>=2일 때 last_pymnt_amnt 존재", ~((inc["K"] >= 2) & L.isna()))
step("K>=2일 때 L <= C_regular", ~((inc["K"] >= 2) & (L > C_reg)))
step("C_recovery >= 0", C_rec >= 0)
step("국채 커버리지 충족", ~(beyond | before))

print(f"시작(FP+CO): {n0:,}\n")
for name, dropped, left in steps:
    print(f"  -{dropped:>9,}  {name:<32} 잔여 {left:>10,}")

OUT.mkdir(parents=True, exist_ok=True)
cascade = pd.DataFrame(steps, columns=["rule", "dropped", "remaining"])
cascade.insert(0, "step", range(1, len(cascade) + 1))
cascade.to_csv(OUT / "realized_return_spec_check_cascade.csv", index=False)
print(f"\n  → {OUT / 'realized_return_spec_check_cascade.csv'}")
print(f"\n최종 계산 가능: {m.sum():,}  ({m.sum()/n0:.1%} of 대상, {m.sum()/len(df):.1%} of 전수)")
print("\n  최종 표본 상태별:")
print(inc[m].groupby("status").size().to_string())
print("\n  국채 커버리지 조건을 뺀 경우:")
m2 = pd.Series(True, index=inc.index)
for c in [inc["issue"].notna(), inc["T"].notna(), inc["last"].notna(), inc["Ktil"] >= 0,
          inc["funded_amnt"] > 0, ~((inc["K"] >= 2) & inc["installment"].isna()),
          ~((inc["K"] >= 2) & L.isna()), ~((inc["K"] >= 2) & (L > C_reg)), C_rec >= 0]:
    m2 &= c
print(f"    {m2.sum():,} ({m2.sum()/n0:.1%})")
print(inc[m2].groupby("status").size().to_string())

# ---------------------------------------------------------------- 9. F=1 근사 수익률
hdr("9. F=1 (재투자 무시) 근사 실현수익률 — 자릿수 sanity check")
calc = inc[m2].copy()
W = calc["C_reg"] + calc["C_rec"]          # F=1 가정
ratio = W / calc["funded_amnt"]
R = ratio ** (12.0 / calc["T"]) - 1.0
calc["R"] = R
print(f"계산 건수: {len(calc):,}")
print(f"ratio(=W/P) 음수: {(ratio < 0).sum():,}  → 분수거듭제곱 불능")
for st, g in calc.groupby("status"):
    print(f"\n  [{st}] n={len(g):,}")
    print(f"    mean R  = {g['R'].mean():+.2%}   median = {g['R'].median():+.2%}")
    print(f"    p01={g['R'].quantile(.01):+.2%}  p25={g['R'].quantile(.25):+.2%}  "
          f"p75={g['R'].quantile(.75):+.2%}  p99={g['R'].quantile(.99):+.2%}")
    print(f"    R < -100%: {(g['R'] < -1).sum():,}   R == -100%: {(g['R'] <= -0.999999).sum():,}")
    print(f"    단순 총수익률 (W/P - 1) mean = {(g['C_reg'] + g['C_rec']).div(g['funded_amnt']).sub(1).mean():+.2%}")

print("\n  ※AGENTS.md 기록: Charged Off 217,366건 평균 실현수익률 -45.19% (중앙값 -49.00%)")
co = calc[calc["status"] == "Charged Off"]
simple = (co["C_reg"] + co["C_rec"]) / co["funded_amnt"] - 1
print(f"    본 계산 CO 단순수익률: mean {simple.mean():+.2%} / median {simple.median():+.2%} (n={len(co):,})")

print("\n  [term별 연율화 효과]")
by_term = calc.groupby(["status", "T"])["R"].agg(["size", "mean", "median", "std"])
print(by_term.to_string())
by_term.reset_index().to_csv(OUT / "realized_return_spec_check_R_by_status_term.csv", index=False)
print(f"\n  → {OUT / 'realized_return_spec_check_R_by_status_term.csv'}")

# ---------------------------------------------------------------- 10. 기타
hdr("10. 기타 점검")
print(f"debt_settlement_flag = Y : {(inc['debt_settlement_flag'] == 'Y').sum():,}")
print(inc.assign(ds=inc["debt_settlement_flag"] == "Y").groupby("status")["ds"].sum().to_string())
print(f"\nFully Paid인데 recoveries > 0 : {((inc['status'] == 'Fully Paid') & (inc['recoveries'] > 0)).sum():,}")
print(f"Charged Off인데 recoveries == 0: {((inc['status'] == 'Charged Off') & (inc['recoveries'].fillna(0) == 0)).sum():,}"
      f"  ({((inc['status'] == 'Charged Off') & (inc['recoveries'].fillna(0) == 0)).mean() / (inc['status'] == 'Charged Off').mean():.1%} of CO)")
print(f"\nK=0 건수 : {(inc['K'] == 0).sum():,}")
print(inc[inc["K"] == 0].groupby("status").size().to_string())
print(f"K=1 건수 : {(inc['K'] == 1).sum():,}")
print(inc[inc["K"] == 1].groupby("status").size().to_string())
print("\n" + "=" * 78)
print("done")
