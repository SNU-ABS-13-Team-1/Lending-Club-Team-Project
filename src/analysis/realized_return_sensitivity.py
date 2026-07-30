"""명세를 실제로 구현해 돌려보고, 세 가지 민감도를 잰다.

(A) 국채 재투자(F)를 넣는 것이 F=1 대비 R을 얼마나 바꾸는가
(B) '마지막 납입액 보존 + 이전 균등배분' 가정 대신 '계약 할부금 우선 배분'을 쓰면
    R이 얼마나 달라지는가  (= 월별 납입내역 부재라는 한계의 실제 비용)
(C) 국채 시계열을 2025-09까지 연장하면(마지막 값 고정 시나리오) 몇 건이 복구되는가

성격: **탐색·검증(재현)** 스크립트다. 본 파이프라인이 아니므로 `config.yaml`을 경유하지 않는다
(`src/analysis/AGENTS.md` 「이 폴더 스크립트의 성격 구분」).

재현 대상: `decision_log.md` #18 / `AGENTS.md`의 **"국채 재투자 가정은 R을 평균 +107.5bp 올린다"**.
`realized_return()`은 #18 재투자 가정(`R = (W/P)^(12/T) − 1`, `W = Σ CFₘ·F(m,T)`)을
**실측 현금흐름 기준으로** 구현한 것이다 — `use_rates=False`가 재투자 0% 민감도 시나리오에 해당한다.

⚠️ 여기서 계산하는 R은 **건별 실현수익률**이다. #20 구조 A′의 정상상환분에 쓰는
**계약 R**(만기까지 납입 가정)은 `src/analysis/realized_return.py`의 `contract_return()`에
구현돼 있다 — **둘의 차이가 B팀 1순위 미결인 조기상환 보정항**이며, 이 스크립트가 그 차이를
재는 재료다. `outputs/reports/handoff_teamb_realized_return.md` 참고.

⚠️ `data/raw/`는 읽기 전용이다 — 이 스크립트는 원본을 읽기만 하고 결과는 `outputs/`에 쓴다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/lending_club_2020_train.csv"
TSY = REPO / "data/processed/us_treasury_GS3_GS5_monthly_2007-06_to_2020-09.csv"
OUT = REPO / "outputs"

COLS = ["loan_status", "funded_amnt", "term", "issue_d", "last_pymnt_d", "total_pymnt",
        "installment", "last_pymnt_amnt", "recoveries", "collection_recovery_fee"]

def hdr(s):
    print("\n" + "=" * 78); print(s); print("=" * 78)

df = pd.read_csv(RAW, usecols=COLS, low_memory=False)
st = df["loan_status"].astype(str).str.replace(
    "Does not meet the credit policy. Status:", "", regex=False).str.strip()
d = df[st.isin(["Fully Paid", "Charged Off"])].copy()
d["status"] = st[st.isin(["Fully Paid", "Charged Off"])]

d["T"] = d["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
d["issue"] = pd.to_datetime(d["issue_d"], format="%b-%Y")
d["last"] = pd.to_datetime(d["last_pymnt_d"], format="%b-%Y", errors="coerce")
d = d[d["last"].notna()].copy()
d["K"] = (((d["last"].dt.year - d["issue"].dt.year) * 12
           + (d["last"].dt.month - d["issue"].dt.month)).clip(lower=0)).astype(int)
d["C_reg"] = d["total_pymnt"] - d["recoveries"]
d["C_rec"] = d["recoveries"] - d["collection_recovery_fee"]
d["L"] = d["last_pymnt_amnt"]
d = d[~((d["K"] >= 2) & (d["L"] > d["C_reg"]))].copy()
print(f"명세 제외규칙 적용 후(국채 커버리지 제외): {len(d):,}")

# ---- 달력월 인덱스 --------------------------------------------------------
d["mi_issue"] = d["issue"].dt.year * 12 + d["issue"].dt.month     # 절대 월 인덱스

tsy = pd.read_csv(TSY, parse_dates=["observation_date"])
tsy["mi"] = tsy["observation_date"].dt.year * 12 + tsy["observation_date"].dt.month
MI_MIN, MI_MAX = int(tsy["mi"].min()), int(tsy["mi"].max())

def build_curves(series, extend_to):
    """월별 연율 국채금리 → 누적성장지수 G(m), 그리고 sum 1/G 용 누적합 H(m).
    extend_to 이후까지 마지막 값을 고정해 연장한다."""
    grid = np.arange(MI_MIN, extend_to + 1)
    r_ann = pd.Series(tsy.set_index("mi")[series]).reindex(grid)
    r_ann = r_ann.ffill()                                  # 2020-09 이후 마지막 값 고정
    r_m = (1 + r_ann.values / 100.0) ** (1 / 12) - 1
    G = np.cumprod(1 + r_m)                                # G(m)
    H = np.cumsum(1.0 / G)                                 # H(m) = sum_{j<=m} 1/G(j)
    return grid, G, H

MI_EXT = (2025 * 12 + 12)   # 2025-12 까지 여유

def realized_return(d, allocation="spec", use_rates=True, restrict_cover=True):
    out = pd.Series(np.nan, index=d.index)
    for term, series in ((36, "GS3"), (60, "GS5")):
        sub = d[d["T"] == term]
        if not len(sub):
            continue
        grid, G, H = build_curves(series, MI_EXT)
        off = grid[0]
        def g(mi):
            return G[np.clip(mi - off, 0, len(G) - 1)]
        def h(mi):
            return H[np.clip(mi - off, 0, len(H) - 1)]

        mi0 = sub["mi_issue"].values
        K = sub["K"].values
        Tm = term
        C_reg, C_rec, L = sub["C_reg"].values, sub["C_rec"].values, sub["L"].values
        inst = sub["installment"].values

        if use_rates:
            G_T = g(mi0 + Tm)
            f_last = G_T / g(mi0 + K)
            f_rec = G_T / g(mi0 + K + 6)
        else:
            G_T = np.ones(len(sub)); f_last = np.ones(len(sub)); f_rec = np.ones(len(sub))

        W = np.zeros(len(sub))

        m0 = K == 0
        m1 = K == 1
        m2 = K >= 2

        if use_rates:
            W[m0] = C_reg[m0] * (G_T[m0] / g(mi0[m0]))
            W[m1] = C_reg[m1] * (G_T[m1] / g(mi0[m1] + 1))
        else:
            W[m0] = C_reg[m0]; W[m1] = C_reg[m1]

        if allocation == "spec":
            # 이전 월 균등배분 A, 마지막 달 L
            A = np.where(m2, (C_reg - L) / np.maximum(K - 1, 1), 0.0)
            if use_rates:
                S = G_T * (h(mi0 + np.maximum(K - 1, 1)) - h(mi0))   # sum_{t=1}^{K-1} F(t,T)
            else:
                S = np.maximum(K - 1, 0).astype(float)
            W[m2] = A[m2] * S[m2] + L[m2] * f_last[m2]
        elif allocation == "installment":
            # 계약 할부금을 1..K-1에 채우고, 남는 금액은 마지막 달로 몰아준다
            cap = inst * np.maximum(K - 1, 0)
            pre = np.minimum(cap, np.maximum(C_reg - L, 0.0))
            A = np.where(m2, pre / np.maximum(K - 1, 1), 0.0)
            tail = np.where(m2, C_reg - L - pre, 0.0)      # 초과 상환분은 마지막 달
            if use_rates:
                S = G_T * (h(mi0 + np.maximum(K - 1, 1)) - h(mi0))
            else:
                S = np.maximum(K - 1, 0).astype(float)
            W[m2] = A[m2] * S[m2] + (L[m2] + tail[m2]) * f_last[m2]
        elif allocation == "lump_end":
            # 극단 비교: 정규 수령액 전부를 마지막 납입월에 수령
            W[m2] = C_reg[m2] * f_last[m2]

        W = W + C_rec * f_rec
        R = (W / sub["funded_amnt"].values) ** (12.0 / Tm) - 1.0
        out.loc[sub.index] = R
    return out

# ---------------------------------------------------------------- 커버리지
hdr("C. 국채 커버리지 — 얼마나 연장해야 하나")
need_last_mi = d["mi_issue"] + np.maximum(d["T"], d["K"] + 6)
need_max = int(need_last_mi.max())
print(f"필요한 최종 달력월: {need_max // 12}-{need_max % 12 or 12:02d}   (국채 파일 최종: 2020-09)")
cover = need_last_mi <= MI_MAX
print(f"현재 파일로 계산 가능: {cover.sum():,} / {len(d):,}  ({cover.mean():.1%})")
print(f"부족분                : {(~cover).sum():,}  ({(~cover).mean():.1%})")
print("\n부족분 상태별:")
print(d.assign(nc=~cover).groupby("status")["nc"].agg(["sum", "size", "mean"]).to_string())
print("\n부족분 issue 연도별 비율 (모형 학습표본 편향 확인):")
print(d.assign(nc=~cover, yr=d["issue"].dt.year).groupby("yr")["nc"]
      .agg(["sum", "size", "mean"]).tail(8).to_string())

# ---------------------------------------------------------------- A
hdr("A. 국채 재투자(F) 유무에 따른 R 차이  [커버리지 충족 799,930건]")
dc = d[cover].copy()
R_F1 = realized_return(dc, allocation="spec", use_rates=False)
R_Ftsy = realized_return(dc, allocation="spec", use_rates=True)
cmp = pd.DataFrame({"status": dc["status"], "T": dc["T"], "R_F1": R_F1, "R_tsy": R_Ftsy})
cmp["diff_bp"] = (cmp["R_tsy"] - cmp["R_F1"]) * 10000
print(cmp.groupby(["status", "T"])[["R_F1", "R_tsy"]].mean().to_string())
print("\n차이(bp) 요약:")
print(cmp.groupby(["status", "T"])["diff_bp"].describe()[["mean", "50%", "max"]].to_string())
print(f"\n전체 평균 차이: {cmp['diff_bp'].mean():.1f}bp   |차이|>50bp 비율: {(cmp['diff_bp'].abs() > 50).mean():.1%}")

# ---------------------------------------------------------------- B
hdr("B. 월별 배분 가정 민감도 (국채 F 적용, 커버리지 충족분)")
R_inst = realized_return(dc, allocation="installment", use_rates=True)
R_lump = realized_return(dc, allocation="lump_end", use_rates=True)
sens = pd.DataFrame({"status": dc["status"], "T": dc["T"],
                     "spec": R_Ftsy, "installment": R_inst, "lump_end": R_lump})
print(sens.groupby(["status", "T"])[["spec", "installment", "lump_end"]].mean().to_string())
print("\nspec - installment (bp):")
sens["d_inst"] = (sens["spec"] - sens["installment"]) * 10000
sens["d_lump"] = (sens["spec"] - sens["lump_end"]) * 10000
print(sens.groupby("status")[["d_inst", "d_lump"]].agg(["mean", "median"]).to_string())
print(f"\n|spec-installment| > 50bp : {(sens['d_inst'].abs() > 50).mean():.2%}")
print(f"|spec-lump_end|    > 50bp : {(sens['d_lump'].abs() > 50).mean():.2%}")

OUT.mkdir(parents=True, exist_ok=True)
g = sens.groupby(["status", "T"])
summary = pd.DataFrame({
    "n": g.size(),
    "R_reinvest_0pct": cmp.groupby(["status", "T"])["R_F1"].mean(),   # 재투자 0% (민감도)
    "R_reinvest_tsy": g["spec"].mean(),                               # 국채 재투자 (#18 확정)
    "diff_bp": cmp.groupby(["status", "T"])["diff_bp"].mean(),        # (A) 재투자 가정 효과
    "R_alloc_installment": g["installment"].mean(),                   # (B) 배분 가정 대안 1
    "R_alloc_lump_end": g["lump_end"].mean(),                         # (B) 배분 가정 대안 2
    "d_inst_bp": g["d_inst"].mean(),
    "d_lump_bp": g["d_lump"].mean(),
})
summary.reset_index().to_csv(OUT / "realized_return_sensitivity.csv", index=False)
print(f"\n  → {OUT / 'realized_return_sensitivity.csv'}")

# ---------------------------------------------------------------- 최종 분포
hdr("최종 R 분포 (명세 그대로, 국채 F 적용)")
for s, g in cmp.groupby("status"):
    print(f"[{s}] n={len(g):,}  mean={g['R_tsy'].mean():+.2%}  median={g['R_tsy'].median():+.2%}  "
          f"sd={g['R_tsy'].std():.2%}  min={g['R_tsy'].min():+.2%}  max={g['R_tsy'].max():+.2%}")
print(f"\n전체 mean={cmp['R_tsy'].mean():+.2%}  sd={cmp['R_tsy'].std():.2%}")
print(f"R < -100% 건수: {(cmp['R_tsy'] < -1).sum():,}   R > +50% 건수: {(cmp['R_tsy'] > 0.5).sum():,}")

# 마지막 값 고정 연장 시나리오
hdr("C-2. 국채를 2020-09 값으로 고정 연장하면 (전체 1,112,552건)")
R_all = realized_return(d, allocation="spec", use_rates=True)
print(f"계산 성공: {R_all.notna().sum():,}")
allc = pd.DataFrame({"status": d["status"], "R": R_all})
print(allc.groupby("status")["R"].agg(["size", "mean", "median", "std"]).to_string())
print("\n※ 2020-09 값 고정은 임시 가정이다. 실제로는 FRED에서 2025-09까지 받아 채워야 한다.")
print("=" * 78)
