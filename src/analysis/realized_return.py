"""실현수익률 · 초과수익률(XR) — 구조 A′의 **잠정(provisional)** 구현.

`decision_log.md` #20이 확정한 구조를 계산 가능한 형태로 옮긴 것이다.

```
E[XR_i] = (1 − p̂_i) · XR_정상,i        +  p̂_i · mu_부도,d(i)
                  ↑ 건별 계약 현금흐름          ↑ 그룹 평균 (PD 분위 × term)
```

## ⚠️ 잠정값으로 고정한 미확정 3건 (#20)

| 항목 | 선택지 | 여기서 쓴 값 |
| --- | --- | --- |
| 국채 금리 기준 | ⓐ실제경로 / ⓑ상수 / **ⓒ발행시점 고정** | **ⓒ** (#18·#20 권고) |
| 투자자 서비스수수료 | ~1% | **0%** |
| 조기상환 보정 `Δ̄_조기상환,d` | 미확정 | **0** (= 보정 없음) |

**조기상환 보정이 0이라는 것은 정상상환분 `R`이 과대추정된다는 뜻이다.** 국채 재투자 가정
아래서 조기상환은 `R`을 낮춘다(12% 대출을 12개월에 회수하면 남은 24개월을 약 2% 국채로
굴려야 한다). 또한 보정이 없으면 **`var_정상 = 0`** 이 되어 `Var[XR]`의 분모가 부도 항만
반영한다 — 정상상환분의 분산은 원래 `실현 R − 계약 R`의 산포에서 나온다(#20).

→ B팀이 확정하면 `PrematureRepaymentAdjustment`만 갈아 끼우면 된다.
   **모형 재학습은 불필요하고 칸별 통계표와 threshold만 재계산**하면 된다(#19·#20).

## 손실값을 상수로 박지 않는다

`src/analysis/AGENTS.md`의 구현 조건이다. 부도 손실은 하드코딩된 −100%나 −45%가 아니라
**데이터에서 칸별로 추정한 `mu_부도,d`** 이며, 재투자 가정·수수료·보정항은 전부
`ReturnAssumptions`로 주입받는다.

## ⚠️ 여기 쓰는 컬럼은 전부 사후(post-approval)다

`total_pymnt`·`recoveries`·`last_pymnt_d` 등은 **결과변수 쪽**이라 피처 테이블에 넣으면
누수다(`src/preprocessing/AGENTS.md`). 이 모듈의 산출물은 threshold·Sharpe 단계에서
`id`로 결합해 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from utils.config import load_config
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.config import load_config


TREASURY_FILENAME = "us_treasury_GS3_GS5_monthly_2007-06_to_2020-09.csv"

# 부도 회수액 수령 시점 — 최종납입 K개월 뒤 6개월.
# ⚠️ 이 규칙은 아직 어느 문서에도 근거가 기록돼 있지 않다(#17 작업 10, B팀).
RECOVERY_LAG_MONTHS = 6


@dataclass(frozen=True)
class ReturnAssumptions:
    """수익률 계산 가정 — **전부 주입받는다.** 상수로 박지 않는다(`src/analysis/AGENTS.md`)."""

    reinvest: str = "treasury"
    """`"treasury"`: 잔존기간 매칭 국채 재투자(#18 확정) / `"cash"`: 0% 재투자(민감도용)."""

    treasury_basis: str = "issue_fixed"
    """ⓒ 발행시점 고정(권고·잠정). ⓐ실제경로·ⓑ상수는 아직 미구현이다."""

    servicing_fee_annual: float = 0.0
    """투자자 서비스수수료 연율. 잠정 0% (#20 미확정)."""

    prepayment_adjustment: float = 0.0
    """`contract_return()`에 더하는 **스칼라** 보정항. 기본 0을 유지한다.

    ✅ 조기상환 보정은 2026-07-31부터 **칸별로 데이터에서 추정**한다
    (`normal_cell_stats()`의 `Δ̄_조기상환,d`) — 이 스칼라는 민감도 실험용 수동 오버라이드로만
    남겨둔다. #20이 남긴 "B팀 1순위"는 칸별 추정으로 해소됐다.
    """

    realized_basis: str = "cashflow"
    """실현 `R`의 기준. `"cashflow"`: 건별 실제 현금흐름 + GS1M 실제경로 재투자(B팀 명세).

    옛 `"contract"`(정상상환을 계약대로 가정)는 FP 36개월의 66%가 조기상환이라
    36m +1.06%p / 60m +2.24%p 과대추정이었다 — 되살리지 않는다.
    """

    def label(self) -> str:
        """산출물 파일명·컬럼에 남길 가정 표기. 재투자 스위치는 통계표까지 바꾼다(#18).

        ⚠️ `realized_basis`가 들어 있어 **조기상환 보정 전후 산출물이 서로 덮어쓰지 않는다.**
        """
        fee = f"fee{self.servicing_fee_annual * 100:g}pct"
        return f"provisional_{self.reinvest}_{self.treasury_basis}_{fee}_{self.realized_basis}"


# ---------------------------------------------------------------------------
# 국채 곡선
# ---------------------------------------------------------------------------
def load_treasury() -> pd.DataFrame:
    """월별 국채수익률(연율 %). 인덱스는 절대 월 서수(`year*12 + month`)다."""
    path = load_config().paths.data_processed / TREASURY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"국채 데이터가 없습니다: {path}")
    t = pd.read_csv(path, parse_dates=["observation_date"])
    t["month_ord"] = t["observation_date"].dt.year * 12 + t["observation_date"].dt.month
    return t.set_index("month_ord")[["GS3", "GS5"]].sort_index()


def issue_risk_free_rate(
    issue_month_ord: pd.Series, term_months: pd.Series, treasury: pd.DataFrame | None = None
) -> pd.Series:
    """발행시점 × 만기매칭 무위험수익률 `rf` (**유효연율**, 소수).

    `decision_log.md` #18 확정 — 36개월은 `GS3`, 60개월은 `GS5`. 고정 상수가 아니므로
    `config.yaml`의 `risk_free_rate.value`는 읽지 않는다(계속 `null`이다).

    ⚠️ **FRED의 GS3·GS5는 반기복리 bond-equivalent yield다** — `y/100`으로 쓰면 유효연율을
    과소평가하고, 그만큼 `XR = R − rf`가 과대평가된다. `R`은 월별 현금흐름을 굴려 만든
    유효연율이므로 같은 기준으로 맞춘다(B팀 명세 8번과 동일).

        rf = (1 + y/200)² − 1

    실측 영향은 작지만 전 건에 **한 방향으로** 걸린다 — 평균 +0.30bp, 최대 +6.25bp.
    """
    t = treasury if treasury is not None else load_treasury()
    series_by_term = load_config().risk_free_rate.series or {36: "GS3", 60: "GS5"}

    out = pd.Series(np.nan, index=issue_month_ord.index, dtype="float64")
    for term, col in series_by_term.items():
        mask = term_months == term
        if not mask.any():
            continue
        out.loc[mask] = t[col].reindex(issue_month_ord.loc[mask]).to_numpy()
    return np.power(1.0 + out / 200.0, 2.0) - 1.0


def _monthly_rate(annual_rate: pd.Series | np.ndarray) -> np.ndarray:
    """연율 → 월율. `(1+r)^(1/12) − 1`."""
    return np.power(1.0 + np.asarray(annual_rate, dtype="float64"), 1.0 / 12.0) - 1.0


# ---------------------------------------------------------------------------
# 정상상환 — 건별 계약 현금흐름
# ---------------------------------------------------------------------------
def contract_return(
    installment: pd.Series,
    funded_amnt: pd.Series,
    term_months: pd.Series,
    reinvest_rate: pd.Series,
    assumptions: ReturnAssumptions = ReturnAssumptions(),
) -> pd.Series:
    """정상상환 건의 **계약** 실현수익률 `R_계약` (연율).

    계약대로 만기까지 매달 `installment`를 받아 잔존기간 매칭 국채에 재투자한다고 본다.
    매달 같은 금액이므로 미래가치는 연금 종가 공식으로 닫힌 형태가 된다.

        W = installment · Σ_{m=1..T} (1+i)^(T−m) = installment · ((1+i)^T − 1) / i
        R = (W / P)^(12/T) − 1                                   ... #18

    `i`는 월 재투자율이다. `reinvest="cash"`(0% 재투자)면 `W = installment · T`가 되어
    #18이 "이중 부과"라고 지적한 옛 관례가 그대로 재현된다 — 민감도 병기용이다.

    예) `P=10,000`, `installment=332.14`(36개월 12%), 재투자 2%:
        `i=0.00165`, `W = 332.14 × 37.06 = 12,309` → `R = (1.2309)^(1/3) − 1 = **7.17%**`.
        0% 재투자면 `W = 11,957` → `R = **6.14%**`. 차이 **+103bp**가 재투자 가정 효과다
        (전수 실측 평균 +107.5bp와 같은 크기 — `realized_return_sensitivity.py`).

    **정합성 검증**: 대출금리 = 국채금리인 *무위험 등가 대출*을 넣으면 `XR = R − rf`가
    **정확히 0.000bp**로 나온다. #18이 국채 재투자를 택한 근거가 이것이다 — 0% 재투자에서는
    같은 대출이 −0.96%p로 나와 무위험 자산에 벌점이 붙는다.
    """
    P = funded_amnt.to_numpy(dtype="float64")
    A = installment.to_numpy(dtype="float64")
    T = term_months.to_numpy(dtype="float64")

    if assumptions.reinvest == "cash":
        W = A * T
    elif assumptions.reinvest == "treasury":
        i = _monthly_rate(reinvest_rate)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(i == 0, T, (np.power(1.0 + i, T) - 1.0) / i)
        W = A * factor
    else:
        raise ValueError(f"알 수 없는 reinvest 가정: {assumptions.reinvest!r}")

    if assumptions.servicing_fee_annual:
        W = W * np.power(1.0 - assumptions.servicing_fee_annual, T / 12.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.power(W / P, 12.0 / T) - 1.0
    R = np.where(P > 0, R, np.nan)
    return pd.Series(R, index=funded_amnt.index, name="R_contract") + assumptions.prepayment_adjustment


# ---------------------------------------------------------------------------
# 부도 — 건별 실현 현금흐름
# ---------------------------------------------------------------------------
def realized_return_defaulted(
    df: pd.DataFrame,
    reinvest_rate: pd.Series,
    assumptions: ReturnAssumptions = ReturnAssumptions(),
) -> pd.Series:
    """부도(`Charged Off`) 건의 **실현** 수익률 `R` (연율).

    LC 데이터에 월별 납입 내역이 없으므로, 총 수령액을 다음처럼 배분한다
    (`realized_return_sensitivity.py`가 이 가정의 민감도를 잰다 — 대안 배분과의 차이가
    정상/부도 모두 50bp 이내였다).

    - 마지막 납입월 `K`에 `last_pymnt_amnt`를 받고,
    - 나머지 `C_regular − last_pymnt_amnt`를 `1..K−1`에 균등 배분하고,
    - 순회수액 `C_recovery`는 `K + 6`개월에 받는다.

    각 수령액을 만기 `T`까지 국채로 굴려 `W`를 만들고 `R = (W/P)^(12/T) − 1`을 푼다.
    `K + 6 > T`이면 지수가 음수가 되어 **역할인**되는데, 만기 후 수령분이라 맞는 처리다.

    필요 컬럼: `funded_amnt`, `term`, `K`, `total_pymnt`, `recoveries`,
    `collection_recovery_fee`, `last_pymnt_amnt`.
    """
    P = df["funded_amnt"].to_numpy(dtype="float64")
    T = df["term"].to_numpy(dtype="float64")
    K = df["K"].to_numpy(dtype="float64")
    L = df["last_pymnt_amnt"].fillna(0).to_numpy(dtype="float64")
    C_reg = (df["total_pymnt"].fillna(0) - df["recoveries"].fillna(0)).to_numpy(dtype="float64")
    C_rec = (
        df["recoveries"].fillna(0) - df["collection_recovery_fee"].fillna(0)
    ).to_numpy(dtype="float64")

    if assumptions.reinvest == "cash":
        W = C_reg + C_rec
    elif assumptions.reinvest == "treasury":
        i = _monthly_rate(reinvest_rate)
        one = 1.0 + i

        # K>=2: 앞선 K-1개월에 균등배분 A, 마지막 달에 L
        n_pre = np.maximum(K - 1.0, 0.0)
        A = np.where(n_pre > 0, (C_reg - L) / np.where(n_pre > 0, n_pre, 1.0), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Σ_{t=1}^{K-1} (1+i)^(T-t) = (1+i)^(T-K+1) · ((1+i)^(K-1) − 1)/i
            geo = np.where(i == 0, n_pre, (np.power(one, n_pre) - 1.0) / i)
        W_pre = A * np.power(one, T - K + 1.0) * geo
        W_last = L * np.power(one, T - K)

        # K<=1: 전액을 K월에 받은 것으로 본다 (배분할 앞선 달이 없다)
        lump = C_reg * np.power(one, T - np.maximum(K, 0.0))
        W = np.where(K >= 2, W_pre + W_last, lump)

        W = W + C_rec * np.power(one, T - K - RECOVERY_LAG_MONTHS)
    else:
        raise ValueError(f"알 수 없는 reinvest 가정: {assumptions.reinvest!r}")

    if assumptions.servicing_fee_annual:
        W = W * np.power(1.0 - assumptions.servicing_fee_annual, T / 12.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(P > 0, W / P, np.nan)
        R = np.where(ratio >= 0, np.power(np.abs(ratio), 12.0 / T) - 1.0, np.nan)
    return pd.Series(R, index=df.index, name="R_realized")


# ---------------------------------------------------------------------------
# 칸별 통계표 · E[XR] · Var[XR]
# ---------------------------------------------------------------------------
def default_cell_stats(
    xr_defaulted: pd.Series,
    pd_quantile: pd.Series,
    term_months: pd.Series,
    min_count_mean: int = 100,
    min_count_var: int = 300,
) -> pd.DataFrame:
    """칸별(`PD 분위 × term`) 부도 초과수익 통계표 — `mu_부도,d`·`var_부도,d`.

    `var_부도`는 표본이 작은 칸에서 불안정하다(상대오차 ≈ `√(2/(n−1))` — n=300에서 8.2%,
    n=50에서 20.2%). 그래서 **분산에는 평균보다 엄한 최소 표본수를 요구하고, 미달 칸은
    pooled 분산으로 축소추정**한다(`src/analysis/AGENTS.md`). 분산이 과소추정된 칸이
    체계적으로 승인되는 선택편향을 막기 위한 것이다.
    """
    frame = pd.DataFrame(
        {"xr": xr_defaulted, "q": pd_quantile, "term": term_months}
    ).dropna(subset=["xr"])

    g = frame.groupby(["term", "q"], observed=True)["xr"]
    stats = pd.DataFrame({"n": g.size(), "mu": g.mean(), "var": g.var(ddof=1)})

    pooled_var = frame["xr"].var(ddof=1)
    stats["var_raw"] = stats["var"]
    stats["shrunk"] = stats["n"] < min_count_var
    stats.loc[stats["shrunk"], "var"] = pooled_var
    stats["mu_unreliable"] = stats["n"] < min_count_mean
    stats["pooled_var"] = pooled_var
    return stats


def normal_cell_stats(
    xr_contract: pd.Series,
    xr_realized: pd.Series,
    pd_quantile: pd.Series,
    term_months: pd.Series,
    min_count_mean: int = 100,
    min_count_var: int = 300,
) -> pd.DataFrame:
    """칸별 **조기상환 보정** `Δ̄_조기상환,d`와 `var_정상,d` — 정상상환(FP) 건에서만 만든다.

    #20이 "B팀 1순위"로 남긴 미결의 구현이다. 계약대로 만기까지 갚는다고 본 `xr_contract`가
    실제(`xr_realized`)보다 얼마나 높은지를 칸별 평균으로 잡는다.

        Δ̄_d = mean_{i∈d, FP}( XR_계약,i − XR_실현,i )      → 점수용 보정항
        var_정상,d = var_{i∈d, FP}( XR_실현,i )             → Var[XR]의 정상 항

    ## 왜 건별 실현값을 그대로 점수에 쓰지 않는가

    개별 대출이 **몇 개월에 조기상환할지는 승인 시점에 알 수 없다.** `XR_실현,i`를 점수
    `E[XR_i]`에 넣으면 미래를 보고 고르는 셈이라 누수다. 그래서 **건별 계약 현금흐름(A′의
    핵심)은 유지하고, 조기상환은 칸별 평균 보정으로만** 넣는다 — `int_rate` 산포(칸별 sd
    중앙값 2.85%p)는 보존되고 조기상환의 체계적 효과만 차감된다.

    실측 보정폭: 36개월 **+1.06%p**, 60개월 **+2.24%p** (FP 36m의 66.03%가 조기상환).

    `var_정상`은 `default_cell_stats()`와 같은 이유로 표본이 작은 칸에서 pooled로 축소한다.
    보정 전에는 이 값이 **0으로 고정**돼 있어 `q_score` 분모가 부도 항만 반영했다.
    """
    frame = pd.DataFrame({
        "gap": xr_contract - xr_realized,
        "xr": xr_realized,
        "q": pd_quantile,
        "term": term_months,
    }).dropna(subset=["gap", "xr"])

    g = frame.groupby(["term", "q"], observed=True)
    stats = pd.DataFrame({
        "n": g.size(),
        "prepay_adj": g["gap"].mean(),
        "var": g["xr"].var(ddof=1),
    })

    pooled_var = frame["xr"].var(ddof=1)
    pooled_adj = frame["gap"].mean()
    stats["var_raw"] = stats["var"]
    stats["shrunk"] = stats["n"] < min_count_var
    stats.loc[stats["shrunk"], "var"] = pooled_var
    stats.loc[stats["n"] < min_count_mean, "prepay_adj"] = pooled_adj
    stats["pooled_var"] = pooled_var
    stats["pooled_adj"] = pooled_adj
    return stats


def expected_excess_return(
    p_hat: pd.Series,
    xr_normal: pd.Series,
    mu_default: pd.Series,
) -> pd.Series:
    """`E[XR_i] = (1 − p̂)·XR_정상,i + p̂·mu_부도,d(i)` (#20 구조 A′).

    `xr_normal`에는 **조기상환 보정이 이미 반영된 값**을 넣는다
    (`xr_contract − Δ̄_조기상환,d`, `normal_cell_stats()` 참고).
    """
    return (1.0 - p_hat) * xr_normal + p_hat * mu_default


def variance_excess_return(
    p_hat: pd.Series,
    xr_normal: pd.Series,
    mu_default: pd.Series,
    var_default: pd.Series,
    var_normal: pd.Series | float = 0.0,
) -> pd.Series:
    """총분산의 법칙 — **교차항을 빠뜨리지 않는다.**

        Var[XR] = (1−p)·var_정상 + p·var_부도 + p(1−p)·(mu_정상 − mu_부도)²

    마지막 항이 지배적이다. 예시(`p=0.10`, `mu_정상=+5%`/sd 3%, `mu_부도=−40%`/sd 20%)에서
    교차항이 총분산의 **79%** 를 차지한다. 빠뜨리면 sd가 15.2% → 6.9%로 축소되고
    `p(1−p)`에 비례해 편향이 걸려 **랭킹 순서가 바뀐다** (#20).

    ⚠️ 잠정 구현에서 `var_정상 = 0`이다 — 조기상환 보정이 미확정이라 계약 현금흐름만
    쓰기 때문이다. 확정되면 칸별 `실현 R − 계약 R`의 분산을 넣는다.
    """
    gap = xr_normal - mu_default
    return (1.0 - p_hat) * var_normal + p_hat * var_default + p_hat * (1.0 - p_hat) * gap**2


def q_score(expected_xr: pd.Series, variance_xr: pd.Series) -> pd.Series:
    """`q = E[XR] / √Var[XR]` — 승인선 랭킹 기준 후보 중 하나(#5·#20, **미확정**).

    ⚠️ 개별 대출 `q` 최대화는 포트폴리오 Sharpe 최대화와 같은 문제가 아니다.
    어느 기준을 쓰든 threshold는 **Validation 실현 XR로 계산한 실제 Sharpe** 그리드서치로
    정한다(`src/analysis/AGENTS.md`).
    """
    sd = np.sqrt(variance_xr.clip(lower=0))
    return (expected_xr / sd.replace(0, np.nan)).rename("q_score")


def build_excess_returns(
    outcome: pd.DataFrame, assumptions: ReturnAssumptions = ReturnAssumptions()
) -> pd.DataFrame:
    """건별 `rf` · `XR_정상`(계약) · `XR_부도`(실현) · `XR_실현`.

    `XR_정상`은 **부도 건에도 정의된다** — "계약대로 갚았다면 얼마였을까"라서 실현 여부와
    무관하게 계산되며, `E[XR] = (1−p̂)·XR_정상 + p̂·mu_부도`의 첫 항이 바로 그 값이다.

    `xr_realized`는 그와 달리 **실제로 벌어진 결과**다 — **정상·부도 모두 건별 실제 현금흐름**
    으로 계산한다(`realized_return_cashflow.py`, B팀 명세). Sharpe는 기대값이 아니라
    **이 값**으로 계산한다(`src/analysis/AGENTS.md`).

    ✅ **조기상환이 반영된다** (2026-07-31). 이전에는 정상상환 건의 `xr_realized`를 계약
    현금흐름으로 두어 36m **+1.06%p** / 60m **+2.24%p** 과대추정이었다 — FP 36개월의
    **66.03%가 조기상환**이기 때문이다(#20 B팀 1순위 해소).

    반환 열
    -------
    `xr_normal`
        **계약** 기준 정상상환 XR. 점수용이며, 쓸 때는 칸별 `Δ̄_조기상환`을 빼서 쓴다
        (`normal_cell_stats()`). 여기서 빼지 않는 것은 보정항이 **Train에서만** 추정돼야
        하기 때문이다 — 이 함수는 Train/Validation을 모른다.
    `xr_realized`
        정상·부도 모두 **실현** 현금흐름 XR. Sharpe 계산용.
    `xr_default` / `xr_normal_realized`
        `xr_realized`를 부도 / 정상으로 각각 마스킹한 것. 칸별 통계표 산출용.
    """
    from analysis.realized_return_cashflow import (
        build_cashflow_schedule,
        realized_return_actual,
    )

    rf = issue_risk_free_rate(outcome["issue_month_ord"], outcome["term"])

    r_contract = contract_return(
        installment=outcome["installment"],
        funded_amnt=outcome["funded_amnt"],
        term_months=outcome["term"],
        reinvest_rate=rf,
        assumptions=assumptions,
    )
    xr_normal = r_contract - rf

    if assumptions.reinvest == "cash":
        # 민감도 병기용 0% 재투자 — 실현분도 같은 관례로 맞춘다(#18).
        r_realized = realized_return_defaulted(
            outcome, reinvest_rate=rf, assumptions=assumptions
        ).where(outcome["is_default"] == 1, r_contract)
    else:
        schedule = build_cashflow_schedule(outcome)
        r_realized = realized_return_actual(schedule)["R"]

    xr_realized = r_realized - rf
    is_def = outcome["is_default"] == 1

    return pd.DataFrame(
        {
            "rf": rf,
            "xr_normal": xr_normal,
            "xr_default": xr_realized.where(is_def),
            "xr_normal_realized": xr_realized.where(~is_def),
            "xr_realized": xr_realized,
            "term": outcome["term"],
            "int_rate": outcome["int_rate"],
            "is_default": outcome["is_default"],
        }
    )


def build_return_inputs(
    csv_path: Path | None = None, verify_sample: bool = True
) -> pd.DataFrame:
    """수익률 계산에 필요한 **사후 컬럼**을 분석 표본(723,563건)에 맞춰 로드한다.

    반환 프레임은 피처 테이블과 **같은 인덱스**를 갖는다 — `id`로 조인하지 않아도
    `loc`으로 정렬이 맞는다. `K`는 발행 → 최종납입 개월 수다.

    `csv_path`는 train 원본 대신 다른 CSV를 읽을 때만 준다(`loader.second_test_path()`).
    표본 건수 검증(723,563)은 train 기준이므로 `verify_sample=False`를 함께 준다.

    ⚠️ 이 프레임을 모델 입력에 섞지 않는다(`src/preprocessing/AGENTS.md` 누수 방지).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.loader import filter_analysis_sample, load_raw_loans

    cols = [
        "id", "loan_status", "term", "issue_d", "funded_amnt", "installment", "int_rate",
        "total_pymnt", "recoveries", "collection_recovery_fee",
        "last_pymnt_amnt", "last_pymnt_d",
    ]
    raw = load_raw_loans(usecols=cols, csv_path=csv_path)
    s = filter_analysis_sample(raw, verify=verify_sample)

    s["term"] = s["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    s["int_rate"] = pd.to_numeric(
        s["int_rate"].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )
    issue = pd.to_datetime(s["issue_d"], format="%b-%Y", errors="coerce")
    last = pd.to_datetime(s["last_pymnt_d"], format="%b-%Y", errors="coerce")

    s["issue_month_ord"] = issue.dt.year * 12 + issue.dt.month
    s["K"] = ((last.dt.year * 12 + last.dt.month) - s["issue_month_ord"]).clip(lower=0)
    s["is_default"] = (s["loan_status"] == "Charged Off").astype("int8")
    return s


def cash_reinvestment(assumptions: ReturnAssumptions) -> ReturnAssumptions:
    """민감도용 0% 재투자 가정 (#18 병기 확정).

    ⚠️ 스위치 하나로 끝나지 않는다 — `mu`·`var`가 `XR`에서 산출되므로 **칸별 통계표까지
    다시 만든다.** 산출물 파일명·컬럼에 `label()`을 남겨 어느 가정인지 추적한다.
    """
    return replace(assumptions, reinvest="cash")
