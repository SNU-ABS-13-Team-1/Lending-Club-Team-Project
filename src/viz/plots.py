"""최종 결과 리포트용 차트 생성 — outputs/figures/에 저장.

규칙은 `src/viz/AGENTS.md`. 핵심: 차트는 outputs/의 산출 CSV만 읽는다(재계산 금지),
그림마다 출처 분할 체계를 파일명·각주에 표기한다, oracle_tau는 그리지 않는다.

실행: `python src/viz/plots.py` — 전량 멱등 재생성.

| 그림 | 소스 CSV |
| --- | --- |
| fig_k50_delta_8_2_3fold.png | sharpe_repeat_k50_8_2_3fold.csv |
| fig_k50_tau_8_2_3fold.png | sharpe_repeat_k50_8_2_3fold.csv |
| fig_2ndtest_benchmark_8_2.png | second_test_evaluation_8_2.csv |
| fig_generalization_8_2_3fold.png | 위 2종 결합 |
| fig_cell_means_6_2_2_oof.png | oof_default_cell_stats_*.csv · oof_c1_cell_means_*.csv (6:2:2 진단) |
| fig_calibration_ece_6_2_2_oof.png | oof_c4_calibration.csv (6:2:2 진단) |
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

# ---- 팔레트 (src/viz/AGENTS.md — 검증된 기본 팔레트, light) ----
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQ_LIGHT = "#86b6ef"
GOOD = "#006300"

ASSUMPTION_CAPTION = "가정: 국채 ⓒ발행시점 고정 · 수수료 0% · 조기상환 현금흐름 반영 (#22) · 등가중 (#23 B)"

mpl.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
})

REINVEST_LABEL = {"treasury": "국채 재투자 (헤드라인)", "cash": "0% 재투자 (민감도)"}
CRITERION_ORDER = ["E[XR]", "pd (보정 전)", "q_score"]  # 아래→위 표시 순서


def _caption(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, fontsize=7, color=MUTED, ha="left")


def _load_k50() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "sharpe_repeat_k50_8_2_3fold.csv")


def plot_k50_delta() -> Path:
    """K=50 Δ Sharpe 분포 — 랭킹 기준 3종, 재투자 가정 2패널. q_score만 강조(파랑)."""
    df = _load_k50()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharex=True)
    rng = np.random.default_rng(0)
    for ax, reinvest in zip(axes, ["treasury", "cash"]):
        sub = df[df["reinvest"] == reinvest]
        for i, crit in enumerate(CRITERION_ORDER):
            vals = sub.loc[sub["criterion"] == crit, "delta_sharpe"].to_numpy()
            color = BLUE if crit == "q_score" else MUTED
            y = i + rng.normal(0, 0.055, len(vals))
            ax.scatter(vals, y, s=14, color=color, alpha=0.65, linewidths=0, zorder=3)
            med = float(np.median(vals))
            ax.plot([med, med], [i - 0.22, i + 0.22], color=INK, lw=1.6, zorder=4)
            ax.annotate(f"{med:+.4f}", (med, i + 0.28), ha="center", fontsize=7.5,
                        color=INK2, zorder=5)
        ax.set_yticks(range(len(CRITERION_ORDER)))
        ax.set_yticklabels(CRITERION_ORDER)
        ax.set_ylim(-0.55, len(CRITERION_ORDER) - 0.25)
        ax.set_title(REINVEST_LABEL[reinvest])
        ax.set_xlabel("Δ Sharpe (모형 − approve-all)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", colors=INK2)
    fig.suptitle("K=50 재분할에서 랭킹 기준 3종의 Δ Sharpe — q_score 우세 (#21 ①)", x=0.01, ha="left")
    _caption(fig, f"세로선=중앙값 · 점=재분할 seed 1개 · 8:2+OOF 3-fold · {ASSUMPTION_CAPTION}")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = FIGURES / "fig_k50_delta_8_2_3fold.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_k50_tau() -> Path:
    """q_score·국채 재투자의 τ* 분포 (K=50) + 중앙값·승자 seed 26 τ*."""
    df = _load_k50()
    sub = df[(df["reinvest"] == "treasury") & (df["criterion"] == "q_score")]
    taus = sub["threshold"].to_numpy()
    median_tau = float(np.median(taus))
    winner = sub.loc[sub["sharpe"].idxmax()]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hist(taus, bins=14, color=SEQ_LIGHT, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.axvline(median_tau, color=INK2, lw=1.4, ls=(0, (4, 2)), zorder=4)
    ax.axvline(float(winner["threshold"]), color=BLUE, lw=2.0, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"중앙값 τ = {median_tau:.4f}", (median_tau, ymax * 0.97),
                ha="right", va="top", fontsize=8, color=INK2, xytext=(-5, 0),
                textcoords="offset points")
    ax.annotate(f"승자 seed {int(winner['seed'])}\nτ* = {winner['threshold']:.4f}",
                (float(winner["threshold"]), ymax * 0.78), ha="left", va="top",
                fontsize=8, color=INK, xytext=(6, 0), textcoords="offset points")
    ax.set_xlabel("τ* (q_score 승인선, Validation Sharpe 최대점)")
    ax.set_ylabel("seed 수")
    ax.grid(axis="x", visible=False)
    ax.set_title("K=50 재분할의 τ* 분포 — 승자 τ*가 중앙값 곁에 있다 (#21 ④)")
    _caption(fig, f"국채 재투자 · q_score · 8:2+OOF 3-fold · {ASSUMPTION_CAPTION}")
    fig.tight_layout()
    out = FIGURES / "fig_k50_tau_8_2_3fold.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_second_test_benchmark() -> Path:
    """2nd Test: 모형(τ* 고정) vs approve-all — 절대 Sharpe 병기 + Δ 헤드라인."""
    df = pd.read_csv(OUTPUTS / "second_test_evaluation_8_2.csv")
    rows = df[df["variant"] == "winner_tau"].set_index("reinvest")
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(2)
    w = 0.32
    order = ["treasury", "cash"]
    model = [float(rows.loc[r, "sharpe"]) for r in order]
    base = [float(rows.loc[r, "sharpe_approve_all"]) for r in order]
    delta = [float(rows.loc[r, "delta_sharpe"]) for r in order]
    b1 = ax.bar(x - w / 2 - 0.01, model, width=w, color=BLUE, zorder=3,
                label="모형 (q_score · τ* 고정)")
    b2 = ax.bar(x + w / 2 + 0.01, base, width=w, color=MUTED, zorder=3,
                label="approve-all (대조군)")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.4f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
    for xi, d in zip(x, delta):
        ax.annotate(f"Δ {d:+.4f}", (xi, max(model[xi], base[xi])),
                    xytext=(0, 16), textcoords="offset points",
                    ha="center", fontsize=9.5, color=GOOD, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([REINVEST_LABEL[r] for r in order], color=INK2)
    ax.set_ylabel("Sharpe (절대값 — 헤드라인은 Δ)")
    ax.set_ylim(0, max(model) * 1.32)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("2nd Test (외부 481,833건): 모형 vs approve-all")
    _caption(fig, f"승자 seed 26 · τ*=0.1895 고정 적용(재탐색 없음, #23) · {ASSUMPTION_CAPTION}")
    fig.tight_layout()
    out = FIGURES / "fig_2ndtest_benchmark_8_2.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_generalization() -> Path:
    """K=50 Validation Δ 분포 위에 승자 Validation Δ와 2nd Test Δ를 겹쳐 일반화를 보인다."""
    k50 = _load_k50()
    sub = k50[(k50["reinvest"] == "treasury") & (k50["criterion"] == "q_score")]
    vals = sub["delta_sharpe"].to_numpy()
    winner = sub.loc[sub["sharpe"].idxmax()]
    st = pd.read_csv(OUTPUTS / "second_test_evaluation_8_2.csv")
    st_delta = float(st[(st["variant"] == "winner_tau")
                        & (st["reinvest"] == "treasury")]["delta_sharpe"].iloc[0])
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.scatter(vals, rng.normal(0, 0.05, len(vals)), s=16, color=MUTED, alpha=0.6,
               linewidths=0, zorder=3, label="Validation Δ (K=50 재분할)")
    mean = float(vals.mean())
    ax.plot([mean, mean], [-0.18, 0.18], color=INK, lw=1.6, zorder=4)
    ax.annotate(f"K=50 평균 {mean:+.4f}", (mean, 0.22), ha="center", fontsize=8, color=INK2)
    ax.scatter([float(winner["delta_sharpe"])], [0], s=70, facecolors="none",
               edgecolors=BLUE, linewidths=1.8, zorder=5, label="승자 seed 26 (Validation)")
    ax.annotate(f"승자 (Val) {float(winner['delta_sharpe']):+.4f}",
                (float(winner["delta_sharpe"]), -0.28), ha="center", fontsize=8, color=INK2)
    ax.scatter([st_delta], [0], s=90, marker="D", color=BLUE, zorder=6,
               edgecolors=SURFACE, linewidths=1.2, label="2nd Test (τ* 고정)")
    ax.annotate(f"2nd Test {st_delta:+.4f}", (st_delta, 0.34), ha="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_ylim(-0.55, 0.62)
    ax.set_yticks([])
    ax.set_xlabel("Δ Sharpe (모형 − approve-all)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=1,
              bbox_to_anchor=(0.0, 1.02))
    ax.set_title("일반화: 2nd Test Δ가 K=50 Validation 분포 안에 든다")
    _caption(fig, f"국채 재투자 · q_score · 8:2+OOF 3-fold · {ASSUMPTION_CAPTION}")
    fig.tight_layout()
    out = FIGURES / "fig_generalization_8_2_3fold.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_cell_means() -> Path:
    """구조 A′(#20)의 칸 구조: 부도분은 PD분위×term 그룹 평균, 정상분과 결합해 E[XR].

    출처: 6:2:2 OOF 진단 산출물(#20·#21 근거 시점) — 8:2 산출물이 아니다(각주 표기).
    """
    label = "provisional_treasury_issue_fixed_fee0pct"
    default_stats = pd.read_csv(OUTPUTS / f"oof_default_cell_stats_{label}.csv")
    cell_means = pd.read_csv(OUTPUTS / f"oof_c1_cell_means_{label}.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    term_color = {36.0: BLUE, 60.0: ORANGE}
    for ax, df, col, title in (
        (ax1, default_stats, "mu", "부도 칸 평균 r̄_부도 (그룹 평균)"),
        (ax2, cell_means, "E_XR", "결합 E[XR] = (1−p̂)·XR_계약 + p̂·r̄_부도"),
    ):
        for term, color in term_color.items():
            sub = df[df["term"] == term].sort_values("q")
            ax.plot(sub["q"], sub[col], color=color, lw=2.0, marker="o", ms=4.5, zorder=3)
            ax.annotate(f"{int(term)}개월", (float(sub["q"].iloc[-1]), float(sub[col].iloc[-1])),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("PD 분위 (보정 전 PD, term별 10분위)")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    ax1.set_ylabel("수익률 (연율)")
    fig.suptitle("구조 A′의 칸 구조 — 분위가 오를수록 부도 손실은 깊어지고 E[XR]는 얇아진다 (#20)",
                 x=0.01, ha="left")
    _caption(fig, "출처: 6:2:2 OOF 진단 산출물(oof_diagnostics.py, #20·#21 근거) — 8:2 본실행 산출물이 아님 · "
                  + ASSUMPTION_CAPTION)
    fig.tight_layout(rect=(0, 0, 0.97, 0.99))
    out = FIGURES / "fig_cell_means_6_2_2_oof.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_calibration_ece() -> Path:
    """isotonic 확률보정(#21 ②)의 효과 — Validation ECE 보정 전/후.

    in-sample 재보정 행(train_oof_calibrated_INSAMPLE)은 진단 전용이라 그리지 않는다.
    출처: 6:2:2 OOF 진단 산출물(진단 C-4).
    """
    df = pd.read_csv(OUTPUTS / "oof_c4_calibration.csv").set_index("stage")
    stages = [("validation_raw", "보정 전", SEQ_LIGHT), ("validation_calibrated", "보정 후", BLUE)]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (stage, name, color) in enumerate(stages):
        v = float(df.loc[stage, "ece_pp"])
        ax.bar(i, v, width=0.34, color=color, zorder=3)
        ax.annotate(f"{v:.3f}%p", (i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([name for _, name, _ in stages], color=INK2)
    ax.set_ylabel("ECE (%p) — Validation")
    ax.grid(axis="x", visible=False)
    ax.set_title("isotonic 보정으로 Validation ECE 감소 (#21 ②)")
    _caption(fig, "출처: 6:2:2 OOF 진단(진단 C-4) · E[XR]의 p̂만 보정 후 PD를 쓴다(역할 분리)")
    fig.tight_layout()
    out = FIGURES / "fig_calibration_ece_6_2_2_oof.png"
    fig.savefig(out)
    plt.close(fig)
    return out


ALL_PLOTS = [
    plot_k50_delta,
    plot_k50_tau,
    plot_second_test_benchmark,
    plot_generalization,
    plot_cell_means,
    plot_calibration_ece,
]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in ALL_PLOTS:
        out = fn()
        print(f"저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
