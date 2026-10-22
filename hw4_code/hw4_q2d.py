from __future__ import annotations

import csv
from math import erf, exp, log, sqrt, tanh
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
MU = 0.2
SIGMA = 0.5
I0 = 5.0
LAMBDAS = (0.0, 0.2)
THRESHOLDS = np.arange(0.5, 4.0 + 0.01, 0.5)
DT = 0.002
N_TRIALS = 10_000
MAX_TIME = 200.0
SEED = 43705370
DX_THEORY = 0.001


def cumulative_trapezoid(values: np.ndarray, dx: float) -> np.ndarray:
    """Cumulative integral on an equally spaced grid, with value zero at left."""
    return np.r_[0.0, np.cumsum((values[:-1] + values[1:]) * (dx / 2.0))]


def theory(lam: float, theta: float) -> tuple[float, float]:
    """Return P(correct) and E[first-passage time] from the backward equations.

    At lam=0 use part (b). At lam>0 use part (c)'s integrating factor g
    for accuracy. For time, solve sigma^2*T''+(lam*x+mu)*T'=-1 with T(+-theta)=0.
    Writing G(x)=int_{-theta}^x g and J(x)=int_{-theta}^x 1/g gives
    T'(x)=g(x)[C-J(x)/sigma^2]. The two boundary conditions determine C.
    """
    if lam == 0.0:
        z = MU * theta / SIGMA**2
        return 1.0 / (1.0 + exp(-z)), theta / MU * tanh(z / 2.0)

    n_intervals = round(2.0 * theta / DX_THEORY)
    if n_intervals % 2:
        n_intervals += 1  # Ensure x=0 is an exact grid point.
    x = np.linspace(-theta, theta, n_intervals + 1)
    dx = x[1] - x[0]
    g = np.exp(-(lam * x*x + 2.0 * MU * x) / (2.0 * SIGMA**2))
    G = cumulative_trapezoid(g, dx)
    J = cumulative_trapezoid(1.0 / g, dx)
    H = cumulative_trapezoid(g * J, dx)
    mid = n_intervals // 2
    accuracy = G[mid] / G[-1]
    mean_time = (H[-1] * G[mid] / G[-1] - H[mid]) / SIGMA**2
    return float(accuracy), float(mean_time)


def simulate(lam: float, seed: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Euler-Maruyama on both original rate equations, sharing paths across theta.

    A path continues after crossing a small threshold so its first crossing
    of every larger threshold is also recorded. A linear within-step crossing
    time reduces grid timing error; the underlying path is still Euler-Maruyama.
    """
    rng = np.random.default_rng(seed)
    ws = wc = (1.0 + lam) / 2.0
    ids = np.arange(N_TRIALS)
    r1 = np.full(N_TRIALS, I0)
    r2 = np.full(N_TRIALS, I0)
    next_unhit = np.zeros(N_TRIALS, dtype=np.int8)
    hit_time = np.full((N_TRIALS, len(THRESHOLDS)), np.nan)
    correct = np.zeros((N_TRIALS, len(THRESHOLDS)), dtype=np.int8)
    smallest_rate = I0
    noise_scale = SIGMA * sqrt(DT)

    for step in range(1, round(MAX_TIME / DT) + 1):
        if ids.size == 0:
            return hit_time, correct, smallest_rate, step - 1

        old_d = r1 - r2
        z1 = rng.standard_normal(ids.size)
        z2 = rng.standard_normal(ids.size)
        new_r1 = r1 + DT * ((ws - 1.0) * r1 - wc * r2 + I0 + MU / 2.0) + noise_scale * z1
        new_r2 = r2 + DT * ((ws - 1.0) * r2 - wc * r1 + I0 - MU / 2.0) + noise_scale * z2
        smallest_rate = min(smallest_rate, float(new_r1.min()), float(new_r2.min()))
        new_d = new_r1 - new_r2

        # Number of thresholds reached by the new absolute difference.
        reached = np.searchsorted(THRESHOLDS, np.abs(new_d), side="right")
        if np.any(reached > next_unhit):
            for j, theta in enumerate(THRESHOLDS):
                crossed = (next_unhit <= j) & (reached > j)
                if not np.any(crossed):
                    continue
                sign = np.where(new_d[crossed] > 0.0, 1.0, -1.0)
                fraction = (sign * theta - old_d[crossed]) / (new_d[crossed] - old_d[crossed])
                fraction = np.clip(fraction, 0.0, 1.0)
                hit_time[ids[crossed], j] = (step - 1 + fraction) * DT
                correct[ids[crossed], j] = (sign > 0.0).astype(np.int8)

        next_unhit = np.maximum(next_unhit, reached)
        keep = next_unhit < len(THRESHOLDS)
        ids = ids[keep]
        r1 = new_r1[keep]
        r2 = new_r2[keep]
        next_unhit = next_unhit[keep]

    raise RuntimeError(f"{ids.size} paths did not reach theta=4 by t={MAX_TIME:g}")


def summarize(hit_time: np.ndarray, correct: np.ndarray, lam: float) -> list[dict]:
    """Per-threshold Monte Carlo estimates, uncertainty, and theory errors."""
    if not np.all(np.isfinite(hit_time)):
        raise RuntimeError("At least one first-passage time was not recorded")
    rows = []
    for j, theta in enumerate(THRESHOLDS):
        times = hit_time[:, j]
        p_mc = float(correct[:, j].mean())
        t_mc = float(times.mean())
        p_se = sqrt(p_mc * (1.0 - p_mc) / N_TRIALS)
        t_se = float(times.std(ddof=1) / sqrt(N_TRIALS))
        p_th, t_th = theory(lam, float(theta))
        rows.append(dict(
            lam=lam, theta=float(theta), n=N_TRIALS,
            p_mc=p_mc, p_se=p_se, p_theory=p_th, p_error=p_mc-p_th,
            t_mc=t_mc, t_se=t_se, t_theory=t_th, t_error=t_mc-t_th,
        ))
    return rows


def plot_results(rows: list[dict], matched_theta: float, matched_p: float,
                 matched_t: float, output: Path) -> None:
    """Two-panel, fully labeled, 300-dpi theory-versus-simulation figure."""
    fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True)
    colors = {0.0: "#176b87", 0.2: "#b4492d"}
    for lam in LAMBDAS:
        group = [r for r in rows if r["lam"] == lam]
        theta = np.array([r["theta"] for r in group])
        for ax, key, se_key, theory_key in (
            (ax_p, "p_mc", "p_se", "p_theory"),
            (ax_t, "t_mc", "t_se", "t_theory"),
        ):
            ax.plot(theta, [r[theory_key] for r in group], "-", color=colors[lam],
                    lw=2.0, label=rf"Theory, $\lambda={lam:g}$")
            ax.errorbar(theta, [r[key] for r in group],
                        yerr=[1.96 * r[se_key] for r in group], fmt="o",
                        color=colors[lam], mfc="white", ms=5.5, capsize=2.5,
                        label=rf"Simulation, $\lambda={lam:g}$ (95% CI)")

    ceiling = 0.5 * (1.0 + erf(MU / (SIGMA * sqrt(0.2)) / sqrt(2.0)))
    ax_p.axhline(ceiling, ls=":", color="#555555", lw=1.4,
                 label=rf"$\lambda=0.2$ ceiling = {ceiling:.3f}")
    ax_p.scatter([matched_theta], [matched_p], marker="*", s=130, color="black",
                 zorder=6, label="Theory-matched perfect integrator")
    ax_t.scatter([matched_theta], [matched_t], marker="*", s=130, color="black",
                 zorder=6, label="Theory-matched perfect integrator")

    ax_p.set_ylabel("P(correct) (dimensionless)")
    ax_t.set_ylabel("Mean decision time (rate time constants)")
    ax_p.set_ylim(0.48, 1.01)
    ax_t.set_ylim(bottom=0.0)
    for ax in (ax_p, ax_t):
        ax.set_xlabel(r"Decision threshold $\theta$ (rate units)")
        ax.set_xticks(THRESHOLDS)
        ax.set_xlim(0.35, 4.15)
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8.1)
    ax_p.set_title("Choice accuracy")
    ax_t.set_title("Time to either decision")
    fig.suptitle(
        rf"Competitive circuit: $\mu={MU}$, $\sigma={SIGMA}$, $I_0={I0:g}$, "
        rf"$dt={DT}$; {N_TRIALS:,} trials per threshold",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_report(rows: list[dict], diagnostics: dict, matched: dict) -> str:
    r = next(r for r in rows if r["lam"] == 0.2 and r["theta"] == 4.0)
    lines = [
        "Problem 2(d): Competitive circuit and drift-diffusion",
        "The figure compares simulated and theoretical accuracy and decision time across thresholds for lambda=0 and 0.2.",
        f"At lambda=0.2, theta=4: P(correct)={r['p_theory']:.4g} (theory), {r['p_mc']:.4g} +/- {1.96*r['p_se']:.4g} (simulation);",
        f"mean time={r['t_theory']:.4g} (theory), {r['t_mc']:.4g} +/- {1.96*r['t_se']:.4g} (simulation).",
        f"A perfect integrator with the same theoretical accuracy uses theta={matched['theta']:.4g} and has mean time {matched['t']:.4g};",
        f"the unstable circuit takes {r['t_theory']/matched['t']:.4g} times as long.",
        "Accuracy rises with threshold for lambda=0; for lambda=0.2 it saturates near 0.814, while decision time increases.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    all_rows: list[dict] = []
    diagnostics = {}
    for j, lam in enumerate(LAMBDAS):
        hit_time, correct, min_rate, steps = simulate(lam, SEED + j)
        all_rows.extend(summarize(hit_time, correct, lam))
        diagnostics[lam] = dict(min_rate=min_rate, steps=steps)

    target = next(r for r in all_rows if r["lam"] == 0.2 and r["theta"] == 4.0)
    target_p = target["p_theory"]
    theta_match = SIGMA**2 / MU * log(target_p / (1.0 - target_p))
    matched_p, matched_t = theory(0.0, theta_match)
    theta_emp = SIGMA**2 / MU * log(target["p_mc"] / (1.0 - target["p_mc"]))
    _, t_emp = theory(0.0, theta_emp)
    ceiling = 0.5 * (1.0 + erf(MU / (SIGMA * sqrt(0.2)) / sqrt(2.0)))
    matched = dict(theta=theta_match, p=matched_p, t=matched_t,
                   theta_emp=theta_emp, t_emp=t_emp, ceiling=ceiling)

    figure = OUT / "hw4_q2d_accuracy_time.png"
    plot_results(all_rows, theta_match, matched_p, matched_t, figure)

    csv_path = OUT / "hw4_q2d_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    report = make_report(all_rows, diagnostics, matched)
    print(report, end="")
    (OUT / "hw4_q2d_report.txt").write_text(report, encoding="utf-8")
    print(f"Saved figure: {figure}")
    print(f"Saved data:   {csv_path}")
    print(f"Saved report: {OUT / 'hw4_q2d_report.txt'}")


if __name__ == "__main__":
    main()
