from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a GUI window.
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
N = 200
SIGMA_E = 1.0
SIGMA_I_VALUES = (2.0, 3.0)
DT = 0.02
T_FINAL = 40.0
N_TRIALS = 20
INITIAL_SD = 0.01
SEED = 4370


def ring_kernel(n: int, sigma_e: float, sigma_i: float) -> np.ndarray:
    """K[j] uses the shortest periodic distance from site 0 to site j."""
    j = np.arange(n)
    distance = np.minimum(j, n - j)
    return np.exp(-distance**2 / (2.0 * sigma_e**2)) - (
        sigma_e / sigma_i
    ) * np.exp(-distance**2 / (2.0 * sigma_i**2))


def continuum_prediction(sigma_e: float, sigma_i: float) -> tuple[float, float]:
    """Return k_* and its full right/left-pair period, 2*pi/k_*."""
    k_star = np.sqrt(
        4.0 * np.log(sigma_i / sigma_e) / (sigma_i**2 - sigma_e**2)
    )
    return float(k_star), float(2.0 * np.pi / k_star)


def simulate(initial: np.ndarray, kernel_fft: np.ndarray) -> np.ndarray:
    """Forward Euler for dw/dt = K*w, with clipping after every step.

    Rows of ``initial`` are independent ring states. NumPy's FFT uses the
    unnormalized forward transform and normalized inverse transform, so this
    product gives the circular convolution without an extra factor of N.
    """
    weights = initial.copy()
    n_steps = round(T_FINAL / DT)
    if not np.isclose(n_steps * DT, T_FINAL):
        raise ValueError("T_FINAL must be an integer multiple of DT")

    for _ in range(n_steps):
        convolution = np.fft.irfft(
            np.fft.rfft(weights, axis=1) * kernel_fft[None, :],
            n=weights.shape[1],
            axis=1,
        )
        weights = np.clip(weights + DT * convolution, -1.0, 1.0)
    return weights


def count_stripe_pairs(weights: np.ndarray) -> int:
    """One right/left pair equals two sign transitions around the ring."""
    eye_sign = np.where(weights >= 0.0, 1, -1)
    transitions = int(np.count_nonzero(eye_sign != np.roll(eye_sign, 1)))
    if transitions == 0 or transitions % 2:
        raise ValueError(f"Invalid number of ring sign transitions: {transitions}")
    return transitions // 2


def plot_patterns(results: dict[float, dict], output_path: Path) -> None:
    """Show the first trial's final eye-preference pattern for each sigma_i."""
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.6), sharex=True)
    sites = np.arange(N + 1)

    for ax, sigma_i in zip(axes, SIGMA_I_VALUES):
        result = results[sigma_i]
        final = result["final_states"][0]
        closed_ring = np.r_[final, final[0]]  # Show the link from site 199 to 0.
        ax.fill_between(
            sites, 0.0, closed_ring, where=closed_ring >= 0.0,
            color="#d95f02", alpha=0.45, label="Right-eye preference",
        )
        ax.fill_between(
            sites, 0.0, closed_ring, where=closed_ring < 0.0,
            color="#1f78b4", alpha=0.45, label="Left-eye preference",
        )
        ax.plot(sites, closed_ring, color="#222222", lw=0.9)
        ax.axhline(0.0, color="#555555", lw=0.7)
        ax.set_ylim(-1.08, 1.08)
        ax.set_title(
            rf"$\sigma_e=1$, $\sigma_i={sigma_i:g}$; "
            rf"trial 1: {result['pair_counts'][0]} right/left pairs, "
            rf"period = {result['periods'][0]:.3f} sites",
            fontsize=11,
        )
        ax.grid(alpha=0.18)

    axes[-1].set_xlabel("Cortical site a (site spacings; periodic ring)")
    axes[-1].set_xlim(0, N)
    fig.supylabel(r"Eye preference $w_-(a)$ (dimensionless)", x=0.015)
    fig.suptitle(
        rf"Ocular-dominance patterns: $N={N}$, $dt={DT:g}$, "
        rf"$t={T_FINAL:g}\,\tau_w/(q_S-q_D)$",
        fontsize=13,
    )
    fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=True, fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.82, hspace=0.42)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_period_comparison(results: dict[float, dict], output_path: Path) -> None:
    """Show all 20 measured periods with continuum and ring-mode benchmarks."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    plot_rng = np.random.default_rng(SEED + 1)  # Reproducible visual jitter only.
    sigma_values = np.array(SIGMA_I_VALUES)

    for sigma_i in SIGMA_I_VALUES:
        periods = results[sigma_i]["periods"]
        x_jitter = plot_rng.uniform(-0.065, 0.065, size=N_TRIALS)
        ax.scatter(
            sigma_i + x_jitter, periods, color="#777777", alpha=0.62,
            s=28, zorder=2, label="Individual trials" if sigma_i == SIGMA_I_VALUES[0] else None,
        )

    means = [results[s]["mean_period"] for s in SIGMA_I_VALUES]
    sds = [results[s]["sd_period"] for s in SIGMA_I_VALUES]
    continuum = [results[s]["continuum_period"] for s in SIGMA_I_VALUES]
    ring = [results[s]["ring_period"] for s in SIGMA_I_VALUES]
    ax.errorbar(
        sigma_values, means, yerr=sds, fmt="D", capsize=5,
        color="#111111", lw=1.5, ms=7, zorder=5,
        label="Measured mean +/- sample SD",
    )
    ax.plot(
        sigma_values, continuum, "o--", color="#8e44ad", lw=1.6, ms=7,
        label=r"Continuum prediction $2\pi/k_*$",
    )
    ax.plot(
        sigma_values, ring, "s-.", color="#008b68", lw=1.6, ms=7,
        label=r"Fastest ring Fourier mode $N/m_*$",
    )
    ax.set_xticks(sigma_values)
    ax.set_xlim(1.65, 3.35)
    ax.set_xlabel(r"Inhibitory width $\sigma_i$ (site spacings; $\sigma_e=1$)")
    ax.set_ylabel("Stripe period (site spacings per right/left pair)")
    ax.set_title(
        rf"20 initial conditions per width; $N={N}$, "
        rf"$dt={DT:g}$, $t={T_FINAL:g}\,\tau_w/(q_S-q_D)$"
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_report(results: dict[float, dict]) -> str:
    lines = [
        "Problem 1(c): Ocular-dominance stripes",
        f"N={N}, dt={DT:g}, t={T_FINAL:g}, 20 initial conditions per sigma_i; periods are in site spacings.",
        "sigma_i  measured period (mean +/- SD)  continuum  fastest ring mode",
    ]
    for sigma_i in SIGMA_I_VALUES:
        r = results[sigma_i]
        lines.append(
            f"{sigma_i:.1f}      {r['mean_period']:.4g} +/- {r['sd_period']:.4g}"
            f"             {r['continuum_period']:.4g}      {r['ring_period']:.4g} (m={r['ring_mode']})"
        )
    lines.append("Increasing inhibitory width increases stripe period; measured values are close to both predictions.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Reuse identical initial patterns for sigma_i=2 and 3 for a fair comparison.
    initial = rng.normal(0.0, INITIAL_SD, size=(N_TRIALS, N))
    results: dict[float, dict] = {}
    trial_rows: list[dict] = []

    for sigma_i in SIGMA_I_VALUES:
        kernel = ring_kernel(N, SIGMA_E, sigma_i)
        kernel_fft = np.fft.rfft(kernel).real
        final_states = simulate(initial, kernel_fft)
        pair_counts = np.array([count_stripe_pairs(row) for row in final_states])
        periods = N / pair_counts.astype(float)
        k_cont, continuum_period = continuum_prediction(SIGMA_E, sigma_i)
        ring_mode = int(np.argmax(kernel_fft[1:]) + 1)
        ring_period = N / ring_mode

        for trial_index, (pairs, period, final) in enumerate(
            zip(pair_counts, periods, final_states, strict=True), start=1
        ):
            dominant_mode = int(np.argmax(np.abs(np.fft.rfft(final))[1:]) + 1)
            trial_rows.append({
                "sigma_i": sigma_i,
                "trial": trial_index,
                "right_left_pairs": int(pairs),
                "period_sites": float(period),
                "dominant_final_fft_mode": dominant_mode,
            })

        mean_period = float(np.mean(periods))
        sd_period = float(np.std(periods, ddof=1))
        results[sigma_i] = {
            "final_states": final_states,
            "pair_counts": pair_counts,
            "periods": periods,
            "k_cont": k_cont,
            "continuum_period": continuum_period,
            "ring_mode": ring_mode,
            "ring_period": ring_period,
            "mean_period": mean_period,
            "sd_period": sd_period,
            "sem_period": sd_period / np.sqrt(N_TRIALS),
            "min_period": float(np.min(periods)),
            "max_period": float(np.max(periods)),
            "error_cont": abs(mean_period - continuum_period),
            "error_ring": abs(mean_period - ring_period),
            "relative_error_cont": 100.0 * abs(mean_period - continuum_period) / continuum_period,
            "relative_error_ring": 100.0 * abs(mean_period - ring_period) / ring_period,
            "zero_mode_gain": float(kernel_fft[0]),
            "mean_clipped_fraction": float(np.mean(np.abs(final_states) >= 1.0 - 1e-12)),
        }

    plot_patterns(results, OUTPUT_DIR / "hw4_q1c_patterns.png")
    plot_period_comparison(results, OUTPUT_DIR / "hw4_q1c_period_comparison.png")

    with (OUTPUT_DIR / "hw4_q1c_trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    report = make_report(results)
    (OUTPUT_DIR / "hw4_q1c_answer.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    print("Saved files:")
    for name in (
        "hw4_q1c_patterns.png",
        "hw4_q1c_period_comparison.png",
        "hw4_q1c_trials.csv",
        "hw4_q1c_answer.txt",
    ):
        print(f"  {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
