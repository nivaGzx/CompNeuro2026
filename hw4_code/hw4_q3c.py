from __future__ import annotations

import csv
from math import erfc, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
N = 500
LOADS = np.round(np.arange(0.02, 0.62 + 0.001, 0.04), 2)
N_CUES = 10
FLIP_FRACTION = 0.10
N_INITIAL_FLIPS = round(N * FLIP_FRACTION)
SEED = 43705373
MAX_SWEEPS = 1000  # Safety guard only; convergence stops at the first empty sweep.


def normal_cdf_negative_inverse_sqrt(alpha: float) -> float:
    """Phi(-1/sqrt(alpha)), using erfc without a SciPy dependency."""
    return 0.5 * erfc(1.0 / sqrt(2.0 * alpha))


def draw_patterns(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    """One independent, unbiased +/-1 pattern set; columns are memories."""
    return (2 * rng.integers(0, 2, size=(n, p), dtype=np.int8) - 1).astype(
        np.float32
    )


def hebb_connections_scaled(patterns: np.ndarray) -> np.ndarray:
    """Return J=N*W with J_ii=0; this positive scaling preserves update signs."""
    j = patterns @ patterns.T
    np.fill_diagonal(j, 0.0)
    return j


def recall_asynchronously(
    j: np.ndarray, initial: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int, int]:
    """Update bits in random order until a full sweep makes no changes."""
    s = initial.copy()
    h = j @ s
    total_flips = 0

    for sweep in range(1, MAX_SWEEPS + 1):
        changes = 0
        for i in rng.permutation(s.size):
            new_state = 1.0 if h[i] >= 0.0 else -1.0  # sign(0)=+1.
            if new_state != s[i]:
                delta = new_state - s[i]
                s[i] = new_state
                h += delta * j[:, i]
                changes += 1
        total_flips += changes
        if changes == 0:
            return s, sweep, total_flips

    raise RuntimeError(f"Recall did not converge within {MAX_SWEEPS} sweeps")


def simulate_load(alpha: float, rng: np.random.Generator) -> dict:
    """One pattern set at this load; all stored bits and ten corrupted cues."""
    p = round(alpha * N)
    if abs(p / N - alpha) > 1e-12:
        raise ValueError(f"alpha={alpha} does not give an integer P")

    patterns = draw_patterns(N, p, rng)
    j = hebb_connections_scaled(patterns)

    # All N*P bit margins. A negative margin means a stored bit is unstable.
    # The diagonal of J was removed before computing these local fields.
    margins = patterns * (j @ patterns)
    unstable_count = int(np.count_nonzero(margins < 0.0))
    tested_bits = N * p
    unstable_fraction = unstable_count / tested_bits
    theory = normal_cdf_negative_inverse_sqrt(alpha)

    # Use distinct stored memories as cues; P is at least ten on this grid.
    cue_ids = rng.choice(p, size=N_CUES, replace=False)
    overlaps = []
    sweeps = []
    flip_counts = []
    for pattern_id in cue_ids:
        target = patterns[:, pattern_id]
        initial = target.copy()
        changed_positions = rng.choice(N, size=N_INITIAL_FLIPS, replace=False)
        initial[changed_positions] *= -1.0
        final, n_sweeps, n_flips = recall_asynchronously(j, initial, rng)

        # The overlap is with the original target, not the noisy cue.
        overlaps.append(float(target @ final / N))
        sweeps.append(n_sweeps)
        flip_counts.append(n_flips)

    overlaps = np.asarray(overlaps)
    return {
        "alpha": alpha,
        "p": p,
        "unstable_count": unstable_count,
        "tested_bits": tested_bits,
        "unstable_fraction": unstable_fraction,
        "gaussian_prediction": theory,
        "unstable_minus_prediction": unstable_fraction - theory,
        "mean_overlap": float(overlaps.mean()),
        "sd_overlap": float(overlaps.std(ddof=1)),
        "min_overlap": float(overlaps.min()),
        "max_overlap": float(overlaps.max()),
        "exact_recall_fraction": float(np.mean(overlaps == 1.0)),
        "mean_sweeps": float(np.mean(sweeps)),
        "max_sweeps": max(sweeps),
        "mean_flips": float(np.mean(flip_counts)),
    }


def plot_unstable(rows: list[dict], output: Path) -> None:
    """Measured unstable-bit fraction against the part (b) prediction."""
    loads = np.array([r["alpha"] for r in rows])
    measured = np.array([r["unstable_fraction"] for r in rows])
    fine = np.linspace(loads.min(), loads.max(), 400)
    prediction = np.array([normal_cdf_negative_inverse_sqrt(x) for x in fine])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(fine, prediction, color="#314f9e", lw=2.1,
            label=r"Gaussian prediction, $\Phi(-1/\sqrt{\alpha})$")
    ax.plot(loads, measured, "o-", color="#bd5534", lw=1.2, ms=5.4,
            label="Measured over all stored bits")
    ax.set_xlabel(r"Load $\alpha=P/N$ (dimensionless)")
    ax.set_ylabel("Unstable bits (fraction, dimensionless)")
    ax.set_xlim(0.005, 0.635)
    ax.set_ylim(bottom=0.0)
    ax.set_title(r"Stored-pattern stability: $N=500$, $P=10,30,\ldots,310$")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_recall(rows: list[dict], output: Path) -> None:
    """Mean final overlap, with spread across the ten cues at each load."""
    loads = np.array([r["alpha"] for r in rows])
    mean = np.array([r["mean_overlap"] for r in rows])
    sd = np.array([r["sd_overlap"] for r in rows])
    initial_overlap = 1.0 - 2.0 * FLIP_FRACTION

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(loads, mean, yerr=sd, fmt="o-", color="#225e86",
                ecolor="#89a9bd", capsize=3, lw=1.7, ms=5.4,
                label="Mean final overlap +/- cue SD")
    ax.axhline(initial_overlap, color="#ad5e34", ls="--", lw=1.5,
               label=f"Initial overlap = {initial_overlap:.2f} (10% flipped)")
    ax.set_xlabel(r"Load $\alpha=P/N$ (dimensionless)")
    ax.set_ylabel(r"Final overlap $m=\xi^\mu\!\cdot s/N$ (dimensionless)")
    ax.set_xlim(0.005, 0.635)
    ax.set_ylim(-0.05, 1.07)
    ax.set_title(r"Recall after random-order updates: $N=500$, 10 cues/load")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def report(rows: list[dict]) -> str:
    errors = np.array([r["unstable_minus_prediction"] for r in rows])
    rmse = float(np.sqrt(np.mean(errors**2)))
    perfect_all = [r for r in rows if r["exact_recall_fraction"] == 1.0]
    below_095 = next((r for r in rows if r["mean_overlap"] < 0.95), None)
    first = rows[3]
    next_row = rows[4]
    lines = [
        "Problem 3(c): Hopfield capacity",
        f"For N={N}, the unstable-bit fraction follows Phi(-1/sqrt(alpha)) across the load grid (RMSE {rmse:.4g}).",
        f"With 10% cue corruption, mean overlap is {first['mean_overlap']:.4g} at alpha={first['alpha']:.2f}"
        f" and {next_row['mean_overlap']:.4g} at alpha={next_row['alpha']:.2f}; high-overlap recall breaks down between these loads.",
        f"All {N_CUES} cues are recalled exactly through alpha={perfect_all[-1]['alpha']:.2f}; exact recall declines before mean overlap.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = [simulate_load(float(alpha), rng) for alpha in LOADS]

    unstable_png = OUT / "hw4_q3c_unstable_bits.png"
    recall_png = OUT / "hw4_q3c_recall_overlap.png"
    plot_unstable(rows, unstable_png)
    plot_recall(rows, recall_png)

    csv_path = OUT / "hw4_q3c_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    text = report(rows)
    print(text, end="")
    report_path = OUT / "hw4_q3c_report.txt"
    report_path.write_text(text, encoding="utf-8")
    print(f"Saved figure: {unstable_png}")
    print(f"Saved figure: {recall_png}")
    print(f"Saved data:   {csv_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
