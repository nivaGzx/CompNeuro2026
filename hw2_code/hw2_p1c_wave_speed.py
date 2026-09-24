import numpy as np
import matplotlib.pyplot as plt


def front_speed(a, D=1.0):
    """Active McKean traveling front speed c(a) for threshold a."""
    return np.sqrt(D) * (1.0 - 2.0 * a) / np.sqrt(a * (1.0 - a))


if __name__ == "__main__":
    D = 1.0

    # Key points for Problem 1(c) plot
    a_crossing = (2.0 - np.sqrt(2.0)) / 4.0
    c_crossing = front_speed(a_crossing, D)

    print("=" * 60)
    print("PROBLEM 1(c): Active Wavefront Speed vs Passive Speed")
    print("=" * 60)
    print(f"Crossing point (c = 2*sqrt(D)): a = {a_crossing:.4f}, c = {c_crossing:.4f}")
    print(f"Stalling point (c = 0):          a = 0.5000, c = 0.0000")
    print("=" * 60)

    # Plot c(a) vs passive apparent speed
    fig, ax = plt.subplots(figsize=(7.5, 5))

    a_grid = np.linspace(0.01, 0.99, 1000)
    c_grid = front_speed(a_grid, D)

    ax.plot(a_grid, c_grid, "b-", lw=2, label=r"Active front speed $c(a)$")
    ax.axhline(2.0 * np.sqrt(D), color="tab:red", linestyle="--", lw=1.8,
               label=r"Passive apparent speed $2\sqrt{D} = 2.0$")
    ax.axhline(0.0, color="gray", linestyle=":", lw=1.2)

    # Highlight crossing point
    ax.plot(a_crossing, 2.0 * np.sqrt(D), "ro", markersize=7, zorder=5)
    ax.annotate(f"Crossing: a = {a_crossing:.4f}\nc = 2.0",
                xy=(a_crossing, 2.0), xytext=(a_crossing + 0.08, 3.2),
                arrowprops=dict(facecolor="tab:red", shrink=0.08, width=1, headwidth=5),
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="tab:red", alpha=0.9))

    # Highlight stalling point
    ax.plot(0.5, 0.0, "go", markersize=7, zorder=5)
    ax.annotate("Stalling: a = 0.5\nc = 0",
                xy=(0.5, 0.0), xytext=(0.53, -1.8),
                arrowprops=dict(facecolor="tab:green", shrink=0.08, width=1, headwidth=5),
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="tab:green", alpha=0.9))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-5.0, 6.0)
    ax.set_xlabel("Excitation threshold a")
    ax.set_ylabel("Propagation speed c")
    ax.set_title("Active Front Speed vs Passive Apparent Speed (Problem 1c)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_fig = "hw2_p1c_wave_speed.png"
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {out_fig}\n")
