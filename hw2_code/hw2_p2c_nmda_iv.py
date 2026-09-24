import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def B_func(V, mg=1.0, V0=16.13):
    """NMDA magnesium unblock fraction B(V)."""
    return 1.0 / (1.0 + (mg / 3.57) * np.exp(-V / V0))


def I_NMDA(V, mg=1.0, g_bar=1.0, EN=0.0, V0=16.13):
    """NMDA current normalized by g_bar (in mV, or pA per nS)."""
    return g_bar * B_func(V, mg=mg, V0=V0) * (V - EN)


def slope_factor(V, mg=1.0, EN=0.0, V0=16.13):
    """Bracket factor [1 + (V - EN)(1 - B)/V0] determining sign of dIN/dV."""
    b = B_func(V, mg=mg, V0=V0)
    return 1.0 + (V - EN) * (1.0 - b) / V0


if __name__ == "__main__":
    # =========================================================================
    # Problem 2(c): NMDA I-V Relation and Vanishing Slope Voltage
    # =========================================================================
    print("=" * 70)
    print("PROBLEM 2(c): NMDA I-V Relation and Magnesium Block")
    print("=" * 70)

    V0 = 16.13
    EN = 0.0
    mg_phys = 1.0

    # Numerical root for vanishing slope when [Mg2+] = 1.0 mM
    res = root_scalar(lambda V: slope_factor(V, mg=mg_phys, EN=EN, V0=V0),
                      bracket=[-90.0, 0.0], method="brentq")
    V_crit = res.root
    I_crit = I_NMDA(V_crit, mg=mg_phys, EN=EN, V0=V0)

    print(f"1. Voltage where slope vanishes for [Mg2+] = 1.0 mM:")
    print(f"   V* = {V_crit:.4f} mV -> Reported to 3 digits: {V_crit:.3g} mV (or {V_crit:.1f} mV).")
    print(f"   Normalized current at V*: I_N(V*) / g_bar = {I_crit:.3f} mV.")
    print()
    print("2. Meaning of the negative slope (dIN/dV < 0) below V*:")
    print("   - Below V* = -27.0 mV, the slope is negative (negative slope conductance).")
    print("   - Since inward synaptic current is negative (IN < 0), as the membrane depolarizes")
    print("     (V becomes less negative), the inward current becomes MORE negative (larger inward current).")
    print("   - This creates positive feedback (regenerative depolarization): depolarization")
    print("     removes Mg2+ block, letting in more inward current, depolarizing the cell further.")
    print("   - This enables NMDA receptors to act as coincidence detectors and support dendritic spikes.")
    print("=" * 70 + "\n")

    # Plotting Part 2(c): Single figure as requested
    fig, ax = plt.subplots(figsize=(7.5, 5))

    V_grid = np.linspace(-90.0, 0.0, 1000)

    ax.plot(V_grid, I_NMDA(V_grid, mg=0.0), color="tab:green", lw=1.8,
            label=r"$[\mathrm{Mg}^{2+}] = 0\,\mathrm{mM}$ (Ohmic)")
    ax.plot(V_grid, I_NMDA(V_grid, mg=0.1), color="tab:orange", lw=1.8,
            label=r"$[\mathrm{Mg}^{2+}] = 0.1\,\mathrm{mM}$")
    ax.plot(V_grid, I_NMDA(V_grid, mg=1.0), color="tab:blue", lw=2.0,
            label=r"$[\mathrm{Mg}^{2+}] = 1.0\,\mathrm{mM}$ (Physiological)")

    # Mark vanishing slope point (local minimum)
    ax.plot(V_crit, I_crit, "ro", markersize=7, zorder=5)

    ax.annotate(f"Minimum ($V^* = {V_crit:.1f}$ mV)\n$dI_N/dV = 0$",
                xy=(V_crit, I_crit), xytext=(-65.0, 0.0),
                arrowprops=dict(facecolor="tab:red", shrink=0.08, width=1, headwidth=5),
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="tab:red", alpha=0.9))

    ax.axhline(0.0, color="gray", linestyle=":", lw=1.0)
    ax.set_xlim(-90.0, 0.0)
    ax.set_ylim(-95.0, 10.0)
    ax.set_xlabel("Membrane potential V (mV)")
    ax.set_ylabel(r"Normalized current $I_N / \bar{g}$ (mV)")
    ax.set_title(r"NMDA Receptor $I$-$V$ Relation with $\mathrm{Mg}^{2+}$ Block (Problem 2c)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", bbox_to_anchor=(0.96, 0.06), framealpha=0.95)

    fig.tight_layout()
    plot_filename = "hw2_p2c_nmda_iv.png"
    fig.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {plot_filename}")
