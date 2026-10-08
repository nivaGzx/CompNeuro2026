import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

__all__ = [
    "lif_exact_period",
    "lif_prc_exact",
    "lif_prc_numeric",
    "fhn_rhs",
    "fhn_find_limit_cycle",
    "fhn_prc_numeric",
]


# =============================================================================
# 1. Leaky Integrate-and-Fire (LIF)
# =============================================================================

def lif_exact_period(I=1.5, tau=1.0, vr=0.0, vth=1.0):
    """Analytic period of periodic LIF neuron."""
    if I * tau <= vth:
        return np.inf
    return tau * np.log((I * tau - vr) / (I * tau - vth))


def lif_prc_exact(phi, I=1.5, tau=1.0, vr=0.0, vth=1.0):
    """Analytic infinitesimal PRC for LIF: R(phi) = tau * exp(phi*Delta/tau) / [Delta*(I*tau - vr)]."""
    Delta = lif_exact_period(I, tau, vr, vth)
    phi_arr = np.asarray(phi)
    return tau * np.exp(phi_arr * Delta / tau) / (Delta * (I * tau - vr))


def lif_prc_numeric(phi, I=1.5, tau=1.0, vr=0.0, vth=1.0, delta_v=1e-3):
    """
    Measure LIF PRC numerically by kicking voltage by delta_v at phase phi.
    R(phi) approx (Delta - T') / (Delta * delta_v).
    """
    Delta = lif_exact_period(I, tau, vr, vth)
    phi_arr = np.asarray(phi)
    R_num = np.zeros_like(phi_arr, dtype=float)

    for i, p in enumerate(phi_arr):
        t_kick = p * Delta
        # Voltage reached right before kick
        v_before = I * tau - (I * tau - vr) * np.exp(-t_kick / tau)
        v_after = v_before + delta_v
        if v_after >= vth:
            T_prime = t_kick
        else:
            dt_rem = tau * np.log((I * tau - v_after) / (I * tau - vth))
            T_prime = t_kick + dt_rem
        R_num[i] = (Delta - T_prime) / (Delta * delta_v)
    return R_num


# =============================================================================
# 2. FitzHugh-Nagumo (FHN) Oscillator
# =============================================================================

def fhn_rhs(t, y):
    """RHS of FitzHugh-Nagumo oscillator."""
    v, w = y
    dv = v - (v**3) / 3.0 - w + 0.5
    dw = 0.08 * (v + 0.7 - 0.8 * w)
    return [dv, dw]


def fhn_find_limit_cycle(t_transient=500.0, rtol=1e-10, atol=1e-10):
    """
    Integrate FHN to settle onto the stable limit cycle and locate reference phase 0
    (defined as upward crossing of v = 0 with dv/dt > 0).
    Returns initial state y_phase0 on the cycle and period Delta.
    """
    # Event for upward crossing of v = 0
    def v_zero_up(t, y):
        return y[0]
    v_zero_up.direction = 1

    # Burn in transient
    sol_relax = solve_ivp(fhn_rhs, [0.0, t_transient], [-1.0, 0.0],
                          rtol=rtol, atol=atol)
    y_init = sol_relax.y[:, -1]

    # Find first upward crossing to fix phase zero
    sol_cross1 = solve_ivp(fhn_rhs, [0.0, 100.0], y_init,
                           events=v_zero_up, rtol=rtol, atol=atol)
    y_phase0 = sol_cross1.y_events[0][0]

    # Measure period Delta to next upward crossing
    sol_cycle = solve_ivp(fhn_rhs, [0.0, 100.0], y_phase0,
                          events=v_zero_up, rtol=rtol, atol=atol)
    Delta = sol_cycle.t_events[0][1]  # second crossing is one full period later
    return y_phase0, Delta


def fhn_prc_numeric(n_phases=80, delta_v=1e-3, n_cycles=5, rtol=1e-10, atol=1e-10):
    """
    Measure FHN PRC numerically by kicking v by delta_v at phase phi.
    Measures the shift in the N-th upward zero-crossing after transverse relaxation.
    """
    y_phase0, Delta = fhn_find_limit_cycle(rtol=rtol, atol=atol)

    def v_zero_up(t, y):
        return y[0]
    v_zero_up.direction = 1

    phis = np.linspace(0.0, 1.0, n_phases, endpoint=False)
    R_vals = np.zeros(n_phases)

    for i, p in enumerate(phis):
        # 1. Evolve unperturbed state to phase phi
        sol_kick = solve_ivp(fhn_rhs, [0.0, p * Delta], y_phase0,
                             rtol=rtol, atol=atol)
        y_pert = sol_kick.y[:, -1].copy()

        # 2. Perturb voltage variable v by delta_v
        y_pert[0] += delta_v

        # 3. Integrate forward across n_cycles to relax back to limit cycle
        T_end = (n_cycles + 2) * Delta
        sol_after = solve_ivp(fhn_rhs, [0.0, T_end], y_pert,
                              events=v_zero_up, rtol=rtol, atol=atol)
        crossings = sol_after.t_events[0]
        crossings = crossings[crossings > 1e-4]  # discard t=0 crossing if any

        # Absolute time of N-th crossing
        T_prime_N = p * Delta + crossings[n_cycles - 1]

        # Phase shift: (N*Delta - T_prime_N) / (Delta * delta_v)
        R_vals[i] = (n_cycles * Delta - T_prime_N) / (Delta * delta_v)

    return phis, R_vals, Delta, y_phase0


if __name__ == "__main__":
    # =========================================================================
    # Problem 2(a) & 2(c): LIF Model PRC Comparison
    # =========================================================================
    tau_lif = 1.0
    vr_lif = 0.0
    vth_lif = 1.0
    I_lif = 1.5
    delta_v_lif = 1e-3

    Delta_lif = lif_exact_period(I_lif, tau_lif, vr_lif, vth_lif)
    phis_lif = np.linspace(0.01, 0.99, 100)
    R_lif_exact = lif_prc_exact(phis_lif, I_lif, tau_lif, vr_lif, vth_lif)
    R_lif_num = lif_prc_numeric(phis_lif, I_lif, tau_lif, vr_lif, vth_lif, delta_v=delta_v_lif)

    lif_max_err = float(np.max(np.abs(R_lif_num - R_lif_exact)))

    print("=" * 75)
    print("PROBLEM 2(c): Phase Response Curves (LIF & FitzHugh-Nagumo)")
    print("=" * 75)
    print("1. Leaky Integrate-and-Fire (LIF) Results:")
    print(f"   - Parameters: tau = {tau_lif:.1f}, vr = {vr_lif:.1f}, vth = {vth_lif:.1f}, I = {I_lif:.1f}")
    print(f"   - Firing Period: Delta = ln(3) = {Delta_lif:.4f}")
    print(f"   - Perturbation size: delta_v = {delta_v_lif:.4g}")
    print(f"   - PRC Range: [{np.min(R_lif_exact):.4f}, {np.max(R_lif_exact):.4f}]")
    print(f"   - Max numerical vs analytic error: {lif_max_err:.4e}")
    print("   - Firing regime: Strictly positive PRC (R(phi) > 0 for all phi).")
    print("   - Classification: TYPE I PRC (strictly excitatory advances, no delay).")
    print()

    # Sample table comparing LIF exact and numeric at key phases
    sample_phis = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"   {'Phase phi':<12} {'Exact R(phi)':<16} {'Numeric R(phi)':<16} {'Abs Error':<14}")
    print("   " + "-" * 60)
    for p in sample_phis:
        r_ex = float(lif_prc_exact(p, I_lif, tau_lif, vr_lif, vth_lif))
        r_nm = float(lif_prc_numeric([p], I_lif, tau_lif, vr_lif, vth_lif, delta_v_lif)[0])
        print(f"   {p:<12.2f} {r_ex:<16.4f} {r_nm:<16.4f} {abs(r_ex - r_nm):<14.4e}")
    print("   " + "-" * 60 + "\n")

    # =========================================================================
    # Problem 2(c): FitzHugh-Nagumo Oscillator PRC
    # =========================================================================
    print("2. FitzHugh-Nagumo (FHN) Oscillator PRC Results:")
    phis_fhn, R_fhn, Delta_fhn, _ = fhn_prc_numeric(n_phases=80, delta_v=1e-3, n_cycles=5)

    idx_max = int(np.argmax(R_fhn))
    idx_min = int(np.argmin(R_fhn))
    max_R = R_fhn[idx_max]
    min_R = R_fhn[idx_min]
    phi_max = phis_fhn[idx_max]
    phi_min = phis_fhn[idx_min]

    neg_mask = R_fhn < 0.0
    neg_indices = np.where(neg_mask)[0]
    phi_neg_start = phis_fhn[neg_indices[0]]
    phi_neg_end = phis_fhn[neg_indices[-1]]

    print(f"   - Oscillator Period: Delta = {Delta_fhn:.4f}")
    print(f"   - Maximum PRC value: R_max = {max_R:.4f} at phase phi = {phi_max:.3f} (Advance)")
    print(f"   - Minimum PRC value: R_min = {min_R:.4f} at phase phi = {phi_min:.3f} (Delay)")
    print(f"   - Negative region (R < 0): phi in [{phi_neg_start:.3f}, {phi_neg_end:.3f}]")
    print("   - Classification: TYPE II PRC (biphasic, contains negative and positive lobes).")
    print()

    # =========================================================================
    # Problem 2(c): Answers to Conceptual & Physical Questions
    # =========================================================================
    print("3. Answers to Problem 2(c) Questions:")
    print("   (a) Which PRC is Type I and which is Type II?")
    print("       - LIF neuron has a TYPE I PRC:")
    print("         R(phi) >= 0 everywhere over the cycle. Any depolarizing kick")
    print("         moves the voltage closer to threshold, strictly advancing the next spike.")
    print("       - FitzHugh-Nagumo oscillator has a TYPE II PRC:")
    print("         R(phi) is biphasic, possessing a substantial negative lobe (R < 0)")
    print("         as well as a positive lobe (R > 0).")
    print()
    print("   (b) What does a negative lobe mean physically?")
    print("       - Definition: A negative PRC value (R(phi) < 0) means that a depolarizing")
    print("         perturbation causes a PHASE DELAY (the next spike fires LATER than")
    print("         unperturbed, T' > Delta).")
    print("       - Biophysical Mechanism in FHN:")
    print("         In the FitzHugh-Nagumo model, during the downstroke and refractory phase")
    print("         (phi in [0.225, 0.788]), the slow recovery variable w is elevated.")
    print("         An extra depolarizing kick in v slightly increases v, but because")
    print("         dw/dt = 0.08*(v + 0.7 - 0.8*w), it accelerates the production of outward")
    print("         recovery variable w. The increased outward current deepens and prolongs")
    print("         the hyperpolarization valley, trapping the state longer before it can")
    print("         initiate the next upstroke. Hence, excitation paradoxically delays firing!")
    print("=" * 75 + "\n")

    # =========================================================================
    # High-Quality Dual-Panel Plotting
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Panel A: LIF PRC
    ax1.plot(phis_lif, R_lif_exact, "k-", lw=2.4,
             label=r"Analytical: $R(\varphi) = \frac{\tau e^{\varphi\Delta/\tau}}{\Delta(I\tau-v_r)}$")
    ax1.plot(phis_lif[::4], R_lif_num[::4], "ro", mfc="none", mew=1.6, ms=6.5,
             label=r"Numerical ($\delta = 10^{-3}$)")
    ax1.set_xlabel(r"Phase $\varphi$", fontsize=11)
    ax1.set_ylabel(r"PRC $R(\varphi)$", fontsize=11)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.5, 2.0)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Panel B: FHN PRC
    ax2.plot(phis_fhn, R_fhn, "b-", lw=2.2, label=r"Numerical FHN PRC ($\Delta = 39.47$)")
    ax2.axhline(0.0, color="gray", linestyle="--", lw=1.2)

    # Shaded lobes
    ax2.fill_between(phis_fhn, 0.0, R_fhn, where=(R_fhn < 0),
                     color="salmon", alpha=0.4, label="Negative lobe (Phase delay)")
    ax2.fill_between(phis_fhn, 0.0, R_fhn, where=(R_fhn > 0),
                     color="skyblue", alpha=0.4, label="Positive lobe (Phase advance)")

    # Mark extrema
    ax2.plot(phi_max, max_R, "go", markersize=6.5, zorder=5)
    ax2.annotate(f"Max $R = {max_R:.3f}$\n($\\phi = {phi_max:.3f}$)",
                 xy=(phi_max, max_R), xytext=(phi_max - 0.25, max_R - 0.08),
                 arrowprops=dict(facecolor="tab:green", shrink=0.08, width=1, headwidth=5),
                 fontsize=9.0,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="tab:green", alpha=0.9))

    ax2.plot(phi_min, min_R, "ro", markersize=6.5, zorder=5)
    ax2.annotate(f"Min $R = {min_R:.3f}$\n($\\phi = {phi_min:.3f}$)",
                 xy=(phi_min, min_R), xytext=(phi_min + 0.06, min_R - 0.08),
                 arrowprops=dict(facecolor="tab:red", shrink=0.08, width=1, headwidth=5),
                 fontsize=9.0,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="tab:red", alpha=0.9))

    ax2.set_xlabel(r"Phase $\varphi$ (zero crossing $v=0, \dot{v}>0$)", fontsize=11)
    ax2.set_ylabel(r"PRC $R(\varphi)$", fontsize=11)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.25, 0.38)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)

    fig.tight_layout()
    out_fig = os.path.join(os.path.dirname(__file__), "hw3_q2c_prc.png")
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {out_fig}\n")
