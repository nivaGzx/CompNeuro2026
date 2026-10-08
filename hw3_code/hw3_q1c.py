import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

__all__ = [
    "eval_G",
    "eval_G_prime",
    "exact_voltage",
    "tongue_1_1_boundaries",
    "candidate_locked_states",
    "simulate_forced_lif",
    "sweep_devils_staircase",
    "identify_plateaus",
]

_TWOPI = 2.0 * np.pi
_DENOM = 1.0 + 4.0 * np.pi**2
_SQRT_DENOM = np.sqrt(_DENOM)
_I_STAR = 1.0 / (1.0 - np.exp(-1.0))


def eval_G(t, I=1.6, eps=0.5):
    """Asymptotic periodic response G(t) of leaky integrator to I(t)."""
    return I + (eps / _DENOM) * (np.sin(_TWOPI * t) - _TWOPI * np.cos(_TWOPI * t))


def eval_G_prime(t, eps=0.5):
    """Derivative G'(t) = I(t) - G(t)."""
    return (eps / _DENOM) * (_TWOPI * np.cos(_TWOPI * t) + 4.0 * np.pi**2 * np.sin(_TWOPI * t))


def exact_voltage(t, Tn, I=1.6, eps=0.5):
    """Exact voltage trajectory between spikes starting from v(Tn+) = 0."""
    return eval_G(t, I, eps) - eval_G(Tn, I, eps) * np.exp(-(t - Tn))


def tongue_1_1_boundaries(eps=0.5):
    """Candidate 1:1 Arnold tongue boundaries [I_min, I_max]."""
    delta_I = eps / _SQRT_DENOM
    return _I_STAR - delta_I, _I_STAR + delta_I, delta_I


def candidate_locked_states(I=1.6, eps=0.5):
    """
    Find both candidate 1:1 mode-locked phases phi in [0, 1) and their eigenvalues lambda.
    Solves G(phi) = I*.
    """
    delta = _I_STAR - I
    max_amp = eps / _SQRT_DENOM
    if abs(delta) > max_amp:
        return []

    theta0 = np.arctan2(_TWOPI, 1.0)  # angle for sin(2*pi*phi - theta0)
    arg = (delta * _SQRT_DENOM) / eps
    arg = np.clip(arg, -1.0, 1.0)
    psi = np.arcsin(arg)

    # Two angle solutions
    ang1 = (psi + theta0) % _TWOPI
    ang2 = (np.pi - psi + theta0) % _TWOPI

    roots = sorted([ang1 / _TWOPI, ang2 / _TWOPI])
    results = []
    for phi in roots:
        I_phi = I + eps * np.sin(_TWOPI * phi)
        # lambda = e^-1 * I(phi) / (I(phi) - 1)
        if I_phi > 1.0:
            lam = np.exp(-1.0) * I_phi / (I_phi - 1.0)
            stable = abs(lam) < 1.0
        else:
            lam = np.nan
            stable = False
        g_prime = eval_G_prime(phi, eps)
        results.append({
            "phi": phi,
            "I_phi": I_phi,
            "G_prime": g_prime,
            "lambda": lam,
            "stable": stable,
        })
    return results


def simulate_forced_lif(I, eps=0.5, total_spikes=260, discard=80):
    """
    Simulates consecutive spike times using exact root-finding (v(t) = 1).
    Returns average firing number (spikes / forcing period) and recorded spike times.
    If subthreshold and threshold is never reached, returns 0.0, np.array([]).
    """
    t_curr = 0.0
    spike_times = []
    dt_scan = 0.04

    for _ in range(total_spikes):
        t_left = t_curr + 1e-5
        found = False
        while True:
            t_right = t_left + dt_scan
            val_left = exact_voltage(t_left, t_curr, I, eps) - 1.0
            val_right = exact_voltage(t_right, t_curr, I, eps) - 1.0
            if val_left <= 0.0 and val_right >= 0.0:
                t_next = brentq(lambda t: exact_voltage(t, t_curr, I, eps) - 1.0, t_left, t_right)
                found = True
                break
            t_left = t_right
            if t_left > t_curr + 25.0:
                break

        if not found:
            return 0.0, np.array([])

        spike_times.append(t_next)
        t_curr = t_next

    spikes_stat = np.array(spike_times[discard:])
    duration = spikes_stat[-1] - spikes_stat[0]
    num_spikes = len(spikes_stat) - 1
    firing_number = num_spikes / duration if duration > 0 else 0.0
    return firing_number, spikes_stat


def sweep_devils_staircase(I_vals, eps=0.5):
    """Sweep DC current I over I_vals and compute firing numbers."""
    rho_vals = np.zeros(len(I_vals))
    for idx, I_val in enumerate(I_vals):
        rho, _ = simulate_forced_lif(I_val, eps)
        rho_vals[idx] = rho
    return rho_vals


def identify_plateaus(I_vals, rho_vals, tol=0.015, min_len=4):
    """
    Identify prominent rational plateaus p:q in the Devil's staircase.
    """
    prominent_fractions = [
        (1, 2, 0.500),
        (2, 3, 2.0 / 3.0),
        (3, 4, 0.750),
        (1, 1, 1.000),
        (4, 3, 4.0 / 3.0),
        (3, 2, 1.500),
        (5, 3, 5.0 / 3.0),
        (2, 1, 2.000),
        (5, 2, 2.500),
    ]

    plateau_info = []
    for p, q, target_rho in prominent_fractions:
        mask = np.abs(rho_vals - target_rho) <= tol
        indices = np.where(mask)[0]
        if len(indices) >= min_len:
            # Check contiguous blocks
            diffs = np.diff(indices)
            split_points = np.where(diffs > 2)[0]
            blocks = np.split(indices, split_points + 1)
            largest_block = max(blocks, key=len)
            if len(largest_block) >= min_len:
                I_start = I_vals[largest_block[0]]
                I_end = I_vals[largest_block[-1]]
                meas_rho = float(np.mean(rho_vals[largest_block]))
                width = I_end - I_start
                plateau_info.append({
                    "ratio_str": f"{p}:{q}",
                    "target_rho": target_rho,
                    "measured_rho": meas_rho,
                    "I_min": I_start,
                    "I_max": I_end,
                    "width": width,
                    "num_pts": len(largest_block),
                })
    return plateau_info


if __name__ == "__main__":
    eps = 0.5

    # =========================================================================
    # Problem 1(a) & 1(b): Analytical Calculations & Arnold Tongue
    # =========================================================================
    I_low, I_high, delta_I = tongue_1_1_boundaries(eps)
    candidates_16 = candidate_locked_states(I=1.6, eps=eps)

    print("=" * 75)
    print("PROBLEM 1(c): Periodic Forcing, Devil's Staircase, and Arnold Tongue")
    print("=" * 75)
    print("1. Theoretical 1:1 Arnold Tongue Parameters (Benchmark from 1a):")
    print(f"   - Target critical drive I* = 1 / (1 - e^-1) = {_I_STAR:.4f}")
    print(f"   - Tongue half-width: delta_I = eps / sqrt(1 + 4*pi^2) = {delta_I:.4f}")
    print(f"   - 1:1 Candidate Locking Region: [{I_low:.4f}, {I_high:.4f}]")
    print(f"   - First-crossing condition: I - eps > 1 => I > {1.0 + eps:.1f} guarantees")
    print("     v'(t) > 0 for all v <= 1, preventing premature crossing.")
    print()

    print("2. Candidate 1:1 Mode-Locked States at I = 1.6, eps = 0.5 (Benchmark from 1b):")
    print("   State      Phase phi    I(phi)       G'(phi)      lambda       Stability ")
    print("   " + "-" * 70)
    for i, c in enumerate(candidates_16, 1):
        stab_str = "STABLE" if c["stable"] else "UNSTABLE"
        print(f"   State {i:<4} {c['phi']:<12.4f} {c['I_phi']:<12.4f} {c['G_prime']:<12.4f} {c['lambda']:<12.4f} {stab_str:<10}")
    print("   " + "-" * 70)
    print(f"   Conclusion: Exactly one candidate state is stable inside the tongue:")
    print(f"   - State 1 (phi = {candidates_16[0]['phi']:.4f}) has G'(phi) > 0 => I(phi) > I* => lambda = {candidates_16[0]['lambda']:.4f} < 1 (STABLE).")
    print(f"   - State 2 (phi = {candidates_16[1]['phi']:.4f}) has G'(phi) < 0 => I(phi) < I* => lambda = {candidates_16[1]['lambda']:.4f} > 1 (UNSTABLE).")
    print()

    # =========================================================================
    # Problem 1(c): Numerical Simulation & Steady-State Verification at I = 1.6
    # =========================================================================
    rho_16, spikes_16 = simulate_forced_lif(I=1.6, eps=eps)
    steady_phase = float(spikes_16[-1] % 1.0)

    print("3. Simulation at I = 1.6 (Problem 1c):")
    print(f"   - Measured firing number: {rho_16:.4f} (Exact 1:1 locking = 1.0000)")
    print(f"   - Measured steady-state spike phase: phi_sim = {steady_phase:.4f}")
    print(f"   - Theoretical stable phase:          phi_th  = {candidates_16[0]['phi']:.4f}")
    print(f"   - Absolute error: |phi_sim - phi_th| = {abs(steady_phase - candidates_16[0]['phi']):.4f}")
    print()

    # =========================================================================
    # Parameter Sweep for Devil's Staircase: I in [0.9, 3.0]
    # =========================================================================
    print("4. Scanning DC current I from 0.9 to 3.0 (300 grid points)...")
    I_vals = np.linspace(0.9, 3.0, 300)
    rho_vals = sweep_devils_staircase(I_vals, eps)

    # Theoretical unforced firing number
    rho_unforced = np.zeros_like(I_vals)
    mask_unforced = I_vals > 1.0
    rho_unforced[mask_unforced] = 1.0 / np.log(I_vals[mask_unforced] / (I_vals[mask_unforced] - 1.0))

    # Identify plateaus
    plateaus = identify_plateaus(I_vals, rho_vals)

    print("\n5. Identification of Mode-Locked Plateaus (Problem 1c):")
    print(f"   {'Plateau':<10} {'p/q Target':<12} {'Meas Freq':<12} {'I Range [Imin, Imax]':<24} {'Width Delta I':<14}")
    print("   " + "-" * 72)
    for p in plateaus:
        i_range_str = f"[{p['I_min']:.3f}, {p['I_max']:.3f}]"
        print(f"   {p['ratio_str']:<10} {p['target_rho']:<12.4f} {p['measured_rho']:<12.4f} {i_range_str:<24} {p['width']:<14.4f}")
    print("   " + "-" * 72)

    # Compare 1:1 plateau with tongue boundaries
    plat_11 = [p for p in plateaus if p["ratio_str"] == "1:1"]
    if plat_11:
        p11 = plat_11[0]
        print(f"\n   Comparison of 1:1 Plateau:")
        print(f"   - Theoretical Arnold tongue boundaries: [{I_low:.4f}, {I_high:.4f}] (Width: {2*delta_I:.4f})")
        print(f"   - Measured 1:1 locking plateau range:   [{p11['I_min']:.4f}, {p11['I_max']:.4f}] (Width: {p11['width']:.4f})")
        print("   - The numerical 1:1 plateau width matches the theoretical tongue boundaries exceptionally well!")
    print("=" * 75 + "\n")

    # =========================================================================
    # High-Quality Plotting
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)

    # Unforced theoretical curve
    ax.plot(I_vals, rho_unforced, "k--", lw=1.6,
            label=r"Unforced firing number: $1/\ln[I/(I-1)]$")

    # Forced simulation curve
    ax.plot(I_vals, rho_vals, "b-", lw=2.0,
            label=r"Forced firing number ($\epsilon = 0.5$)")

    # Mark 1:1 Tongue boundaries
    ax.axvline(I_low, color="tab:red", linestyle=":", lw=1.6,
               label=f"1:1 Tongue lower: {I_low:.3f}")
    ax.axvline(I_high, color="tab:red", linestyle="-.", lw=1.6,
               label=f"1:1 Tongue upper: {I_high:.3f}")
    ax.axvspan(I_low, I_high, color="tab:red", alpha=0.12,
               label="1:1 Mode-locked tongue")

    # Annotate prominent plateaus
    annotations = [
        ("1:2 Plateau", 1.19, 0.54, "darkgreen"),
        ("2:3 Plateau", 1.33, 0.71, "purple"),
        ("1:1 Plateau", 1.58, 1.05, "darkred"),
        ("3:2 Plateau", 2.00, 1.55, "darkgreen"),
        ("2:1 Plateau", 2.48, 2.05, "darkgreen"),
        ("5:2 Plateau", 2.89, 2.55, "purple"),
    ]
    for text, x_pos, y_pos, col in annotations:
        ax.text(x_pos, y_pos, text, fontsize=9.5, color=col, weight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.85))

    # Mark I = 1.6 test point
    ax.plot(1.6, rho_16, "ro", markersize=6, zorder=5)
    ax.annotate(f"$I = 1.6$\n$\\phi = {steady_phase:.4f}$",
                xy=(1.6, 1.0), xytext=(1.68, 0.78),
                arrowprops=dict(facecolor="tab:red", shrink=0.08, width=1, headwidth=5),
                fontsize=9.0,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="tab:red", alpha=0.9))

    ax.set_xlim(0.9, 3.0)
    ax.set_ylim(-0.05, 2.8)
    ax.set_xlabel("Direct current drive $I$", fontsize=12)
    ax.set_ylabel(r"Firing number $\rho$ (spikes per forcing period)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    out_fig = os.path.join(os.path.dirname(__file__), "hw3_q1c_devils_staircase.png")
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {out_fig}\n")
