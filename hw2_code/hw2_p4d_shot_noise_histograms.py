import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.stats import norm, skew


def simulate_shot_noise(nu_tau, V_mean_target=10.0, tau=20.0, dt=0.02, T=500000.0, seed=42):
    """
    Simulate subthreshold shot noise voltage trace:
        tau * dV/dt = -V + tau * sum_k w * delta(t - t_k)
    where t_k is a Poisson process with rate nu, step size w = V_mean_target / (nu * tau).
    """
    nu = nu_tau / tau           # rate in ms^(-1)
    w = V_mean_target / nu_tau  # step size in mV

    n_steps = int(T / dt)
    lam = nu * dt               # Poisson mean per dt bin

    rng = np.random.default_rng(seed)
    dN = rng.poisson(lam, size=n_steps)

    alpha = np.exp(-dt / tau)
    # AR(1) exact exponential filter: V[n] = alpha * V[n-1] + w * dN[n]
    V = lfilter([w], [1.0, -alpha], dN)

    # Discard initial transient (burn-in = 500 ms = 25 * tau)
    burn_steps = int(500.0 / dt)
    V_stat = V[burn_steps:]

    meas_mean = float(np.mean(V_stat))
    meas_var = float(np.var(V_stat))
    meas_skew = float(skew(V_stat))

    th_mean = V_mean_target
    th_var = 0.5 * nu * (w**2) * tau
    th_skew = (2.0 * np.sqrt(2.0)) / (3.0 * np.sqrt(nu_tau))

    stats = {
        "nu_tau": nu_tau,
        "w": w,
        "nu_hz": nu * 1000.0,
        "meas_mean": meas_mean,
        "th_mean": th_mean,
        "meas_var": meas_var,
        "th_var": th_var,
        "meas_skew": meas_skew,
        "th_skew": th_skew,
    }
    return V_stat, stats


if __name__ == "__main__":
    # =========================================================================
    # Problem 4(d): Direct Simulation of Shot Noise
    # =========================================================================
    print("=" * 70)
    print("PROBLEM 4(d): Shot Noise Simulation Results (<V> = 10 mV, tau = 20 ms)")
    print("=" * 70)

    nu_tau_list = [2, 20, 200]
    dt = 0.02     # ms
    T = 500000.0  # ms (500 s)
    V_target = 10.0
    tau = 20.0    # ms

    results = {}
    traces_for_plot = {}

    print(f"{'nu*tau':<8} {'w (mV)':<10} {'nu (Hz)':<10} {'Mean (th/meas)':<20} {'Variance (th/meas)':<22} {'Skewness (th/meas)':<20}")
    print("-" * 90)

    for nu_tau in nu_tau_list:
        V_stat, stats = simulate_shot_noise(nu_tau, V_target, tau, dt, T, seed=123)
        results[nu_tau] = stats
        if nu_tau in [2, 200]:
            traces_for_plot[nu_tau] = V_stat

        m_str = f"{stats['th_mean']:.3f} / {stats['meas_mean']:.3f}"
        v_str = f"{stats['th_var']:.4g} / {stats['meas_var']:.4g}"
        s_str = f"{stats['th_skew']:.4g} / {stats['meas_skew']:.4g}"
        print(f"{nu_tau:<8} {stats['w']:<10.4g} {stats['nu_hz']:<10.1f} {m_str:<20} {v_str:<22} {s_str:<20}")

    print("=" * 70 + "\n")

    # =========================================================================
    # Plotting Part 4(d): Histograms overlaid with matched Gaussians
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel 1: nu*tau = 2
    V2 = traces_for_plot[2]
    s2 = results[2]
    mu2 = s2["th_mean"]
    sigma2 = np.sqrt(s2["th_var"])
    v_grid2 = np.linspace(0.0, 36.0, 1000)

    ax1.hist(V2, bins=100, range=(0.0, 36.0), density=True,
             color="tab:blue", alpha=0.55, edgecolor="none",
             label="Simulated shot noise")
    ax1.plot(v_grid2, norm.pdf(v_grid2, loc=mu2, scale=sigma2),
             "r--", lw=2.0,
             label=f"Matched Gaussian $\\mathcal{{N}}({mu2:.0f}, {sigma2**2:.0f})$\nSkew: {s2['th_skew']:.3f} (th), {s2['meas_skew']:.3f} (sim)")

    ax1.set_xlim(0.0, 36.0)
    ax1.set_ylim(0.0, 0.105)
    ax1.set_xlabel("Membrane potential V (mV)")
    ax1.set_ylabel(r"Probability density (mV$^{-1}$)")
    ax1.set_title(r"(A) $\nu\tau = 2$ (Sparse, $w = 5.0$ mV)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9.0)

    # Panel 2: nu*tau = 200
    V200 = traces_for_plot[200]
    s200 = results[200]
    mu200 = s200["th_mean"]
    sigma200 = np.sqrt(s200["th_var"])
    v_grid200 = np.linspace(7.5, 12.5, 1000)

    ax2.hist(V200, bins=80, range=(7.5, 12.5), density=True,
             color="tab:green", alpha=0.55, edgecolor="none",
             label="Simulated shot noise")
    ax2.plot(v_grid200, norm.pdf(v_grid200, loc=mu200, scale=sigma200),
             "r--", lw=2.0,
             label=f"Matched Gaussian $\\mathcal{{N}}({mu200:.0f}, {sigma200**2:.2f})$\nSkew: {s200['th_skew']:.3f} (th), {s200['meas_skew']:.3f} (sim)")

    ax2.set_xlim(7.5, 12.5)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Membrane potential V (mV)")
    ax2.set_ylabel(r"Probability density (mV$^{-1}$)")
    ax2.set_title(r"(B) $\nu\tau = 200$ (Dense, $w = 0.05$ mV)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.95, fontsize=9.0)

    fig.tight_layout()
    out_fig = "hw2_p4d_shot_noise_histograms.png"
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {out_fig}")
