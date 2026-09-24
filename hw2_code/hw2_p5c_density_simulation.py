import numpy as np
import matplotlib.pyplot as plt


def simulate_em(mu, theta, sigma2, dt, n_steps, burn_steps, seed=42):
    """
    Euler-Maruyama simulation of noisy integrator with reset:
        du = mu * dt + sqrt(2 * sigma2 * dt) * dW
    Spike emitted at theta, reset to u = 0.
    """
    rng = np.random.default_rng(seed)

    chunk_size = 500000
    n_chunks = int(np.ceil(n_steps / chunk_size))

    drift = mu * dt
    diff = np.sqrt(2.0 * sigma2 * dt)

    u = 0.0
    spikes = 0
    total_samples = 0
    neg_samples = 0

    hist_capacity = 600000
    hist_subsample = max(1, (n_steps - burn_steps) // hist_capacity)
    hist_samples = []

    step = 0
    for _ in range(n_chunks):
        current_chunk_len = min(chunk_size, n_steps - step)
        dW = rng.standard_normal(current_chunk_len)
        increments = drift + diff * dW

        for k in range(current_chunk_len):
            u += increments[k]
            if u >= theta:
                if step >= burn_steps:
                    spikes += 1
                u = 0.0

            if step >= burn_steps:
                total_samples += 1
                if u < 0.0:
                    neg_samples += 1
                if (step - burn_steps) % hist_subsample == 0:
                    hist_samples.append(u)

            step += 1

    hist_samples = np.array(hist_samples)
    T_stat = (n_steps - burn_steps) * dt
    meas_rate = spikes / T_stat
    meas_frac = neg_samples / total_samples

    return hist_samples, meas_rate, meas_frac


def theory_p(u, mu=1.0, theta=1.0, sigma2=1.0):
    """Stationary probability density p(u) of noisy integrator."""
    nu = mu / theta
    k = mu / sigma2
    p = np.zeros_like(u)

    pos_mask = (u >= 0.0) & (u <= theta)
    p[pos_mask] = (nu / mu) * (1.0 - np.exp(k * (u[pos_mask] - theta)))

    neg_mask = (u < 0.0)
    p[neg_mask] = (nu / mu) * (1.0 - np.exp(-k * theta)) * np.exp(k * u[neg_mask])

    return p


if __name__ == "__main__":
    mu = 1.0
    theta = 1.0
    sigma2_list = [0.25, 1.0, 4.0]

    # =========================================================================
    # Problem 5(c): Euler-Maruyama Simulation and Comparison
    # =========================================================================
    print("=" * 70)
    print("PROBLEM 5(c): Simulation Results vs Theory")
    print("=" * 70)

    dt = 0.0005
    T = 5000.0     # 5000 time units
    T_burn = 100.0 # 100 time units of burn-in

    n_steps = int(T / dt)
    burn_steps = int(T_burn / dt)

    print(f"{'sigma^2':<10} {'Meas Rate':<14} {'Th Rate':<12} {'Meas Frac(<0)':<18} {'Th Frac(<0)':<14}")
    print("-" * 68)

    sim_results = []
    for idx, s2 in enumerate(sigma2_list):
        samples, meas_rate, meas_frac = simulate_em(
            mu, theta, s2, dt, n_steps, burn_steps, seed=2026 + idx
        )
        th_rate = mu / theta
        th_frac = (s2 / (mu * theta)) * (1.0 - np.exp(-mu * theta / s2))
        sim_results.append((s2, samples, meas_rate, th_rate, meas_frac, th_frac))

        print(f"{s2:<10.2f} {meas_rate:<14.4f} {th_rate:<12.4f} {meas_frac:<18.4f} {th_frac:<14.4f}")

    print("-" * 68)
    print()
    print("Why discarding initial transient matters:")
    print("- The neuron starts at u(0) = 0. It takes time for diffusion to populate the negative")
    print("  voltage region (u < 0) and reach the steady-state distribution p(u).")
    print("- Discarding the initial transient ensures statistics reflect the stationary state.")
    print("=" * 70 + "\n")

    # =========================================================================
    # Plotting Part 5(c): Subthreshold voltage histograms with theoretical p(u) overlay
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    x_limits = [(-1.5, 1.0), (-5.0, 1.0), (-14.0, 1.0)]
    y_limits = [(0.0, 1.15), (0.0, 0.75), (0.0, 0.26)]

    for idx, (s2, samples, meas_rate, th_rate, meas_frac, th_frac) in enumerate(sim_results):
        ax = axes[idx]
        xlim = x_limits[idx]
        ylim = y_limits[idx]

        bins = np.linspace(xlim[0], theta, 70)
        ax.hist(samples, bins=bins, density=True, alpha=0.55,
                color="tab:blue", edgecolor="none", label="Simulated (EM)")

        u_grid = np.linspace(xlim[0], theta, 500)
        p_grid = theory_p(u_grid, mu=mu, theta=theta, sigma2=s2)
        ax.plot(u_grid, p_grid, "r-", lw=2.0, label="Theory $p(u)$")

        ax.axvline(0.0, color="gray", linestyle="--", lw=1.2, label="Reset ($u=0$)")
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("Membrane voltage u")
        ax.set_ylabel("Probability density $p(u)$")
        ax.set_title(f"$\\sigma^2 = {s2:g}$")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    fig.tight_layout()
    plot_path = "hw2_p5c_density_simulation.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {plot_path}")
