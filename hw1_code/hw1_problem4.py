import numpy as np
import matplotlib.pyplot as plt


def lif_sim(I=1.1, T=30.0, dt=1e-4, taum=10.0, R=1.0, urest=0.0, uth=1.0, reset=False):
    Ifun = I if callable(I) else (lambda t: I)
    nt = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, nt)
    u = np.zeros(nt)
    u[0] = urest

    for j in range(nt - 1):
        du = dt * (R * Ifun(t[j]) - (u[j] - urest)) / taum
        u[j + 1] = u[j] + du
        if reset and u[j + 1] > uth:
            u[j + 1] = urest

    return t, u


if __name__ == "__main__":
    # Parameters
    R = 1.0
    taum = 10.0
    taus = 2.0
    urest = 0.0
    uth = 1.0

    # Formulas from part (b)
    r = taus / taum
    tpeak = (taum * taus) / (taus - taum) * np.log(taus / taum)
    I0_crit = (uth / R) * (r ** (1.0 / (r - 1.0)))

    # Report tpeak and Icrit_0 to 4 digits
    print(f"tpeak: {tpeak:.4f} ms")
    print(f"Icrit_0: {I0_crit:.4f} mA\n")

    # Three amplitudes: below, at, and above Icrit_0
    amplitudes = [
        ("Below (0.5 Icrit_0)", 0.5 * I0_crit),
        ("At (1.0 Icrit_0)",    1.0 * I0_crit),
        ("Above (1.5 Icrit_0)", 1.5 * I0_crit),
    ]

    # Report table to four digits
    print(f"{'Condition':<24} {'I0 (mA)':<12} {'Theor umax (mV)':<18} {'Num umax (mV)':<16}")
    print("-" * 70)

    traces = []
    for label, I0 in amplitudes:
        Isyn = lambda t, I0=I0: I0 * np.exp(-t / taus)
        t, u = lif_sim(I=Isyn, T=30.0, dt=1e-4, taum=taum, R=R, urest=urest, uth=uth)
        umax_num = np.max(u)
        umax_theor = R * I0 * (r ** (-1.0 / (r - 1.0)))
        print(f"{label:<24} {I0:<12.4f} {umax_theor:<18.4f} {umax_num:<16.4f}")
        traces.append((label, I0, t, u))

    print("-" * 70 + "\n")

    # Plot u(t) for I0 below, at, and above Icrit_0 with horizontal threshold line
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, I0, t, u in traces:
        ax.plot(t, u, label=f"{label}: I0 = {I0:.4f} mA")
    ax.axhline(uth, color="k", linestyle="--", label=f"Threshold uth = {uth:.1f} mV")
    ax.set_xlim(0.0, 25.0)
    ax.set_xlabel("Time t (ms)")
    ax.set_ylabel("Membrane potential u (mV)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("hw1_p4_voltage_traces.png", dpi=300)
    plt.close(fig)

    # Plot Icrit_0 against taus over [0.5, 100] ms on log-log axes
    taus_arr = np.logspace(np.log10(0.5), np.log10(100.0), 300)
    r_arr = taus_arr / taum
    I0_crit_arr = (uth / R) * (r_arr ** (1.0 / (r_arr - 1.0)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(taus_arr, I0_crit_arr, "b-", lw=2)
    ax.set_xlim(0.5, 100.0)
    ax.set_xlabel("taus (ms)")
    ax.set_ylabel("I0_crit (mA)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("hw1_p4_loglog_critical_current.png", dpi=300)
    plt.close(fig)

    # Comments on the two limits
    print("=" * 70)
    print("Comment on the two limits:")
    print()
    print("1. Fast limit (taus -> 0):")
    print("   When the synaptic input is very short, the membrane does not have")
    print("   time to leak. The neuron needs a fixed amount of total charge to")
    print("   spike (Q = I0 * taus). Therefore, as taus decreases, the required")
    print("   peak current I0_crit must increase proportionally (I0_crit ~ 1/taus).")
    print("   This corresponds to a slope of -1 on the log-log plot.")
    print()
    print("2. Slow limit (taus -> inf):")
    print("   When the synaptic input decays very slowly, it acts like a constant")
    print("   DC current. The membrane potential reaches a steady state given by")
    print("   Ohm's law (u = R * I0). Therefore, the critical current approaches")
    print("   a constant value: I0_crit -> (uth - urest) / R = 1.0 mA.")
    print("   This corresponds to a slope of 0 (a flat line) on the log-log plot.")
    print("=" * 70)
