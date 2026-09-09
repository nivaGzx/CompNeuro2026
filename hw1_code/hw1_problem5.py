import numpy as np
import matplotlib.pyplot as plt
from hh_sim import hh_sim


def find_rheobase(gK=36.0, tol=1e-4):
    """Find rheobase using bisection on constant injected current Id."""
    low, high = 0.0, 15.0
    while (high - low) > tol:
        mid = (low + high) / 2.0
        _, _, spk, rate = hh_sim(Id=mid, gK=gK)
        # Criterion from hh_sim: steady rate > 0 (>= 2 spikes in steady-state window)
        if rate > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


if __name__ == "__main__":
    # =========================================================================
    # Problem 5(a): Locating Rheobase via Bisection (gK = 36)
    # =========================================================================
    print("=" * 70)
    print("PROBLEM 5(a): Rheobase and Voltage Traces")
    print("=" * 70)

    rheobase_36 = find_rheobase(gK=36.0, tol=1e-4)

    # Voltage traces just below and just above rheobase
    Id_below = 6.25
    Id_above = 6.27

    t, V_below, spk_below, rate_below = hh_sim(Id=Id_below)
    t, V_above, spk_above, rate_above = hh_sim(Id=Id_above)

    # Plot 5(a) voltage traces with identical axes limits
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

    axes[0].plot(t, V_below, color="tab:blue", lw=1.2)
    axes[0].set_title(f"Just below rheobase: Id = {Id_below:.2f} uA/cm^2 (Subthreshold steady state, rate = {rate_below:.1f} Hz)")
    axes[0].set_ylabel("Membrane potential V (mV)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(t, V_above, color="tab:red", lw=1.2)
    axes[1].set_title(f"Just above rheobase: Id = {Id_above:.2f} uA/cm^2 (Sustained repetitive firing, rate = {rate_above:.1f} Hz)")
    axes[1].set_xlabel("Time t (ms)")
    axes[1].set_ylabel("Membrane potential V (mV)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[0].set_xlim(0, 300)
    axes[0].set_ylim(-20, 120)

    fig.tight_layout()
    fig.savefig("hw1_p5a_voltage_traces.png", dpi=300)
    plt.close(fig)

    # Print answers for 5(a)
    print(f"1. Rheobase value: {rheobase_36:.3f} uA/cm^2 (or {rheobase_36:.2f} uA/cm^2 to 3 digits).")
    print()
    print("2. Criterion for 'repetitive firing':")
    print("   - Discard the first 40% of time (t <= 120 ms) to ignore onset transients.")
    print("   - Require steady firing rate > 0 (at least 2 spikes in [120, 300] ms).")
    print()
    print("3. Voltage traces comparison:")
    print(f"   - Just below ({Id_below:.2f} uA/cm^2): transient oscillations die out,")
    print(f"     settling into a constant resting state (rate = {rate_below:.1f} Hz).")
    print(f"   - Just above ({Id_above:.2f} uA/cm^2): sustained periodic action potentials")
    print(f"     fire non-stop through the whole trace (rate = {rate_above:.1f} Hz).")
    print("   - Saved figure to: hw1_p5a_voltage_traces.png")
    print("=" * 70 + "\n")

    # =========================================================================
    # Problem 5(b): f-I Curve and Onset Rate (gK = 36)
    # =========================================================================
    print("=" * 70)
    print("PROBLEM 5(b): f-I Curve, Onset Rate, and Bifurcation Type")
    print("=" * 70)

    I_vals_36 = np.linspace(6.26, 13.0, 20)
    rates_36 = np.array([hh_sim(Id=I, gK=36.0)[3] for I in I_vals_36])
    onset_rate_36 = rates_36[0]

    # Plot 5(b) f-I curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([5.0, 6.25], [0.0, 0.0], "b--", lw=1.5, label="Quiescent (f = 0)")
    ax.plot(I_vals_36, rates_36, "b-o", markersize=4, lw=1.5, label="f-I curve (gK = 36 mS/cm^2)")
    ax.plot(I_vals_36[0], onset_rate_36, "ro", markersize=7, label=f"Onset rate: {onset_rate_36:.1f} Hz")
    ax.axvline(rheobase_36, color="gray", linestyle=":", label=f"Rheobase ({rheobase_36:.2f} uA/cm^2)")

    ax.set_xlabel("Injected current Id (uA/cm^2)")
    ax.set_ylabel("Firing rate f (Hz)")
    ax.set_title("Hodgkin-Huxley f-I Curve (Problem 5b)")
    ax.set_xlim(5.0, 13.5)
    ax.set_ylim(-2, 85)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig("hw1_p5b_fi_curve.png", dpi=300)
    plt.close(fig)

    # Print answers for 5(b)
    print(f"1. Current range evaluated: [{I_vals_36[0]:.2f}, {I_vals_36[-1]:.2f}] uA/cm^2 (~2x rheobase).")
    print(f"2. Onset firing rate: {onset_rate_36:.1f} Hz (to 3 digits: {onset_rate_36:.3f} Hz).")
    print()
    print("3. Does rate rise continuously from zero, or switch on at a nonzero value?")
    print("   - The firing rate switches on abruptly at a nonzero value (~49.3 Hz).")
    print("   - It does NOT rise continuously from zero.")
    print()
    print("4. Which bifurcation scenario does Hodgkin-Huxley resemble?")
    print("   - It resembles Problem 2 (FitzHugh-Nagumo / Hopf bifurcation).")
    print("   - In Problem 2, the limit cycle is born with a nonzero natural frequency,")
    print("     so firing starts abruptly at a high frequency (Class 2 excitability).")
    print("   - In contrast, Problem 3 (Theta neuron / SNIC bifurcation) exhibits Class 1")
    print("     excitability, where firing rate starts continuously from zero (f ~ sqrt(I - Ic)).")
    print()
    print("5. Physiological consequence:")
    print("   - The neuron has a discontinuous onset frequency (Class 2 excitability).")
    print("   - It cannot encode weak inputs using very low firing rates (such as 1 to 5 Hz).")
    print("   - Once the threshold current is reached, it immediately fires at a high frequency (~50 Hz).")
    print("   - Saved figure to: hw1_p5b_fi_curve.png")
    print("=" * 70 + "\n")

    # =========================================================================
    # Problem 5(c): Reducing gK to 30 mS/cm^2 and Overlaying Curves
    # =========================================================================
    print("=" * 70)
    print("PROBLEM 5(c): Changing gK to 30 mS/cm^2 and Comparison")
    print("=" * 70)

    rheobase_30 = find_rheobase(gK=30.0, tol=1e-4)

    # Compute f-I curve for gK = 30 from just above rheobase to 13.0
    I_vals_30 = np.linspace(2.78, 13.0, 25)
    rates_30 = np.array([hh_sim(Id=I, gK=30.0)[3] for I in I_vals_30])
    onset_rate_30 = rates_30[0]

    # Overlay plot
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Quiescent lines
    ax.plot([1.5, 6.25], [0.0, 0.0], color="tab:blue", linestyle="--", lw=1.5)
    ax.plot([1.5, 2.77], [0.0, 0.0], color="tab:orange", linestyle="--", lw=1.5)

    # Two curves
    ax.plot(I_vals_36, rates_36, "b-o", markersize=4, lw=1.5, label="gK = 36 mS/cm^2 (Control)")
    ax.plot(I_vals_30, rates_30, color="tab:orange", marker="s", markersize=4, lw=1.5, label="gK = 30 mS/cm^2 (Reduced gK)")

    # Vertical rheobase markers
    ax.axvline(rheobase_36, color="blue", linestyle=":", lw=1.5, label=f"Rheobase (gK=36): {rheobase_36:.2f} uA/cm^2")
    ax.axvline(rheobase_30, color="tab:orange", linestyle=":", lw=1.5, label=f"Rheobase (gK=30): {rheobase_30:.2f} uA/cm^2")

    # Onset points
    ax.plot(I_vals_36[0], onset_rate_36, "ro", markersize=7, zorder=5)
    ax.plot(I_vals_30[0], onset_rate_30, "ro", markersize=7, zorder=5)

    ax.set_xlabel("Injected current Id (uA/cm^2)")
    ax.set_ylabel("Firing rate f (Hz)")
    ax.set_title("Hodgkin-Huxley f-I Curves for gK = 36 vs gK = 30 mS/cm^2")
    ax.set_xlim(1.5, 13.5)
    ax.set_ylim(-2, 90)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig("hw1_p5c_fi_overlay.png", dpi=300)
    plt.close(fig)

    # Print answers for 5(c)
    print("1. Table of Rheobase values (reported to 3 digits):")
    print(f"   {'Condition':<20} {'gK (mS/cm^2)':<16} {'Rheobase (uA/cm^2)':<20}")
    print("   " + "-" * 56)
    print(f"   {'Control':<20} {'36.0':<16} {f'{rheobase_36:.3f} (or {rheobase_36:.2f})':<20}")
    print(f"   {'Reduced gK':<20} {'30.0':<16} {f'{rheobase_30:.3f} (or {rheobase_30:.2f})':<20}")
    print("   " + "-" * 56)
    print()
    print("2. Did rheobase go up or down?")
    print("   - The rheobase went DOWN significantly (from 6.26 to 2.77 uA/cm^2).")
    print()
    print("3. Explanation in terms of currents:")
    print("   - In the HH model, inward sodium current (INa) depolarizes the membrane and creates spikes,")
    print("     while outward potassium current (IK) hyperpolarizes the membrane and opposes spike initiation.")
    print("   - Lowering gK from 36 to 30 mS/cm^2 reduces the outward potassium current.")
    print("   - Because the opposing outward current is smaller, a smaller injected current Id is needed")
    print("     for INa to overcome IK and trigger repetitive firing.")
    print("   - Therefore, the neuron is more excitable, and the rheobase decreases.")
    print()
    print("4. Did the shape change or only its position?")
    print("   - The shape is preserved: both curves start at a nonzero frequency (~45-50 Hz)")
    print("     and increase sublinearly with injected current (Class 2 excitability).")
    print("   - Only the position changed: the curve shifted horizontally to the left by about 3.5 uA/cm^2.")
    print("   - Saved figure to: hw1_p5c_fi_overlay.png")
    print("=" * 70)
