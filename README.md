# CompNeuro2026

Computational Neuroscience (APPM 4370 / 5370) Coursework and Homework Code.

**Author / GitHub Account:** [nivaGzx](https://github.com/nivaGzx)  
**Repository URL:** [https://github.com/nivaGzx/CompNeuro2026](https://github.com/nivaGzx/CompNeuro2026)

---

## Repository Structure

### hw1_code/ (Homework 1)
Runnable Python scripts and output figures, labeled by problem:

- **hw1_problem4.py (Problem 4: Synaptic Pulse Driven LIF Neuron)**
  - Simulates the Leaky Integrate-and-Fire (LIF) model driven by an exponentially decaying synaptic current (t) = I_0 e^{-t/\tau_s}$.
  - Verifies theoretical {\mathrm{peak}}$ and critical current ^{\mathrm{crit}}$ against numerical integration.
  - Plots membrane voltage traces and log-log analysis of critical current across $\tau_s$.
  - Output figures: hw1_p4_voltage_traces.png, hw1_p4_loglog_critical_current.png.

- **hw1_problem5.py (Problem 5: Hodgkin-Huxley Rheobase & -I$ Curve)**
  - **Part (a)**: Locates the rheobase to 3 digits using bisection method, states the criterion for repetitive firing, and plots voltage traces just below ( = 6.25\ \mu\mathrm{A/cm^2}$) and just above ( = 6.27\ \mu\mathrm{A/cm^2}$) on identical axes.
  - **Part (b)**: Computes the -I$ curve from rheobase up to $\approx 2\times$ rheobase, reports onset firing rate, and classifies the bifurcation (Class 2 / Hopf bifurcation).
  - **Part (c)**: Recomputes rheobase for reduced potassium conductance ( = 30\ \mathrm{mS/cm^2}$), provides comparison table, and overlays both -I$ curves on a single set of axes.
  - Output figures: hw1_p5a_voltage_traces.png, hw1_p5b_fi_curve.png, hw1_p5c_fi_overlay.png.

- **hh_sim.py**
  - Classical 4D Hodgkin-Huxley model integrator under constant current injection.

---

## Instructions to Run

Navigate to the problem folder and run with Python:

`ash
cd hw1_code
python hw1_problem4.py
python hw1_problem5.py
`
