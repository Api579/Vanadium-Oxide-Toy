# Vanadium Oxide (VOₓ) Crossbar – Weight Storage and In-Memory Computing

Simplified toy model, written in Python.

- **Device:** V₂O₅-based resistive memory cell (non-volatile, analog conductance switching)
- **Array:** crossbar of VOₓ cells storing a weight matrix and computing matrix-vector products in place
- **Test application:** handwritten digit recognition (8 × 8 digits dataset)

> **Note:** this is a toy model. V₂O₅ shows non-volatile bipolar resistive switching [1], [2]. Two V₂O₅ device types have been reported: one with abrupt SET and gradual RESET (D1), and an interface-engineered one with gradual SET and RESET (D2) [1]. The device here uses a phenomenological soft-bound update rule [5]. Pulse conditions are taken from measured V₂O₅ synaptic devices [1]. The default conductance window, read voltage and switching threshold come from a comparable metal-oxide crossbar [3], because no V₂O₅ analog conductance range is reported, so the window is swept. Variability is swept, not claimed. See [Parameters](#parameters) for the source and status of every value.

---

## Main question

**How do the two measured V₂O₅ device types (abrupt vs gradual SET) and the unknown analog memory window affect weight-storage accuracy and digit classification on a crossbar?**

The project builds the hardware model first: one cell, then programming, then the array. A small digit classifier is used only at the end as a test application, to measure how device behaviour affects a real task.

## Project stages

```
1. VOₓ cell model  →  2. Programming  →  3. Crossbar compute  →  4. Test application (digits)
   (D1 / D2 presets)    (D1 vs D2)          (memory-window sweep)   (accuracy vs device type,
                                                                     window, noise, spread)
                                                                 5. (optional) real wires
```

- [ ] **Stage 1 – VOₓ cell model:** conductance state, read, asymmetric SET/RESET pulses with threshold, separate cycle-to-cycle noise per direction; **D1 (abrupt SET)** and **D2 (gradual)** presets; optional exponent γ
- [ ] **Stage 2 – Programming:** write-verify loop for both D1 and D2; one cell, then an 8 × 8 array to random targets; pulses needed, final error, endurance budget used
- [ ] **Stage 3 – Crossbar compute:** I = Gᵀ·V with differential pairs; error vs exact NumPy product; **memory-window sweep** (G_max/G_min = 3, 10, 30, 100); device-to-device spread and read noise
- [ ] **Stage 4 – Test application:** program logistic-regression weights into a 65 × 10 crossbar pair; classify test digits; crossbar vs software accuracy for **D1 vs D2**, and vs window, noise and spread
- [ ] **Stage 5 (optional) – Real array:** wire resistance and sneak paths via nodal analysis

## Output

![Crossbar output](<crossbar_output.png>)

*Add your plots here, e.g. D1 vs D2 pulse curves, programming error histograms, accuracy vs memory window.*

## Running

```bash
pip install numpy scipy matplotlib scikit-learn
python vox_crossbar.py
```

---

## Governing equations

**Circuit physics**

| Equation | Form | Used for |
|---|---|---|
| Ohm's law | I = G·V | Current through one cell |
| Kirchhoff's current law | I_j = Σᵢ G_ij·V_i | Currents add along each column wire |
| Matrix-vector product | I = Gᵀ·V | What the whole array computes [6] |
| Differential pair | W_ij ∝ G⁺_ij − G⁻_ij | Signed weights from two positive conductances [3] |
| Weight mapping | G± = G_min + (G_max − G_min)·max(±W, 0) / max\|W\| | Weight → target conductance |

**Device model (VOₓ cell)**

The soft-bound model [5] for a normalised weight w = (G − G_min)/(G_max − G_min) is
δw₊ = α₊(1 − w)^γ (potentiation) and δw₋ = −α₋·w^γ (depression).
This project uses **γ = 1** by default, with **separate SET and RESET parameters** because potentiation and depression behave differently in measured devices [1], [5]:

| Equation | Form | Used for |
|---|---|---|
| SET pulse (potentiation) | G ← G + α_set(G_max − G) | Conductance increase, smaller steps near G_max [5] |
| RESET pulse (depression) | G ← G − α_reset(G − G_min) | Conductance decrease, smaller steps near G_min [5] |
| Cycle-to-cycle noise | α → α·(1 + σ·N(0,1)), with σ_set and σ_reset | Random size of each step, per direction [5] |
| Switching threshold | pulse changes G only if \|V\| > V_th | Reads never disturb stored weights |
| Conductance bounds | G_min ≤ G ≤ G_max | Physical limits of the device |
| Optional nonlinearity | δw₊ = α_set(1 − w)^γ, δw₋ = −α_reset·w^γ, γ > 1 | Stronger state dependence, reported in the analog region [5] |

**Device presets**

| Preset | SET behaviour | RESET behaviour | Based on |
|---|---|---|---|
| **D1** (Nb/V₂O₅/Ni) | Abrupt: large α_set | Gradual: small α_reset | Abrupt SET, gradual RESET [1] |
| **D2** (Nb/NbOₓ/V₂O₅/Ni) | Gradual: small α_set | Gradual: small α_reset | Fully gradual switching [1] |

**Test application**

| Equation | Form | Used for |
|---|---|---|
| Class scores | z = Wᵀx + b | Linear classifier (logistic regression, trained off-chip [9]) |
| Prediction | digit = argmax_j (I⁺_j − I⁻_j) | Class from the largest column current difference |

## Numerical methods and techniques

- **Write-verify programming:** iterative closed loop (read → compare with target → SET or RESET pulse → repeat) until within tolerance or a pulse limit [4].
- **Vectorised array operations** (NumPy [10]) to read the whole array and classify all test images at once.
- **Monte Carlo variability:** Gaussian cycle-to-cycle noise on each pulse (separate for SET and RESET) and device-to-device spread per cell, averaged over several random seeds.
- **Parameter sweeps:** accuracy measured while varying one factor at a time: device preset (D1/D2), memory window G_max/G_min, noise level, device spread, programming tolerance, and optionally γ.
- **Off-chip training:** logistic regression (scikit-learn [9]) gives the weights; only inference runs on the crossbar.
- **(Stage 5)** **Nodal analysis:** Kirchhoff's current law at every node gives a sparse linear system, solved by **LU decomposition** (`scipy.sparse.linalg.splu` [11]) [7].
- **Plotting:** Matplotlib [12].

## Assumptions and simplifications

- **Phenomenological device model:** no oxygen-vacancy, filament or thermal physics; conductance follows the soft-bound update rule of [5], γ = 1 by default.
- **Mixed-source parameters:** pulse conditions and device types from V₂O₅ [1]; default conductance window, read voltage and threshold from a TiO₂-based metal-oxide crossbar [3]; update rule, SET/RESET asymmetry and noise guidance from HfO₂ devices [5].
- **"Abrupt" is qualitative:** [1] describes D1's SET as abrupt but gives no step size, so α_set for D1 is an assumption chosen to reach most of the window in very few pulses.
- **Analog memory window unknown for V₂O₅:** the reported V₂O₅ ON/OFF ratio (5 × 10²) is the full switching range [1], not the gradual analog range; the window is therefore swept (3–100), covering the analog window seen in HfO₂ (~3–4) [5] and a metal-oxide crossbar (<100) [3].
- **Non-volatile, no drift:** G stays constant between operations. Measured V₂O₅ retention is 4500–6000 s [1], but no drift rate is reported, so drift is not modelled.
- **Variability not claimed for V₂O₅:** cycle-to-cycle and device-to-device spread are not reported numerically in the V₂O₅ sources [1], [2]; noise levels follow the step-size spread measured in HfO₂ [5] and are swept.
- **Linear I–V when reading:** a cell behaves as a fixed resistor at the read voltage.
- **Ideal circuit (Stages 1–4):** zero wire resistance, ideal drivers, columns held at 0 V (virtual ground), ideal current measurement.
- **DC steady state:** no capacitance, timing or transient effects.
- **No selector device:** sneak paths studied only in Stage 5.
- **Off-chip training, single layer:** one 65 × 10 weight matrix (64 pixels + bias), stored as 130 × 10 cells.

## Parameters

| Parameter | Value | Source | Status |
|---|---|---|---|
| Read voltage | 0.1 V | [3], [5] | Reported (TiO₂ crossbar; HfO₂ device) |
| SET / RESET pulse | +1.1 V / −1.1 V, 1 ms | [1] | Reported (V₂O₅ synaptic device) |
| Switching threshold V_th | 0.8 V | [3] | Reported range 0.8–1.2 V; between V₂O₅ read (0.5 V) and pulse (1.1 V) levels [1] |
| G_min (default) | 10 µS | [3] | Reported (TiO₂ crossbar) |
| Memory window G_max/G_min | 10 (default); swept 3, 10, 30, 100 | [1], [3], [5] | V₂O₅ analog range not reported; sweep spans HfO₂ (~3–4) to metal-oxide crossbar (<100) |
| Pulses to cross full range (gradual) | ~40 | [1] | Reported pulse-train length |
| α_set, α_reset (D2, gradual) | 0.07, 0.07 | – | Derived: 40 pulses cover 95 % of range, α = 1 − 0.05^(1/40) |
| α_set (D1, abrupt) | 0.6 | – | Assumption: reaches ~85 % of range in 2 pulses |
| α_reset (D1) | 0.07 | [1] | Gradual RESET, same derivation as D2 |
| Cycle-to-cycle noise σ_set / σ_reset | 0.2 / 0.3 (swept 0.05–0.3) | [5] | α spread ~0.2 (potentiation), ~0.3 (depression) in HfO₂; not reported for V₂O₅ |
| Soft-bound exponent γ | 1 (optional sweep 1–3) | [5] | γ > 1 reported in the analog region |
| Device-to-device spread | 0–20 % (swept) | – | Assumption, not reported numerically |
| Programming tolerance | 1–2 % | [4] | 1 % demonstrated |
| Max pulses per cell | 100 | – | Choice; compared against endurance below |
| Endurance (for budget) | 5000–7000 cycles | [1] | Reported (V₂O₅ D1/D2; D2 value inconsistent in source) |
| Wire resistance (Stage 5) | ~50–70 Ω per segment | [3] | Derived from 600–800 Ω per 12-cell line |
| Array size (Stages 1–3) | 8 × 8 | – | Choice |
| Array size (Stage 4) | 65 × 10 (× 2 differential) | – | 64 pixels + bias, 10 digits |
| Dataset | 1797 images, 8 × 8, pixels 0–16 | [8] | 80 % train / 20 % test |

## Results to report

- D1 vs D2 SET/RESET pulse curves (G vs pulse number)
- Programming for D1 vs D2: pulses per cell, final error histogram, target vs achieved G, overshoot
- Endurance budget: pulses used per full programming vs reported endurance [1]
- Crossbar output vs exact NumPy product, and error vs memory window
- Digit accuracy: software vs crossbar, for D1 vs D2, and vs memory window, noise and device spread
- Which factor hurts accuracy most

## Discussion points (not modelled)

- **Retention:** measured V₂O₅ retention is 4500–6000 s [1] (about 1–1.7 h), so a real V₂O₅ crossbar may need frequent reprogramming.
- **Device maturity:** an earlier ITO/V₂O₅ device degraded after about 10 cycles [2], compared with thousands of cycles for the interface-engineered devices [1].
- **Benchmark caution:** the reported V₂O₅ network accuracy is inconsistent within [1] (95.5 % in the text, ~86.75 % in the abstract), so it is not used as a benchmark.

## Out of scope

- Atomistic / DFT simulation of the vanadium oxide structure
- VO₂ insulator–metal transition (volatile switching)
- Conductance drift over time (no V₂O₅ data)
- Pulse-amplitude and pulse-width dependence of the step size
- Training on the crossbar itself (in-situ learning)
- Peripheral electronics (ADC/DAC, sense amplifiers)

---

## References

[1] K. Kaushlendra and D. Kaur, "Interface engineered V₂O₅-based flexible memristors towards high-performance brain-inspired neuromorphic computing," *Nanoscale Horiz.*, vol. 11, no. 3, pp. 763–777, 2026, doi: 10.1039/D5NH00562K.

[2] Z. Wan, R. B. Darling, A. Majumdar, and M. P. Anantram, "A forming-free bipolar resistive switching behavior based on ITO/V₂O₅/ITO structure," *Appl. Phys. Lett.*, vol. 111, no. 4, Art. no. 041601, 2017, doi: 10.1063/1.4995411.

[3] M. Prezioso, F. Merrikh-Bayat, B. D. Hoskins, G. C. Adam, K. K. Likharev, and D. B. Strukov, "Training and operation of an integrated neuromorphic network based on metal-oxide memristors," *Nature*, vol. 521, no. 7550, pp. 61–64, 2015, doi: 10.1038/nature14441.

[4] F. Alibart, L. Gao, B. D. Hoskins, and D. B. Strukov, "High precision tuning of state for memristive devices by adaptable variation-tolerant algorithm," *Nanotechnology*, vol. 23, no. 7, Art. no. 075201, 2012.

[5] J. Frascaroli, S. Brivio, E. Covi, and S. Spiga, "Evidence of soft bound behaviour in analogue memristive devices for neuromorphic computing," *Sci. Rep.*, vol. 8, Art. no. 7178, 2018, doi: 10.1038/s41598-018-25376-x.

[6] D. Ielmini and H.-S. P. Wong, "In-memory computing with resistive switching devices," *Nat. Electron.*, vol. 1, no. 6, pp. 333–343, 2018, doi: 10.1038/s41928-018-0092-2.

[7] A. Chen, "A comprehensive crossbar array model with solutions for line resistance and nonlinear device characteristics," *IEEE Trans. Electron Devices*, vol. 60, no. 4, pp. 1318–1326, 2013.

[8] E. Alpaydin and C. Kaynak, "Optical recognition of handwritten digits," UCI Machine Learning Repository, 1998, doi: 10.24432/C50P49.

[9] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[10] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.

[11] P. Virtanen *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nat. Methods*, vol. 17, no. 3, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2.

[12] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.
