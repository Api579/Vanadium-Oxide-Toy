
# Vanadium Oxide (V₂O₅) Crossbar – Weight Storage and Inference

Simplified toy model, written in Python.

- **Device:** V₂O₅-based resistive memory cell, in two reported versions: abrupt SET (D1) and fully gradual switching (D2) [1]
- **Array:** crossbar of cells storing a weight matrix and computing matrix-vector products in place
- **Test application:** handwritten digit recognition (8 × 8 digits)

> **Note:** this is a toy model with deliberately limited scope. V₂O₅ shows bipolar resistive switching [1], [2], and retention up to 4500–6000 s has been reported [1]. Two V₂O₅ device types have been reported: one with abrupt SET and gradual RESET (D1), and an interface-engineered one with gradual SET and RESET (D2) [1]. The cell uses a phenomenological soft-bound update rule [5]. Pulse conditions come from the measured V₂O₅ devices [1]. Values that are not reported for V₂O₅ (the conductance window, the step size of the abrupt SET, and cycle-to-cycle noise) are swept, not claimed. The one exception is the gradual step size, which is derived from the reported 40-pulse trains under a stated assumption. Every parameter is listed under [Parameters](#parameters) with its source and status.

## Main question

How do the two measured V₂O₅ device types (abrupt vs gradual SET) and the unknown analog memory window affect how accurately weights can be programmed into a crossbar, and the resulting digit-classification accuracy?

## Scope

The project builds the hardware model first and uses a small classifier only at the end, as a test of how device behaviour affects a real task.

1. **Cell model** → 2. **Programming** → 3. **Crossbar compute** → 4. **Test application**

- **Stage 1 – Cell model:** conductance state, read, SET/RESET pulses, optional cycle-to-cycle noise. Presets D1 (abrupt SET) and D2 (gradual).
- **Stage 2 – Programming:** simplified write-verify loop, based on [4], for D1 and D2. First one cell, then an 8 × 8 array programmed to random targets. Report pulses needed and final error.
- **Stage 3 – Crossbar compute:** I = Gᵀ·V with differential pairs on an ideal circuit. Error against the exact NumPy product, as a function of the memory window.
- **Stage 4 – Test application:** program logistic-regression weights (trained in software) into a 65 × 10 differential crossbar and classify test digits. Compare crossbar and software accuracy.

**Sweeps** (one factor at a time, others at default; runs with noise are averaged over several random seeds):

| Factor | Values | Why swept |
|---|---|---|
| Device preset | D2; D1 with α_set = 0.3, 0.6, 0.9 | D1's SET is reported only as "abrupt", with no step size [1] |
| Memory window G_max/G_min | 3, 10, 30, 100 | No analog window reported for V₂O₅ [1] |
| Cycle-to-cycle noise σ | 0 (default), 0.05, 0.1, 0.2, 0.3 | Not reported for V₂O₅ [1], [2]; studied separately so the main D1/D2 and window results contain no assumed noise level |

## Output

Add plots here: D1 vs D2 pulse curves, programming error histograms, accuracy vs memory window and noise.

## Running

```bash
pip install numpy matplotlib scikit-learn
python vox_crossbar.py
```

---

## Governing equations

**Circuit**

| Equation | Form | Used for |
|---|---|---|
| Ohm's law | I = G·V | Current through one cell |
| Kirchhoff's current law | I_j = Σᵢ G_ij·V_i | Currents add along each column wire [6] |
| Matrix-vector product | I = Gᵀ·V | What the whole array computes, via Ohm's and Kirchhoff's laws [6]. Here inputs go on the rows and outputs are read on the columns, hence the transpose |
| Differential pair | W_ij ∝ G⁺_ij − G⁻_ij | Signed weights from two positive conductances [3] |
| Weight mapping | G± = G_min + (G_max − G_min)·max(±W, 0) / max\|W\| | Weight → target conductance |

**Cell (soft-bound model [5])**

For a normalised state w = (G − G_min)/(G_max − G_min), the soft-bound model gives δw₊ = α₊(1 − w)^γ for potentiation and δw₋ = −α₋·w^γ for depression; γ = 1 is the simple soft-bound case [5]. This project uses γ = 1 with separate SET and RESET parameters, because potentiation and depression are asymmetric in measured devices [1], [5].

| Equation | Form | Used for |
|---|---|---|
| SET pulse | G ← G + α_set(G_max − G) | Conductance increase, smaller steps near G_max [5] |
| RESET pulse | G ← G − α_reset(G − G_min) | Conductance decrease, smaller steps near G_min [5] |
| Cycle-to-cycle noise | α → α·(1 + σ·N(0,1)), same σ for SET and RESET | Random size of each step (noise sweep only) |
| Read vs write | reads (V_read) never change G; only ±1.1 V programming pulses do | Reads do not disturb stored weights |
| Bounds | G_min ≤ G ≤ G_max | Physical limits |

**Device presets**

| Preset | SET | RESET | Based on |
|---|---|---|---|
| D1 (Nb/V₂O₅/Ni) | Abrupt: large α_set (swept) | Gradual: small α_reset | Abrupt SET, gradual RESET [1] |
| D2 (Nb/NbOₓ/V₂O₅/Ni) | Gradual: small α_set | Gradual: small α_reset | Fully gradual switching [1] |

**Test application**

| Equation | Form | Used for |
|---|---|---|
| Class scores | z = Wᵀx + b | Logistic regression, trained in software [8] |
| Inputs | pixel values 0–16 scaled to 0–V_read; one extra input fixed at V_read for the bias | Voltages applied to the rows |
| Prediction | digit = argmax_j (I⁺_j − I⁻_j) | Largest differential column current |

Because the mapping from weights to conductance difference and from pixels to voltage are both linear, and the bias row uses the same scaling as the weights, the differential column currents are proportional to the class scores, so the argmax is unchanged on an ideal, perfectly programmed array. For the same reason, on the ideal circuit the results depend on the window G_max/G_min but not on the absolute values of G_min or V_read, which only scale the currents.

## Numerical methods

- **Write-verify programming (simplified from [4]):** read → compare with target → SET or RESET pulse → repeat, until within tolerance or a pulse limit. As in [4], the state is read after every pulse and the pulse polarity follows the sign of the error. Unlike [4], which uses ramps of increasing pulse amplitude, every pulse here has the fixed V₂O₅ amplitude of ±1.1 V [1].
- **Vectorised array operations** with NumPy [9].
- **Monte Carlo variability (noise sweep only):** Gaussian cycle-to-cycle noise on each pulse, averaged over several random seeds.
- **Off-chip training:** logistic regression with scikit-learn [8]; only inference runs on the crossbar.
- **Plotting:** Matplotlib [10].

## Assumptions and simplifications

- **Phenomenological cell:** no oxygen-vacancy, filament or thermal physics. Conductance follows the soft-bound rule of [5] with γ = 1.
- **Mixed-source parameters:** pulse conditions and device types from V₂O₅ [1]; read voltage and default conductance range from a TiO₂-based crossbar [3] and an HfO₂ device [5]; update rule from HfO₂ [5]. Each is marked in [Parameters](#parameters).
- **"Abrupt" is qualitative:** [1] describes D1's SET as abrupt in its switching characteristics but gives no step size. [1] also applies the 40-pulse potentiation/depression trains to both D1 and D2, and in that section describes both as gradual, which is not consistent with the abrupt SET. α_set for D1 is therefore swept; the D2 preset (α = 0.07) represents the fully gradual case.
- **Analog window unknown for V₂O₅:** [1] reports an ON/OFF ratio of ≈ 5 × 10² for D2 (not reported for D1), but no conductance values and not the range covered by the 40-pulse potentiation/depression trains [1]. The window used for weight storage is therefore swept from 3 (the analog window of the HfO₂ device in [5], about 3–4) to 100 (an ON/OFF ratio below 100 in the integrated TiO₂ crossbar [3]).
- **40 pulses span the window:** [1] uses 40-pulse potentiation and depression trains. It is assumed here that one train spans about 95 % of the window, which sets α for the gradual case.
- **Noise is not modelled by default:** no cycle-to-cycle variability is reported for V₂O₅ [1], [2]. Noise is studied only in a separate sweep, with one relative σ for both SET and RESET. (In HfO₂, [5] reports fairly high dispersion of the step size, but its values are absolute widths of its own fitted α and cannot be transferred to this model.)
- **Non-volatile, no drift:** G stays constant between operations. Retention is reported [1], but no drift rate.
- **Linear read, no read disturb:** a cell behaves as a fixed resistor at the read voltage, and reads never change G. No switching threshold is modelled.
- **Ideal circuit:** zero wire resistance, ideal drivers, columns at virtual ground, ideal current measurement, DC steady state.
- **Off-chip training, single layer:** one 65 × 10 weight matrix (64 pixels + bias), stored as 130 × 10 cells.

## Parameters

| Parameter | Value | Source | Status |
|---|---|---|---|
| SET / RESET pulse | +1.1 V / −1.1 V, 1 ms | [1] | Reported (V₂O₅ LTP/LTD pulse trains, applied to both D1 and D2) |
| Pulses per train | 40 | [1] | Reported |
| Read voltage | 0.1 V | [3], [5] | Reported for the TiO₂ crossbar [3] and HfO₂ device [5]. [1] does not state a read voltage for its potentiation/depression or endurance measurements (a 0.5 V read pulse is used in its STDP tests). On the ideal circuit the value only scales the currents |
| G_min (default) | 10 µS | [3] | Reported: conductances in [3] range from 10 to 100 µS. On the ideal circuit the value only scales the currents |
| Memory window G_max/G_min | 10 (default); swept 3, 10, 30, 100 | [3], [5] | Default from the 10–100 µS range in [3]; analog range not reported for V₂O₅ [1] |
| α_set, α_reset (D2) | 0.07, 0.07 | [1] | Derived: one 40-pulse train covers 95 % of the window, α = 1 − 0.05^(1/40) ≈ 0.072 (assumption that the train spans the window) |
| α_set (D1) | 0.3, 0.6, 0.9 (swept) | – | Assumption: [1] reports "abrupt" SET with no step size |
| α_reset (D1) | 0.07 | [1] | Gradual RESET; same derivation and same assumption as D2 |
| Cycle-to-cycle noise σ (relative, same for SET and RESET) | 0 (default); swept 0.05–0.3 | – | Choice: not reported for V₂O₅ [1], [2] |
| Soft-bound exponent γ | 1 | [5] | γ = 1 is the simple soft-bound case; [5] finds γ > 1 in the analog region, not modelled here |
| Programming tolerance | 1 % | [4] | Target only: [4] reached 1 % on single Pt/TiO₂ devices using increasing-amplitude ramps. Whether 1 % is reachable with fixed ±1.1 V pulses is one of the results |
| Max pulses per cell | 100 | – | Choice |
| Array size (Stages 2–3) | 8 × 8 | – | Choice |
| Array size (Stage 4) | 65 × 10 (× 2 differential) | – | 64 pixels + bias, 10 digits |
| Dataset | 1797 images, 8 × 8, pixels 0–16 | [7], [8] | scikit-learn `load_digits`, a copy of the test set of the UCI dataset [7]. 80 % train / 20 % test split is a choice |

## Results to report

- D1 vs D2 pulse curves (G vs pulse number)
- Programming D1 vs D2: pulses per cell, final error histogram, overshoot
- Crossbar output vs exact product, error vs memory window
- Digit accuracy: software vs crossbar, for D1 vs D2 and vs memory window (no noise), then vs noise
- Which factor hurts accuracy most

## Discussion points (not modelled)

- **Retention:** retention is reported up to 6000 s for D1 and 4500 s for D2 [1] (about 1.25–1.7 h), with no drift rate and no failure criterion given. Whether a V₂O₅ crossbar would hold its weights for longer is therefore unknown.
- **Endurance:** reported endurance is 7000 cycles for D1 and 5000 cycles for D2 (the abstract says "beyond 7000" for D2) [1]. How partial programming pulses count against these switching cycles is not reported, so wear is not modelled.
- **Device maturity:** an earlier ITO/V₂O₅ device showed destructive changes of the surface topography after about 10 cycles [2], compared with thousands of cycles for the interface-engineered devices [1].
- **Benchmark caution:** the network accuracy reported for the V₂O₅ device in [1] differs between the abstract (≈ 86.75 %) and the main text (95.5 %), both on MNIST, so it is not used as a benchmark.

## Out of scope

Left out to keep every result traceable to a source or a clearly labelled sweep:

- **Device-to-device spread:** no numerical value reported for V₂O₅ [1].
- **Switching threshold and read disturb:** no threshold value is reported for the V₂O₅ devices [1].
- **Asymmetric SET/RESET noise:** no V₂O₅ data.
- **State-dependent nonlinearity (γ > 1):** reported for HfO₂ [5], not for V₂O₅.
- **Wire resistance and sneak paths:** would need an array model beyond the ideal circuit.
- **ADC/DAC quantisation and peripheral circuits.**
- **Conductance drift:** no V₂O₅ drift data.
- **Pulse-amplitude and pulse-width dependence of the step size.**
- **Training on the crossbar itself (in-situ learning).**
- **Atomistic or DFT simulation, and VO₂ insulator–metal (volatile) switching.**

## References

[1] K. Kaushlendra and D. Kaur, "Interface engineered V₂O₅-based flexible memristors towards high-performance brain-inspired neuromorphic computing," *Nanoscale Horiz.*, vol. 11, no. 3, pp. 763–777, 2026, doi: 10.1039/D5NH00562K.

[2] Z. Wan, R. B. Darling, A. Majumdar, and M. P. Anantram, "A forming-free bipolar resistive switching behavior based on ITO/V₂O₅/ITO structure," *Appl. Phys. Lett.*, vol. 111, no. 4, Art. no. 041601, 2017, doi: 10.1063/1.4995411.

[3] M. Prezioso, F. Merrikh-Bayat, B. D. Hoskins, G. C. Adam, K. K. Likharev, and D. B. Strukov, "Training and operation of an integrated neuromorphic network based on metal-oxide memristors," *Nature*, vol. 521, no. 7550, pp. 61–64, 2015, doi: 10.1038/nature14441.

[4] F. Alibart, L. Gao, B. D. Hoskins, and D. B. Strukov, "High precision tuning of state for memristive devices by adaptable variation-tolerant algorithm," *Nanotechnology*, vol. 23, no. 7, Art. no. 075201, 2012, doi: 10.1088/0957-4484/23/7/075201.

[5] J. Frascaroli, S. Brivio, E. Covi, and S. Spiga, "Evidence of soft bound behaviour in analogue memristive devices for neuromorphic computing," *Sci. Rep.*, vol. 8, Art. no. 7178, 2018, doi: 10.1038/s41598-018-25376-x.

[6] D. Ielmini and H.-S. P. Wong, "In-memory computing with resistive switching devices," *Nat. Electron.*, vol. 1, no. 6, pp. 333–343, 2018, doi: 10.1038/s41928-018-0092-2.

[7] E. Alpaydin and C. Kaynak, "Optical recognition of handwritten digits," UCI Machine Learning Repository, 1998, doi: 10.24432/C50P49.

[8] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[9] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.

[10] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.

