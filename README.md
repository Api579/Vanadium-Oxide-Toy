# Vanadium Oxide (V₂O₅) Crossbar Array.

Simplified model of Vanadium Oxide Crossbar Array in python.

- **Device:** V₂O₅-based memory cell, based on two reports [1]: Device 1,D1 , (Nb/V₂O₅/Ni) with an abrupt SET and gradual RESET, and Device 2, D2, (Nb/NbOₓ/V₂O₅/Ni) with gradual SET and gradual RESET.
- **Array:** crossbar storing weights and calculates vector products.
- **Test application:** handwritten digit recognition using 8 by 8 digits.

> **Note:** Pulse conditions come from the V₂O₅ devices in one of the cited articles [1], while cell update rule comes from an HfO₂ device article [4]. Values not for V₂O₅ in cited articles are derived from other devices.

## Main question

How the 2 V₂O₅ device types (abrupt vs gradual SET) and the unknown ranges of memory values it can store, affects the devices abilities to be used as a system of storage of numeric values?

## Stages

1. **Cell model:** initialise the conductance state, read, SET/RESET pulses values, and presets of D1 and D2.
2. **Programming:** write a verify loop simplified from a cited article [3] for D1 and D2. First for one cell, and then the 8 × 8 array programmed to random targets, and report pulses needed and final error.
3. **Crossbar compute:** I = Gᵀ·V with differential pairs assume on ideal circuit, using arrays programmed in step/stage 2. Error against the exact NumPy product, as a function of the memory window.
4. **Test application:** program logistic-regression weights (trained in software) into a 65 × 10 differential crossbar and classify test digits. Compare crossbar and software accuracy.

**Sweeps** (one factor at a time; defaults are D2 and a window of 10):

| Factor | Values | Why swept |
|---|---|---|
| Device preset | D2 (default); D1 with α_set = 0.3, 0.6, 0.9 | [1] describes D1's SET as abrupt but gives no step size |
| Memory window G_max/G_min | 3, 10 (default), 30, 100 | [1] does not report the conductance range of its pulse trains |

The model is deterministic: assumes no noise, or randomness, so each configuration needs only be run once.

## Running

```bash
pip install numpy matplotlib scikit-learn
python vox_crossbar.py
```

## Output

Add plots here later: D1 V D2 pulse curves, and programming errors histograms, plus accuracy vs memory window.

---

## Governing equations

**Circuit**

| Equation | Form | Used for |
|---|---|---|
| Ohm's law | I = G·V | Current through one cell |
| Kirchhoff's current law | I_j = Σᵢ G_ij·V_i | Currents add along each column |
| Matrix-vector product | I = Gᵀ·V | What the array computes [5]. Inputs on rows, outputs on columns, hence the transpose |
| Differential pair | W_ij ∝ G⁺_ij − G⁻_ij | Signed weights from two positive conductances, as in [2] |
| Weight mapping | G± = G_min + (G_max − G_min)·max(±W, 0) / max\|W\| | Weight → target conductance; max\|W\| is taken over the whole matrix, bias row included |

**Cell**

[4] models the conductance update with a soft-bound rule. For a normalised state w = (G − G_min)/(G_max − G_min), it gives δw₊ = α(1 − w)^γ for potentiation and δw₋ = −α·w^γ for depression [4]. This project uses γ = 1 and, as its own modelling choice, separate step sizes α_set and α_reset. In G, this becomes:

| Equation | Form |
|---|---|
| SET pulse | G ← G + α_set(G_max − G) |
| RESET pulse | G ← G − α_reset(G − G_min) |
| Read | reads never change G; only programming pulses do |

**Device presets**

| Preset | α_set | α_reset | Based on |
|---|---|---|---|
| D1 (Nb/V₂O₅/Ni) | 0.3, 0.6 or 0.9 (swept) | 0.07 | Abrupt SET, gradual RESET [1] |
| D2 (Nb/NbOₓ/V₂O₅/Ni) | 0.07 | 0.07 | Gradual SET and RESET [1] |

**Test application**

| Equation | Form | Used for |
|---|---|---|
| Features | x = pixel / 16, so x is in 0–1 | Logistic regression is trained on these features [7] |
| Class scores | z = Wᵀx + b | Software model |
| Inputs | row voltage = x·V_read; one extra row fixed at V_read for the bias | Voltages applied to the rows |
| Prediction | digit = argmax_j (I⁺_j − I⁻_j) | Largest differential column current |

Every row, including the bias row, maps a feature value of 1 to V_read, and every weight uses the same conductance scaling. The differential column currents are therefore proportional to z, and on a perfectly programmed ideal array the crossbar gives the same predictions as the software model.

## Numerical methods

- **Write-verify programming (simplified from [3]):** read → compare with target → SET or RESET pulse → repeat, until within tolerance or the pulse limit. As in [3], the state is read after each pulse and the pulse polarity depends on the sign of the error. Unlike [3], which uses ramps of increasing amplitude, every pulse here has the fixed amplitude of ±1.1 V used in [1].
- **Array operations** with NumPy [8].
- **Off-chip training** with scikit-learn [7]; only inference runs on the crossbar.
- **Plotting** with Matplotlib [9].

## Assumptions

- **Phenomenological cell:** no filament, oxygen-vacancy or thermal physics.
- **Update rule from another material:** the soft-bound rule was measured on HfO₂ [4], not V₂O₅. [4] also finds γ > 1 in the analogue region; γ = 1 is used here.
- **Gradual step size:** [1] uses trains of 40 pulses. It is assumed that one train covers about 95 % of the window, giving α = 1 − 0.05^(1/40) ≈ 0.072, rounded to 0.07.
- **Window range:** the sweep runs from 3, the low end of the analogue window of about 3–4 for the HfO₂ device in [4], to 100, the upper bound of the ON/OFF ratio (reported as below 100) of the devices in the integrated crossbar of [2].
- **No noise, no drift, no device-to-device spread:** none is modelled.
- **Linear read:** a cell behaves as a fixed resistor at the read voltage.
- **Ideal circuit:** zero wire resistance, ideal drivers, columns at virtual ground, ideal current measurement.
- **Single layer:** one 65 × 10 weight matrix (64 pixels + bias), stored as 130 × 10 cells.

## Parameters

| Parameter | Value | Source | Status |
|---|---|---|---|
| SET / RESET pulse | +1.1 V / −1.1 V, 1 ms | [1] | Reported for the potentiation/depression trains |
| Pulses per train | 40 | [1] | Reported |
| Read voltage | 0.1 V | [2], [4] | Reported for those devices; [1] does not state one for its pulse trains. On the ideal circuit it only scales the currents |
| G_min | 10 µS | [2] | Device conductances in [2] range from 10 to 100 µS. On the ideal circuit it only scales the currents |
| Memory window G_max/G_min | 10 (default); swept 3–100 | [2], [4] | Not reported for V₂O₅; see Assumptions |
| α_set, α_reset (D2); α_reset (D1) | 0.07 | – | Derived under the 95 % assumption |
| α_set (D1) | 0.3, 0.6, 0.9 | – | Swept |
| Soft-bound exponent γ | 1 | [4] | Simplest case of the rule in [4] |
| Programming tolerance | 1 % of the target conductance | [3] | [3] reached 1 % on Pt/TiO₂ devices using increasing-amplitude ramps; whether fixed ±1.1 V pulses can reach it is a result |
| Max pulses per cell | 100 | – | Choice |
| Dataset | 1797 images, 8 × 8, pixels 0–16 | [6], [7] | scikit-learn `load_digits`, a copy of the UCI test set [6]. 80/20 train/test split with a fixed seed |

## Results 

To be included
- D1 vs D2 pulse curves, G V pulse num,
- Programming of D1 V D2: pulses for each cell, and final error histogram
- Crossbar output vs exact product (hand calculated) and also error vs memory window
- Digit accuracy: software vs crossbar ........ for D1 vs D2 and vs memory window the accuracies etc.

## References

[1] K. Kaushlendra and D. Kaur, "Interface engineered V₂O₅-based flexible memristors towards high-performance brain-inspired neuromorphic computing," *Nanoscale Horiz.*, vol. 11, no. 3, pp. 763–777, 2026, doi: 10.1039/D5NH00562K.

[2] M. Prezioso, F. Merrikh-Bayat, B. D. Hoskins, G. C. Adam, K. K. Likharev, and D. B. Strukov, "Training and operation of an integrated neuromorphic network based on metal-oxide memristors," *Nature*, vol. 521, no. 7550, pp. 61–64, 2015, doi: 10.1038/nature14441.

[3] F. Alibart, L. Gao, B. D. Hoskins, and D. B. Strukov, "High precision tuning of state for memristive devices by adaptable variation-tolerant algorithm," *Nanotechnology*, vol. 23, no. 7, Art. no. 075201, 2012, doi: 10.1088/0957-4484/23/7/075201.

[4] J. Frascaroli, S. Brivio, E. Covi, and S. Spiga, "Evidence of soft bound behaviour in analogue memristive devices for neuromorphic computing," *Sci. Rep.*, vol. 8, Art. no. 7178, 2018, doi: 10.1038/s41598-018-25376-x.

[5] D. Ielmini and H.-S. P. Wong, "In-memory computing with resistive switching devices," *Nat. Electron.*, vol. 1, no. 6, pp. 333–343, 2018, doi: 10.1038/s41928-018-0092-2.

[6] E. Alpaydin and C. Kaynak, "Optical recognition of handwritten digits," UCI Machine Learning Repository, 1998, doi: 10.24432/C50P49.

[7] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[8] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.

[9] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.
