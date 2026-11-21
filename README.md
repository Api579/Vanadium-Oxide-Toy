# Vanadium-Oxide-Toy

# Vanadium Oxide (VOₓ) Crossbar – Handwritten Digit Recognition with In-Memory Computing

Simplified toy model, written in Python.

- **Device:** V₂O₅/VOₓ resistive memory cell (non-volatile, filament-type resistive switching)
- **Array:** crossbar of VOₓ cells that stores neural-network weights and computes matrix-vector products in place
- **Application:** handwritten digit recognition (scikit-learn 8×8 digits dataset)

> **Note:** this is a toy model. Device behaviour uses simple phenomenological equations with typical values, not measured V₂O₅ data or atomistic simulation. Many simplifications were made so it runs on a local machine (see [Assumptions](#assumptions-and-simplifications)).

---

## Main question

**How well does a neural network work when its weights are stored in imperfect VOₓ memory cells?**

A small network is trained in software, its weights are programmed into a simulated VOₓ crossbar, and digits are classified by reading currents from the array. Accuracy is compared with the software result while device flaws (programming error, limited levels, variation, drift, read noise) are switched on one at a time.

## What it does

1. **Train:** a single-layer neural network (64 inputs → 10 classes, softmax) is trained in NumPy on the digits dataset.
2. **Store:** the trained weights are mapped onto cell conductances (differential pair for signed weights).
3. **Program:** each cell is set to its target conductance using voltage pulses and a write-verify loop.
4. **Compute (inference):** each test image becomes a set of row voltages; the column currents give the class scores in one analog step; the largest score is the predicted digit.
5. **Evaluate:** crossbar accuracy vs software accuracy, under different device non-idealities.

## Output

![Crossbar output](<crossbar_output.png>)

*Add your final plot(s) here, e.g. programming curves, accuracy vs device variation, confusion matrix.*

## Running

```bash
pip install numpy scipy matplotlib scikit-learn
python vox_crossbar.py
```

---

## Dataset

- **scikit-learn `load_digits`:** 1797 images of handwritten digits 0–9, each 8 × 8 pixels (64 inputs), pixel values 0–16.
- **Split:** 80 % training, 20 % testing.
- **Inputs to voltages:** pixels scaled to 0–1, then to 0–V_read (e.g. 0–0.1 V), one pixel per row.
- **Bias:** one extra row driven at a constant voltage, so the array is 65 rows × 10 columns.

## Governing equations

**Circuit physics**

| Equation | Form | Used for |
|---|---|---|
| Ohm's law | I = G·V | Current through one cell |
| Kirchhoff's current law | I_j = Σᵢ G_ij·V_i | Currents add along each column wire |
| Matrix-vector product | I = Gᵀ·V | What the whole array computes |
| Differential pair | I_j = Σᵢ (G⁺_ij − G⁻_ij)·V_i ∝ (Wᵀx)_j | Signed weights from two positive conductances |
| Weight mapping | G± = G_min + (G_max − G_min)·max(±W, 0) / max\|W\| | Weight → target conductance |

**Device model (VOₓ cell)**

| Equation | Form | Used for |
|---|---|---|
| SET pulse (potentiation) | G ← G + α(G_max − G) | Conductance increase, smaller steps near G_max |
| RESET pulse (depression) | G ← G − α(G − G_min) | Conductance decrease, smaller steps near G_min |
| Switching threshold | pulse changes G only if \|V\| > V_th | Reads (low V) never disturb stored weights |
| Conductance bounds | G_min ≤ G ≤ G_max | Physical limits of the device |
| Drift (optional) | G(t) = G₀·(t / t₀)^(−ν) | Slow loss of stored state (retention) |

**Neural network (software training)**

| Equation | Form | Used for |
|---|---|---|
| Class scores | z = Wᵀx + b | Linear layer |
| Softmax | p_k = e^(z_k) / Σ e^(z_j) | Scores → probabilities |
| Cross-entropy loss | L = −Σ y_k log p_k | Training objective |
| Gradient descent | W ← W − η ∂L/∂W | Weight update |
| Prediction | digit = argmax_j I_j | Class from largest column current |

## Numerical methods and techniques

- **Gradient descent** (batch or mini-batch) for training the network in NumPy.
- **Write-verify programming:** an iterative closed loop (read → compare with target → SET or RESET pulse → repeat) until the error is within tolerance or a pulse limit is reached.
- **Vectorised array operations** (NumPy) to read the whole array and classify all test images at once.
- **Monte Carlo variability:** random Gaussian spread for cycle-to-cycle (each pulse) and device-to-device (each cell) variation, and read noise; results averaged over several random seeds.
- **Quantisation:** conductances rounded to a fixed number of levels to study limited precision.
- **Parameter sweeps:** accuracy measured while varying one non-ideality at a time.
- **(Optional, Stage 5)** **Nodal analysis** for wire resistance and sneak paths: Kirchhoff's current law at every node gives a sparse linear system solved by **LU decomposition** (`scipy.sparse.linalg.splu`).

## Assumptions and simplifications

- **Phenomenological device model:** no filament, ion or thermal physics; conductance changes follow simple update rules.
- **Non-volatile:** G stays constant with no voltage applied (except optional drift).
- **Linear I–V when reading:** the cell behaves as a fixed resistor at the read voltage.
- **Ideal circuit (Stages 1–4):** zero wire resistance, ideal voltage drivers, columns held at exactly 0 V (virtual ground), ideal current measurement.
- **DC steady state:** no capacitance, timing or transient effects.
- **No selector device:** no sneak-path suppression (sneak paths studied only in Stage 5).
- **Software training, hardware inference:** weights are trained off-chip and only programmed once.
- **Single layer:** one 65 × 10 weight matrix (130 × 10 cells as differential pairs); no hidden layer.
- **Typical toy values:** parameters are representative of oxide memristors, not fitted to a specific V₂O₅ device.

## Parameters

| Parameter | Value |
|---|---|
| Crossbar size | 65 rows × 10 columns (× 2 for differential pair) |
| G_min / G_max | 1 µS / 50 µS (on/off ≈ 50) |
| Read voltage (max input) | 0.1 V |
| Switching threshold V_th | ~0.8 V |
| SET / RESET pulse | +1.2 V / −1.2 V |
| Step factor α | 0.05 |
| Programming tolerance | 2 % |
| Max pulses per cell | 100 |
| Cycle-to-cycle variation | ~5 % per step |
| Device-to-device variation | ~10 % |
| Conductance levels (sweep) | 4, 8, 16, 32, 64 |
| Learning rate η | ~0.1 |
| Training epochs | ~200 |

## Project stages

- [ ] **Stage 1 – Train:** softmax network in NumPy on the digits dataset; record software accuracy
- [ ] **Stage 2 – Store:** map weights onto conductances (differential pair)
- [ ] **Stage 3 – Program:** pulse model + write-verify loop; programming curves and error histogram
- [ ] **Stage 4 – Compute and evaluate:** crossbar inference on the test set; accuracy vs
  - [ ] programming tolerance
  - [ ] number of conductance levels
  - [ ] device-to-device variation
  - [ ] read noise
  - [ ] drift over time (retention)
- [ ] **Stage 5 (optional) – Real array:** wire resistance and sneak paths via nodal analysis

## Results to report

- Software accuracy vs crossbar accuracy (ideal and non-ideal devices)
- Accuracy vs each non-ideality (one plot per sweep)
- Programming cost: pulses per cell, final weight error
- Confusion matrix for the crossbar classifier
- Which device flaw hurts accuracy most

## Out of scope

- Atomistic / DFT simulation of the vanadium oxide structure
- VO₂ insulator–metal transition (volatile switching); possible follow-up as a selector or neuron
- Training on the crossbar itself (in-situ learning)
- Multi-layer networks and larger datasets (e.g. MNIST 28×28)
- Peripheral electronics (ADC/DAC, sense amplifier design)
