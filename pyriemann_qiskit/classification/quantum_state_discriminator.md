# QuantumStateDiscriminator

## In Plain Terms (for EEG practitioners)

Imagine the brain can be in one of two mental states — say, rest `|A>` or task `|B>`. We
don't observe the mental state directly: we observe the EEG, which is a kind of
measurement of the brain. Each EEG recording is a noisy, partial snapshot of the
underlying brain state.

The idea here is to learn, from labelled training data, what the EEG "looks like" on
average for each mental state. We call these learned descriptions the **class quantum
states**. At test time, we ask: given this new EEG recording, which class quantum state
does it match best? The answer is the predicted mental state.

What makes this quantum is not the hardware — it is the mathematical framework. The brain
state is described as a density matrix (a generalisation of a probability distribution
over possible states), the EEG is treated as a measurement operator, and the matching
score is computed using the Born rule from quantum mechanics. Crucially, the classifier
accounts for how often each class appears in the data (class priors), so it is not fooled
by imbalanced datasets.

---

## Concept

The mental state of the user (class A or B) is described by a **mixed quantum state**
(density matrix `ρ_c`). The EEG signal is the **measurement operator** `M` (observable).
The classifier is a **POVM** (Positive Operator-Valued Measure) learned from training
data. Classification scores `trace(Π_c · M)` are valid probabilities by construction —
they are non-negative and sum to 1 — directly from the Born rule.

## Quantum Mechanics Interpretation

| Quantum concept                   | Implementation                                              |
| --------------------------------- | ----------------------------------------------------------- |
| System state (mixed)              | Class density matrix `ρ_c` (learned per class)              |
| Observable / measurement operator | EEG covariance `M = X Xᵀ / T`, normalized to trace=1        |
| Class prior                       | `π_c = N_c / N_total` (estimated from training frequencies) |
| Measurement (POVM element)        | `Π_c = ρ_total^{-1/2} (π_c ρ_c) ρ_total^{-1/2}`             |
| Born rule score                   | `trace(Π_c · M) ∈ [0, 1]`, sums to 1 over classes           |
| Predicted class                   | `argmax_c trace(Π_c · M)`                                   |

## Relationship with pyriemann's MDM

### The EEG operator is a normalized covariance matrix

The measurement operator constructed from a raw EEG trial is:

```
M = X Xᵀ / T / trace(X Xᵀ / T)
```

pyriemann's `Covariances(estimator='scm')` computes the biased sample covariance:

```
Σ_scm = X Xᵀ / T
```

So `M = Σ_scm / trace(Σ_scm)` exactly. The unnormalized EEG operator is the **SCM (Sample
Covariance Matrix)** estimator — the MLE for a zero-mean Gaussian. The trace normalization
is the only extra step, rescaling the matrix so it has trace=1 and is therefore a valid
quantum density matrix.

Other common pyriemann estimators relate as follows:

| Estimator             | Formula                       | Relationship to `M`                                |
| --------------------- | ----------------------------- | -------------------------------------------------- |
| `'scm'`               | `X Xᵀ / T`                    | `M` before normalization — exact match             |
| `'cov'`               | `X Xᵀ / (T-1)`                | Same as SCM up to scalar, negligible for large `T` |
| `'lwf'` (Ledoit-Wolf) | `(1-α) Σ_scm + α (trace/n) I` | Regularized toward identity — not equivalent       |
| `'oas'`               | similar shrinkage             | Not equivalent                                     |

The current implementation uses no regularization. This can be ill-conditioned when the
number of channels `n_channels` is large relative to the number of time samples `n_times`.
In such cases, replacing the raw covariance with a shrinkage estimator (e.g. Ledoit-Wolf)
before normalization would be a natural extension.

### Both are nearest-mean classifiers

|                    | `Covariances() + MDM` | `QuantumStateDiscriminator` |
| ------------------ | --------------------- | --------------------------- |
| Feature            | `X Xᵀ / (T-1)`        | `X Xᵀ / T`, normalized      |
| Class prototype    | Riemannian mean `μ_c` | Density matrix `ρ_c`        |
| Similarity measure | Riemannian distance   | Quantum fidelity (PGM)      |
| Class coupling     | Independent per class | Coupled via `ρ_total`       |
| Priors             | Not used              | `π_c = N_c / N_total`       |

Both fit a per-class prototype from training data and classify a test sample by comparing
it to each prototype. Maximizing quantum fidelity and minimizing Riemannian distance are
dual notions of similarity.

### Where the analogy breaks down

MDM treats each class independently — the distance to class A does not depend on class B.
The PGM couples all classes through `ρ_total = Σ_c π_c ρ_c`, so each `Π_c` depends on all
other classes and their priors. This makes QuantumStateDiscriminator closer in spirit to
**LDA** than to MDM, despite the name.

A true quantum MDM would compute:

```
argmin_c  quantum_distance(ρ_test, ρ_c)
```

using e.g. the Bures distance or quantum relative entropy — without a POVM. The current
implementation is better described as **quantum state discrimination** (Pretty Good
Measurement), which the name does not fully convey.

## Fitting: Quantum State Tomography + Pretty Good Measurement

**Step 1 — Quantum state tomography.** For each class `c`, estimate the density matrix
from labelled training EEG:

```
Σ_c    = (1/N_c) Σ_i  X_i Xᵀ_i / T_i        (mean EEG covariance)
ρ_c    = Σ_c / trace(Σ_c)                     (normalize to trace=1)
π_c    = N_c / N_total                         (class prior)
```

**Step 2 — Pretty Good Measurement (PGM).** Build the prior-weighted average state and
compute the POVM:

```
ρ_total = Σ_c  π_c · ρ_c                      (prior-weighted mean state)

Π_c     = ρ_total^{-1/2} (π_c · ρ_c) ρ_total^{-1/2}
```

By construction, `Σ_c Π_c = I`, so scores are valid probabilities. Rare classes
automatically receive smaller `Π_c`; dominant classes receive larger ones.

For two classes with equal priors, the PGM approximates the **Helstrom measurement**
(theoretically optimal quantum state discrimination).

## Prediction: Born Rule

For a new EEG trial `X`, the measurement operator is:

```
M = X Xᵀ / T,    M ← M / trace(M)    (normalized to trace=1)
```

The score for each class is the quantum expectation value:

```
score_c = trace(Π_c · M) = Σ_{ij} (Π_c)_{ij} · M_{ij}    (Frobenius inner product)
```

Scores are returned directly as probabilities in `predict_proba` — no softmax is applied.

## Input / Output

- **Input**: raw EEG epochs `(n_trials, n_channels, n_times)` — no covariance step needed
- **Output**: class label or class probabilities (proper, summing to 1)
