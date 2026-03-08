---
name: igl
description: Guide for Intrinsic Green's Learning (IGL) — a PDE-inspired kernel learning framework with automatic intrinsic dimension discovery. Use when the user asks about IGLClassifier, IGLSklearnClassifier, IGLTimeSeriesSklearnClassifier, TimeIGLClassifier, Variable Projection training, Hard Concrete Gates, GreenKernel, intrinsic dimension, ResIGL, train_vp_timeseries, or igl_reference.py.
argument-hint: [question or task]
---

# Intrinsic Green's Learning (IGL)

## What IGL Is

IGL models a target function as the solution to a PDE `Lu = f`, where `f` is a learned
source term integrated against a Green's kernel:

```
u(x) = ∫ G(x,y) f(y) dy
```

The key insight: assuming `f` and `G` factorize over a **low-dimensional latent coordinate**
`z = enc(x) ∈ R^d`, the d-dimensional integral collapses (via Fubini) into d independent
1D integrals at cost O(KRd) — far cheaper than the ambient O(D).

## File Location

`pyriemann_qiskit/classification/igl_reference.py`

## Two Variants

| Variant | Input | Use case |
|---------|-------|----------|
| `IGLSklearnClassifier` | `[N, D]` flat vectors | Post-covariance / TangentSpace features |
| `IGLTimeSeriesSklearnClassifier` | `[N, C, T]` raw epochs | EEG time series, preserves temporal structure |

---

## Variant 1: Feature-Vector IGL

**Architecture:**
```
x ∈ R^D → Encoder → z_raw ∈ R^d → HardConcreteGates → z ∈ R^d
        → GreenKernel → Φ ∈ [N,R] → Φ@W + bias → logits
```

### `IGLClassifier` (PyTorch `nn.Module`)

```python
from pyriemann_qiskit.classification.igl_reference import IGLClassifier

model = IGLClassifier(
    input_dim=55,       # ambient D (e.g. tangent space dim for 10-channel EEG)
    max_dim=32,         # d: max latent dimension
    n_classes=2,
    n_anchors=128,      # R: number of basis functions
    n_scales=4,         # K: kernel scales (multi-resolution)
    operator="gaussian",
    hidden=256,         # MLP encoder width
    use_gates=True,     # enable Hard Concrete gates
    encoder="mlp",      # "mlp" or "linear"
)
```

### `IGLSklearnClassifier` (sklearn-compatible)

Infers `input_dim` and `n_classes` from data at `fit()` time.

```python
from pyriemann_qiskit.classification.igl_reference import IGLSklearnClassifier, VPConfig

clf = IGLSklearnClassifier(
    max_dim=32,
    n_anchors=128,
    n_scales=4,
    operator="gaussian",
    hidden=256,
    encoder="mlp",
    training="vp",          # "vp" (recommended) or "joint"
    vp_config=VPConfig(epochs=1000, warmup_epochs=200, log_every=500, verbose=False),
    random_state=42,
)
clf.fit(X_train, y_train)   # X_train: [N, D]
proba = clf.predict_proba(X_test)

# Introspect effective dimension
d_eff = clf.effective_dimension()  # calls model_.get_active_dims()
```

**In MOABB pipeline** (after TangentSpace):
```python
pipelines["TS+IGL"] = make_pipeline(
    Covariances(estimator="lwf"),
    TangentSpace(metric="riemann"),
    IGLSklearnClassifier(max_dim=16, n_anchors=64, training="vp",
                         vp_config=VPConfig(epochs=500, warmup_epochs=100,
                                            log_every=500, verbose=False),
                         random_state=seed),
)
```

---

## Variant 2: Time-Series IGL (`TimeIGLClassifier`)

Operates directly on raw EEG epochs — **no covariance, no TangentSpace**.
Time is the PDE domain: `u_k(T) = Σ_s G(T,s) · f_k(x(s))`

**Architecture:**
```
x[n,t,:] ∈ R^C  →  source_enc (Linear(C,K))  →  f[n,t,:] ∈ R^K
t_grid ∈ [0,1]^T  →  GreenKernel  →  Φ ∈ [T, R]
G_row[s] = Φ[-1] · Φ[s]                     (G(T,s): kernel at query=T, source=s)
emb[n,k] = Σ_s G_row[s] · f[n,s,k]  →  [N, K]   (integral over time)
logits   = emb @ W_out + bias         →  [N, n_classes]
```

**Key design decisions:**
- `source_enc`: `Linear(C, K)` — K spatial filters (like CSP), no bias
- `G(T,s) = Φ[-1]·Φ[s]`: endpoint query evaluates solution at the final time.
  With Gaussian kernel → recent events weighted more; Helmholtz → oscillatory weighting.
- No attention mechanism: the kernel IS the temporal weighting
- Anchors initialized uniformly in [0,1] (the normalized time range)

### `IGLTimeSeriesSklearnClassifier` (sklearn-compatible)

Infers `n_channels`, `T`, `n_classes` from data. Accepts `[N, C, T]` (MOABB convention).

```python
from pyriemann_qiskit.classification.igl_reference import (
    IGLTimeSeriesSklearnClassifier, VPConfig
)

clf = IGLTimeSeriesSklearnClassifier(
    n_components=16,    # K: spatial filter bank width
    n_anchors=64,       # R: temporal anchor points
    n_scales=3,         # number of kernel scales
    operator="gaussian",
    training="vp",
    vp_config=VPConfig(epochs=500, warmup_epochs=100, log_every=500, verbose=False),
    random_state=42,
)
clf.fit(X_train, y_train)   # X_train: [N, C, T]
```

**In MOABB pipeline** (raw epochs, no preprocessing):
```python
pipelines["IGL"] = make_pipeline(
    IGLTimeSeriesSklearnClassifier(
        n_components=16, n_anchors=64, n_scales=3, operator="gaussian",
        training="vp",
        vp_config=VPConfig(epochs=500, warmup_epochs=100, log_every=500, verbose=False),
        random_state=seed,
    ),
)
```

### `train_vp_timeseries`

Called internally by `IGLTimeSeriesSklearnClassifier` when `training="vp"`.
Can also be called directly on a `TimeIGLClassifier` instance.

```python
from pyriemann_qiskit.classification.igl_reference import (
    TimeIGLClassifier, train_vp_timeseries, VPConfig
)
import torch

model = TimeIGLClassifier(n_channels=16, T=250, n_components=16,
                           n_anchors=64, n_scales=3, n_classes=2)
X_t = torch.tensor(X_train, dtype=torch.float32)  # [N, C, T]
y_t = torch.tensor(y_train, dtype=torch.long)

history = train_vp_timeseries(model, X_t, y_t, config=VPConfig(epochs=500))
```

VP phase for time-series:
- **Warmup**: joint gradient on all params (source_enc + GreenKernel + W_out)
- **VP phase**: Stage 1 = gradient on `source_enc + GreenKernel`; Stage 2 = Tikhonov lstsq for `W_out ∈ [K, n_classes]` from embeddings `∈ [N, K]`

---

## `GreenKernel` (shared by both variants)

```python
from pyriemann_qiskit.classification.igl_reference import GreenKernel

kernel = GreenKernel(
    latent_dim=1,       # d: dimension of the integration domain
    n_anchors=64,       # R: number of basis functions
    n_scales=3,         # K: multi-scale kernel widths (log-spaced)
    operator="gaussian",
)
Phi = kernel.compute_design_matrix(z)  # z: [N, d] → Phi: [N, R]
# NOTE: n_scales are accumulated internally → output is [N, R], not [N, R*n_scales]
```

**Anchor positions** are learnable `[R, d]`, initialized `randn * 0.5`.
For `TimeIGLClassifier`: anchors are re-initialized to `Uniform(0, 1)` after construction.

**Available operators:**
| Operator | Kernel | EEG inductive bias |
|----------|--------|--------------------|
| `"gaussian"` | `exp(-r²/(2σ²))` | Smooth states, recent-time emphasis at endpoint |
| `"helmholtz"` | `cos(ω·r)` | Oscillatory rhythms (alpha, beta) |
| `"cauchy"` | `1/(1+r²/σ²)` | Robust to outlier time points |
| `"gabor"` | `cos(ω·r)·exp(-r²/(2σ²))` | Frequency-localized bursts |
| `"laplacian"` | `exp(-r/σ)` | Sharp transients |
| `"mexican_hat"` | `(1-r²/σ²)·exp(-r²/(2σ²))` | Band-pass, edge detection |

---

## `HardConcreteGates` (feature-vector IGL only)

L0 regularization for automatic intrinsic dimension discovery.
Stochastic in training, deterministic at eval.

```python
from pyriemann_qiskit.classification.igl_reference import HardConcreteGates

gates = HardConcreteGates(max_dim=32)
z_masked = gates(z_raw)                   # apply gates
loss_reg  = gates.l0_regularization_term()
d_eff     = gates.effective_dimension()   # count active gates (P(g>0) > 0.5)
```

Gate probability: `P(g_j > 0) = σ(log_α_j - β·log(-γ/ζ))`.

**Not used in `TimeIGLClassifier`**: time is inherently 1D, no dimension selection needed.
The temporal structure is captured by the Green's kernel and the `n_components` spatial filters.

---

## `VPConfig` / `JointConfig`

```python
from pyriemann_qiskit.classification.igl_reference import VPConfig, JointConfig

# VPConfig fields (dataclass)
VPConfig(
    encoder_lr=1e-3,
    epochs=1500,            # total = warmup + VP epochs
    warmup_epochs=300,      # warmup: all params jointly
    batch_size=256,
    gate_weight=0.015,      # L0 penalty weight
    source_l2=1e-3,         # Tikhonov lambda in lstsq
    inner_batch_size=4096,
    grad_clip=1.0,
    scheduler="cosine_warm_restarts",
    log_every=100,
    verbose=True,
)

JointConfig(
    lr=1e-3,
    weight_decay=1e-4,
    epochs=1500,
    batch_size=256,
    gate_weight=0.015,
    grad_clip=1.0,
    early_stopping_patience=0,  # 0 = disabled
    log_every=100,
    verbose=True,
)
```

---

## Training: VP vs Joint

### VP (recommended)

```
Warmup (epochs 0..warmup_epochs):   joint gradient on all params
VP phase:
  Stage 1: gradient on encoder + gates (W_out detached)
  Stage 2: W_out ← lstsq(Φ(z), y)  [Tikhonov-regularized, CPU]
```

**Why VP prevents collapse**: `W_out` is always optimal for current encoder → gradient
signal on the encoder reflects coordinate quality only (envelope theorem).
Joint training collapses `d_eff → 0` because `W_out` absorbs all variance.

### Joint (ablation only)

Full gradient on all params simultaneously. Useful for baselines/debugging.
`train_joint(model, X_t, y_t, task="classification", config=JointConfig(...))`

---

## ResIGL: IGL as Geometric Regularizer

**Critical: two-phase training only** — joint ResIGL collapses `d_eff → 0`.

```
Phase 1: train IGL normally (VP) → discovers intrinsic coordinates, d_eff preserved
Phase 2: freeze IGL, train MLP residual: output = IGL(x) + MLP(x)
```

---

## Hyperparameter Guide

**Feature-vector IGL** (`IGLSklearnClassifier`):
| Param | Range | Effect |
|-------|-------|--------|
| `max_dim` | 16–100 | Upper bound on discoverable dims |
| `n_anchors` | 64–512 | Expressivity |
| `gate_weight` (in VPConfig) | 0.005–0.05 | Gate sparsity |
| `n_scales` | 2–5 | Multi-resolution coverage |
| `source_l2` | 1e-4–1e-2 | lstsq regularization |

**Time-series IGL** (`IGLTimeSeriesSklearnClassifier`):
| Param | Range | Effect |
|-------|-------|--------|
| `n_components` | 4–32 | Spatial filter bank width (like CSP rank) |
| `n_anchors` | 32–128 | Temporal basis expressivity |
| `operator` | `"gaussian"` | Default; `"helmholtz"` for oscillatory EEG |
| `n_scales` | 2–4 | Temporal multi-resolution |

**Tuning time-series:**
1. Start: `n_components=16, n_anchors=64, operator="gaussian", training="vp"`
2. If underfitting → increase `n_components` or `n_anchors`
3. For rhythmic EEG (alpha/beta tasks) → try `operator="helmholtz"`

---

## Experimental Results

| Dataset | Model | Accuracy | d_eff |
|---------|-------|----------|-------|
| WiC/BERT (R^768) | IGL-Gaussian | 62.6% ± 0.6% | ~18-21 |
| WiC/BERT | Random Forest | 67.2% | n/a |
| MNIST | IGL | ~97% | ~12 |
| Swiss Roll (D=100, d=2) | IGL | R²=0.998 | 2 (exact) |

Only 2-2.7% of 768 BERT dims used. Permutation test confirms gates reflect task structure.

---

## Key Theoretical Properties

- **Separability barrier**: Cannot model `f(x,y) = g(x·y)` from 1D integrals alone.
- **Complexity**: O(DW² + KRd); Stage 2 lstsq adds O(KRd) only.
- **Envelope theorem**: At `W`-optimum `∂L/∂W=0`, so `∇_θ L_red` reflects coordinate quality.
- **`direct_solve_weights()`**: Tikhonov lstsq forced to CPU (MPS instability bug).
- **`TimeIGLClassifier`**: `G(T,s) = Φ[-1]·Φ[s]` — the factorized Green's function at
  the endpoint query is the inner product of the two basis vectors. The kernel operator
  determines how temporal distance from `T` is weighted.
