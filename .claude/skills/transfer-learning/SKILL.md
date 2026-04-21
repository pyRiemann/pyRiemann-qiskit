---
name: transfer-learning
description:
  Guide for implementing cross-subject transfer learning in pyriemann-qiskit with MOABB.
  Use when the user asks about Adapter, TLCrossSubjectEvaluation, domain adaptation,
  Riemannian alignment, or cross-subject generalization.
argument-hint: [question or task]
---

# Transfer Learning: pyriemann + MOABB

## The Core Problem

MOABB's `CrossSubjectEvaluation` calls `pipeline.fit(X_train, y_train)` with plain class
labels only. Subject IDs live in a `metadata` DataFrame inside the evaluation loop but are
never forwarded to the pipeline — they are only used by the CV splitter
(`LeaveOneGroupOut` or `GroupKFold`).

`TLCenter` from pyriemann requires `y_enc`, an array of strings encoding both domain and
class: `"subject_01/0"`, produced by `encode_domains(X, y, domains)`. There is no way to
pass this through MOABB's standard pipeline interface without modification.

## The Solution: Adapter + TLCrossSubjectEvaluation

Both classes live in `pyriemann_qiskit/utils/transfer.py`.

### pyriemann `target_domain` Convention (Critical)

`target_domain` = the **first training subject** (the domain everything aligns TO), NOT
the test subject. This is because the test subject has no data in the training fold, so
TLRotate cannot compute its rotation.

- `TLCenter(target_domain="subj_1")` → aligns all domains so "subj_1"'s mean is the
  reference (identity). All other source domains get recentered to match.
- At `transform()` time, the reference domain's stored statistics are applied to the test
  data.

### Adapter

A meta-estimator that wraps preprocessing + a plain pyriemann TL estimator.
`target_domain=None` is used as a placeholder at construction time; the actual value is
injected via `set_params` inside `Adapter.fit()`.

```python
from pyriemann_qiskit.utils.transfer import Adapter

pipelines["MDM+TL"] = Adapter(
    preprocessing=Covariances(estimator="lwf"),
    estimator=make_pipeline(
        TLCenter(target_domain=None),
        TLScale(target_domain=None, centered_data=True),
        TLRotate(target_domain=None),
        TLClassifier(target_domain=None, estimator=MDM(), domain_weight=None),
    ),
)

pipelines["MDWM(0.5)"] = Adapter(
    preprocessing=Covariances(estimator="lwf"),
    estimator=MDWM(domain_tradeoff=0.5, target_domain=None, metric="riemann"),
)
```

`Adapter.fit(X, y, groups, target_domain)`:

1. `preprocessing.fit_transform(X)` → covariance matrices
2. `encode_domains(X_cov, y, groups)` → `y_enc` with all subject IDs
3. `deepcopy(estimator)` → fresh copy
4. `_set_target_domain(estimator_, target_domain)` → walks Pipeline steps, calls
   `step.set_params(target_domain=target_domain)` on any step that has the attribute
5. `estimator_.fit(X_cov, y_enc)` → fits in one shot

Because the inner estimators (`MDM()`, `TangentSpace()+LDA()`, etc.) are fully visible in
`repr(adapter.get_params())`, MOABB assigns distinct digests to each pipeline
automatically — no digest collision risk.

### TLClassifier and sample_weight (pyriemann patch)

`TLClassifier` always passes `sample_weight` to the wrapped estimator, but some estimators
(e.g. LDA) do not accept it. The installed pyriemann has been patched at:

`miniforge3/envs/quantum-layout/Lib/site-packages/pyriemann/transfer/_estimators.py`

The patch adds `import inspect` and makes `sample_weight` conditional:

```python
# Pipeline branch
for step_name, step_est in self.estimator.steps:
    sig = inspect.signature(step_est.fit).parameters
    if "sample_weight" in sig:
        sample_weight[step_name + "__sample_weight"] = weights
self.estimator.fit(X_dec, y_dec, **sample_weight)

# Non-pipeline branch
sig = inspect.signature(self.estimator.fit).parameters
if "sample_weight" in sig:
    self.estimator.fit(X_dec, y_dec, sample_weight=weights)
else:
    self.estimator.fit(X_dec, y_dec)
```

This should be upstreamed to pyriemann as a PR.

### TLCrossSubjectEvaluation

Subclass of `CrossSubjectEvaluation` that forwards `groups` and `target_domain` to
pipelines whose `fit()` declares them:

```python
evaluation = TLCrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    suffix="my_study",
    overwrite=True,
    n_splits=3,         # None = LeaveOneGroupOut
    random_state=seed,
)
results = evaluation.process(pipelines)
```

- `target_domain = str(groups[train[0]])` — first training subject, guaranteed to be in
  the training fold (test subject is NOT used as target_domain).
- Both TL and non-TL pipelines can be mixed in the same `pipelines` dict.
- Fitting and scoring errors are caught per-pipeline and printed without crashing the full
  evaluation.

## CrossSubject CV: LeaveOneOut vs GroupKFold

```python
# n_splits=None  →  LeaveOneGroupOut (one test subject per fold)
# n_splits=N     →  GroupKFold(n_splits=N)
evaluation = TLCrossSubjectEvaluation(..., n_splits=3)    # GroupKFold
evaluation = TLCrossSubjectEvaluation(..., n_splits=None)  # LOSO
```

## File Locations

- Implementation: `pyriemann_qiskit/utils/transfer.py`
- Example usage: `examples/resting_states/noplot_nch_study.py`
- pyriemann patch:
  `miniforge3/envs/quantum-layout/Lib/site-packages/pyriemann/transfer/_estimators.py`
