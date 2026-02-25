"""Transfer learning utilities compatible with MOABB's evaluation framework."""

import inspect
from copy import deepcopy
from time import time

import numpy as np
from mne.epochs import BaseEpochs
from moabb.evaluations import CrossSubjectEvaluation
from moabb.evaluations.splitters import CrossSubjectSplitter
from moabb.evaluations.utils import _create_save_path, _ensure_fitted, _save_model_cv
from pyriemann.transfer import encode_domains
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def _pipeline_accepts_groups(clf):
    """Return True if clf.fit() declares a groups parameter."""
    return "groups" in inspect.signature(clf.fit).parameters


def _pipeline_accepts_target_domain(clf):
    """Return True if clf.fit() declares a target_domain parameter."""
    return "target_domain" in inspect.signature(clf.fit).parameters



def _set_target_domain(estimator, target_domain):
    """Set target_domain on any step (or the estimator itself) that declares it."""
    if hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            if hasattr(step, "target_domain"):
                step.set_params(target_domain=target_domain)
    elif hasattr(estimator, "target_domain"):
        estimator.set_params(target_domain=target_domain)


class Adapter(BaseEstimator, ClassifierMixin):
    """Meta-estimator bridging MOABB's evaluation interface with pyriemann TL pipelines.

    Handles preprocessing (raw → covariances), domain encoding, and
    target-domain propagation, then delegates to a plain pyriemann TL estimator.

    Parameters
    ----------
    preprocessing : sklearn transformer
        Transforms raw input into covariance matrices.
    estimator : sklearn estimator
        Any pyriemann TL-compatible estimator (e.g. a pipeline of
        TLCenter/TLScale/TLRotate/TLClassifier, or MDWM) constructed with
        ``target_domain=None``. The actual value is injected via
        ``set_params`` at fit time from ``TLCrossSubjectEvaluation``.
    """

    def __init__(self, preprocessing, estimator):
        self.preprocessing = preprocessing
        self.estimator = estimator

    def fit(self, X, y, groups=None, target_domain=None):
        self.preprocessing_ = deepcopy(self.preprocessing)
        X_cov = self.preprocessing_.fit_transform(X)
        _, y_enc = encode_domains(X_cov, y, groups)
        self.estimator_ = deepcopy(self.estimator)
        _set_target_domain(self.estimator_, target_domain)
        self.estimator_.fit(X_cov, y_enc)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.estimator_.predict(self.preprocessing_.transform(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.preprocessing_.transform(X))


class TLCrossSubjectEvaluation(CrossSubjectEvaluation):
    """CrossSubjectEvaluation with group and target-domain forwarding.

    Extends CrossSubjectEvaluation with one behavioural change:

    Pipelines whose fit() declares ``groups`` receive the training subject IDs.
    Pipelines whose fit() also declares ``target_domain`` receive the test
    subject's ID as target_domain. All other pipelines are unaffected.

    **kwargs
        Forwarded to CrossSubjectEvaluation.__init__().
    """

    # flake8: noqa: C901
    def evaluate(
        self,
        dataset,
        pipelines,
        param_grid,
        process_pipeline,
        postprocess_pipeline=None,
    ):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")

        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(
                self.results.not_yet_computed(
                    pipelines, dataset, subject, process_pipeline
                )
            )
        if len(run_pipes) == 0:
            return

        X, y, metadata = self.paradigm.get_data(
            dataset=dataset,
            return_epochs=self.return_epochs,
            return_raws=self.return_raws,
            cache_config=self.cache_config,
            postprocess_pipeline=postprocess_pipeline,
            process_pipelines=[process_pipeline],
        )
        le = LabelEncoder()
        y = y if self.mne_labels else le.fit_transform(y)

        groups = metadata.subject.values
        sessions = metadata.session.values
        n_subjects = len(dataset.subject_list)

        scorer = get_scorer(self.paradigm.scoring)

        if self.n_splits is None:
            cv_class = LeaveOneGroupOut
            cv_kwargs = {}
        else:
            cv_class = GroupKFold
            cv_kwargs = {"n_splits": self.n_splits}
            n_subjects = self.n_splits

        self.cv = CrossSubjectSplitter(
            cv_class=cv_class, random_state=self.random_state, **cv_kwargs
        )

        inner_cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)

        for cv_ind, (train, test) in enumerate(
            tqdm(
                self.cv.split(y, metadata),
                total=n_subjects,
                desc=f"{dataset.code}-CrossSubject",
            )
        ):
            subject = groups[test[0]]
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )

            for name, clf in run_pipes.items():
                t_start = time()
                clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=clf, inner_cv=inner_cv
                )

                try:
                    if _pipeline_accepts_target_domain(clf):
                        model = deepcopy(clf).fit(
                            X[train],
                            y[train],
                            groups=groups[train],
                            target_domain=str(groups[train[0]]),
                        )
                    elif _pipeline_accepts_groups(clf):
                        model = deepcopy(clf).fit(
                            X[train], y[train], groups=groups[train]
                        )
                    else:
                        model = deepcopy(clf).fit(X[train], y[train])
                    _ensure_fitted(model)
                except Exception as e:
                    import traceback

                    print(f"\n[TLCrossSubjectEvaluation] ERROR fitting '{name}': {e}")
                    traceback.print_exc()
                    continue

                duration = time() - t_start

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = _create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=self.search,
                        eval_type="CrossSubject",
                    )
                    _save_model_cv(
                        model=model,
                        save_path=model_save_path,
                        cv_index=str(cv_ind),
                    )

                for session in np.unique(sessions[test]):
                    ix = sessions[test] == session
                    try:
                        score = scorer(model, X[test[ix]], y[test[ix]])
                    except Exception as e:
                        import traceback

                        print(
                            f"\n[TLCrossSubjectEvaluation] ERROR scoring '{name}': {e}"
                        )
                        traceback.print_exc()
                        continue
                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }
                    yield res
