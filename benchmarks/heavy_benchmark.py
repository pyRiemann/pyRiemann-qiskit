"""
====================================================================
Benchmark Alpha
====================================================================

A benchmark on a predefined list of databases for P300 and Motor Imagery (LR).
Currently it requires the latest version of MOABB where:
    - cache_config is availabe for WithinSessionEvaluation|()
    - this bug is fixed: https://github.com/NeuroTechX/moabb/issues/514

Adapts both the pipeline and the paradigms depending on the test databaase.
Automatically changes the first transformers from XDawnCovariances() to Covariances()
when switching from P300 to MotorImagery.

"""
# Author: Anton Andreev

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
from moabb import set_log_level

# P300
from moabb.datasets import (
    BI2013a,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_003,
    EPFLP300,
    Lee2019_ERP,
    BI2014a,
    BI2014b,
    BI2015a,
    BI2015b,
    EPFLP300,
    Sosulski2019,
)

# Motor imagery
from moabb.datasets import (
    BNCI2014_001,
    Zhou2016,
    BNCI2015_001,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2015_004,
    AlexMI,
    Weibo2014,
    Cho2017,
    GrosseWentrup2009,
    PhysionetMI,
    Shin2017A,
)
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
)
from moabb.paradigms import P300, MotorImagery, LeftRightImagery

# from pyriemann.classification import MDM
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPRegressor
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
# from sklearn import svm

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)

print(__doc__)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
set_log_level("info")


def benchmark_alpha(pipelines, max_n_subjects=-1, overwrite=False, n_jobs=12):
    """

    Parameters
    ----------
    pipelines : TYPE
        DESCRIPTION.
    overwrite : TYPE, optional
        # set to True if we want to overwrite cached results

    Returns
    -------
    None.

    """

    cache_config = dict(
        use=True,
        save_raw=False,
        save_epochs=True,
        save_array=True,
        overwrite_raw=False,
        overwrite_epochs=False,
        overwrite_array=False,
    )

    paradigm_P300 = P300()
    paradigm_MI = MotorImagery()
    paradigm_LR = LeftRightImagery()

    datasets_P300 = [
        BI2013a(),
        BNCI2014_008(),
        BNCI2014_009(),
        BNCI2015_003(),
        BI2015a(),
        BI2015b(),
        BI2014a(),
        BI2014b(),
    ]  # Sosulski2019(), EPFLP300()

    datasets_MI = [  # BNCI2015_004(), #5 classes, Error: Classification metrics can't handle a mix of multiclass and continuous targets
        BNCI2015_001(),  # 2 classes
        BNCI2014_002(),  # 2 classes
        # AlexMI(),       #3 classes, Error: Classification metrics can't handle a mix of multiclass and continuous targets
    ]

    datasets_LR = [
        BNCI2014_001(),
        BNCI2014_004(),
        Cho2017(),  # 49 subjects
        GrosseWentrup2009(),
        PhysionetMI(),  # 109 subjects
        Shin2017A(accept=True),
        Weibo2014(),
        Zhou2016(),
    ]

    # each MI dataset can have different classes and events and this requires a different MI paradigm
    paradigms_MI = []
    for dataset in datasets_MI:
        events = list(dataset.event_id)
        paradigm = MotorImagery(events=events, n_classes=len(events))
        paradigms_MI.append(paradigm)

    # checks if correct paradigm is used
    for d in datasets_P300:
        name = type(d).__name__
        print(name)
        if name not in [
            (lambda x: type(x).__name__)(x) for x in paradigm_P300.datasets
        ]:
            print("Error: dataset not compatible with selected paradigm", name)
            import sys

            sys.exit(1)

    for d in datasets_MI:
        name = type(d).__name__
        print(name)
        if name not in [(lambda x: type(x).__name__)(x) for x in paradigm_MI.datasets]:
            print("Error: dataset not compatible with selected paradigm", name)
            import sys

            sys.exit(1)

    for d in datasets_LR:
        name = type(d).__name__
        print(name)
        if name not in [(lambda x: type(x).__name__)(x) for x in paradigm_LR.datasets]:
            print("Error: dataset not compatible with selected paradigm", name)
            import sys

            sys.exit(1)

    # adjust the number of subjects, the Quantum pipeline takes a lot of time
    # if executed on the entire dataset
    if max_n_subjects != -1:
        for dataset in datasets_P300:
            n_subjects_ds = min(max_n_subjects, len(dataset.subject_list))
            dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

        for dataset in datasets_MI:
            n_subjects_ds = min(max_n_subjects, len(dataset.subject_list))
            dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

        for dataset in datasets_LR:
            n_subjects_ds = min(max_n_subjects, len(dataset.subject_list))
            dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

    print("Total pipelines to evaluate: ", len(pipelines))

    evaluation_P300 = WithinSessionEvaluation(
        paradigm=paradigm_P300,
        datasets=datasets_P300,
        suffix="examples",
        overwrite=overwrite,
        n_jobs=n_jobs,
        n_jobs_evaluation=n_jobs,
        cache_config=cache_config,
    )
    evaluation_LR = WithinSessionEvaluation(
        paradigm=paradigm_LR,
        datasets=datasets_LR,
        suffix="examples",
        overwrite=overwrite,
        n_jobs=n_jobs,
        n_jobs_evaluation=n_jobs,
        cache_config=cache_config,
    )

    results_P300 = evaluation_P300.process(pipelines)

    # replace XDawnCovariances with Covariances when using MI or LeftRightMI
    for pipe_name in pipelines:
        pipeline = pipelines[pipe_name]
        if pipeline.steps[0][0] == "xdawncovariances":
            pipeline.steps.pop(0)
            pipeline.steps.insert(0, ["covariances", Covariances("oas")])
            print("xdawncovariances repalced by covariances")

    results_LR = evaluation_LR.process(pipelines)

    results = pd.concat([results_P300, results_LR], ignore_index=True)

    # each MI dataset uses its own configured MI paradigm
    for paradigm_MI, dataset_MI in zip(paradigms_MI, datasets_MI):
        evaluation_MI = WithinSessionEvaluation(
            paradigm=paradigm_MI,
            datasets=[dataset_MI],
            overwrite=overwrite,
            n_jobs=n_jobs,
            n_jobs_evaluation=n_jobs,
            cache_config=cache_config,
        )

        results_per_MI_pardigm = evaluation_MI.process(pipelines)
        results = pd.concat([results, results_per_MI_pardigm], ignore_index=True)

    return results


def plot_stat(results):
    fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

    sns.stripplot(
        data=results,
        y="score",
        x="pipeline",
        ax=ax,
        jitter=True,
        alpha=0.5,
        zorder=1,
        palette="Set1",
    )
    sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1")

    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.3, 1)

    plt.show()

    # generate statistics for the summary plot
    # Compute matrices of p-values and effects for all algorithms over all datasets via combined p-values and
    # combined effects methods
    stats = compute_dataset_statistics(results)
    P, T = find_significant_differences(stats)
    # agg = stats.groupby(['dataset']).mean()
    # print(agg)
    print(stats.to_string())  # not all datasets are in stats

    # Negative SMD value favors the first algorithm, postive SMD the second
    # A meta-analysis style plot that shows the standardized effect with confidence intervals over
    # all datasets for two algorithms. Hypothesis is that alg1 is larger than alg2
    pipelines = results["pipeline"].unique()
    pipelines_sorted = sorted(pipelines)
    for i in range(0, len(pipelines_sorted)):
        for j in range(i + 1, len(pipelines_sorted)):
            fig = moabb_plt.meta_analysis_plot(
                stats, pipelines_sorted[i], pipelines_sorted[j]
            )
            plt.show()

    # summary plot - significance matrix to compare pipelines.
    # Visualize significances as a heatmap with green/grey/red for significantly higher/significantly lower.
    moabb_plt.summary_plot(P, T)
    plt.show()
