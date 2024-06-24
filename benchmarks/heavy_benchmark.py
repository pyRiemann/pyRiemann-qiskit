"""
====================================================================
Benchmark Alpha
====================================================================

A benchmark on a predefined list of databases for P300 and Motor Imagery (MI).

Currently it requires the latest version of MOABB (from Github) where:
    - cache_config is availabe for WithinSessionEvaluation|()
    - bug 514 is fixed: https://github.com/NeuroTechX/moabb/issues/514

Adapts both the pipeline and the paradigms depending on the evaluated database.
Automatically changes the first transformers from XDawnCovariances() to Covariances()
when switching from P300 to MI.

"""
# Author: Anton Andreev

from pyriemann.estimation import Covariances
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
from moabb import set_log_level

# P300 databases
from moabb.datasets import (
    BI2013a,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_003,
    # EPFLP300,
    # Lee2019_ERP,
    BI2014a,
    BI2014b,
    BI2015a,
    BI2015b,
)

# Motor imagery databases
from moabb.datasets import (
    BNCI2014_001,
    Zhou2016,
    # BNCI2015_001,
    # BNCI2014_002,
    BNCI2014_004,
    # BNCI2015_004, #not tested
    # AlexMI,
    Weibo2014,
    Cho2017,
    GrosseWentrup2009,
    PhysionetMI,
    Shin2017A,
    Lee2019_MI,  # new
    Schirrmeister2017,  # new
)
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSubjectEvaluation,
)
from moabb.paradigms import P300, MotorImagery, LeftRightImagery

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (
    compute_dataset_statistics,
    find_significant_differences,
)

print(__doc__)
print("Version 1.0 24/06/2024")

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")
set_log_level("info")


def benchmark_alpha(
    pipelines,
    params_grid=None,
    evaluation_type="withinsession",
    max_n_subjects=-1,
    overwrite=False,
    n_jobs=12,
    skip_MI=False,
    skip_P300=False,
    replace_x_dawn_cov_par_cov_for_MI=True,
    use_cache=False,
    save_cache=False,
):
    """

    Parameters
    ----------
    pipelines :
        Pipelines to test. The pipelines are expected to be configured for P300.
        When switching from P300 to Motor Imagery and if the first transformer
        is XdawnCovariances then it will be automatically replaced by Covariances()
    params_grid: default = None
        Used for hyperparameter tuning with GridSearchCV and MOABB.
    evaluation_type: Default = "withinsession"
        Two options are available: "withinsession" and "crosssubject".
    max_n_subjects : int, default = -1
        The maxmium number of subjects to be used per database.
    overwrite : bool, optional
        Set to True if we want to overwrite cached results.
    n_jobs : int, default=12
        The number of jobs to use for the computation.
        It is used in WithinSessionEvaluation().
    skip_MI : default = False
        Skip Motor Imagery datasets for this benchmark.
    skip_P300 : default = False
        Skip P300 ERP datasets for this benchmark.
    replace_x_dawn_cov_par_cov_for_MI : default = True
        XDawn covariances is automatically replaced with Covariances when
        used in Motor Imagery.
    use_cache: default = False
        Used cache that was previously generated with MOABB.
    save_cache: default = False
        Force generation of cache using MOABB.

    Returns
    -------
    df : Pandas dataframe
        Returns a dataframe with results from the tests.

    History:
        24/06/2024 Initial version.
    """

    cache_config = dict(
        use=use_cache,
        save_raw=False,
        save_epochs=save_cache,
        save_array=save_cache,
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
    ]

    datasets_MI = [
        # BNCI2015_004(), #gives very low scores like 0.2 for most users
        # BNCI2015_001(),
        # BNCI2014_002(),
        # AlexMI(),
    ]

    # Dataset                 Electrodes   Subjects
    # BNCI2014_001(),            22           9
    # BNCI2014_004(),             3           9
    # Cho2017(),                 64          52
    # GrosseWentrup2009(),      128          10
    # PhysionetMI(),             64         109
    # Shin2017A(accept=True),    30          29
    # Weibo2014(),               60          10
    # Zhou2016(),                14           4
    # Lee2019_MI(),              62          54
    # Schirrmeister2017()       128          14

    datasets_LR = [
        BNCI2014_001(),
        BNCI2014_004(),
        Cho2017(),
        GrosseWentrup2009(),
        PhysionetMI(),  # D2
        Shin2017A(accept=True),
        Weibo2014(),
        # gives error with covariance estimator "cov" so regularized
        # "oas" covariance estimator is required
        Zhou2016(),
        Lee2019_MI(),  # requires a newer version of MOABB with url fixed
        Schirrmeister2017(),
    ]

    # each MI dataset can have different classes and events
    # and this requires a different MI paradigm
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

    # adjust the number of subjects min(max_n_subjects, total_count)
    if max_n_subjects != -1:
        for dataset in datasets_P300:
            n_subjects_ds = min(max_n_subjects, len(dataset.subject_list))
            dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

        for dataset in datasets_MI:
            n_subjects_ds = min(max_n_subjects, len(dataset.subject_list))
            # print(dataset, n_subjects_ds)
            dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

        for dataset in datasets_LR:
            n_subjects_ds = min(max_n_subjects, len(dataset.subject_list))
            # print(dataset, n_subjects_ds)
            dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

    print("Total pipelines to evaluate: ", len(pipelines))

    if evaluation_type == "withinsession":
        evaluation_P300 = WithinSessionEvaluation(
            paradigm=paradigm_P300,
            datasets=datasets_P300,
            suffix="examples",
            overwrite=overwrite,
            n_jobs=n_jobs,
            # n_jobs_evaluation=n_jobs,
            cache_config=cache_config,
        )
        evaluation_LR = WithinSessionEvaluation(
            paradigm=paradigm_LR,
            datasets=datasets_LR,
            suffix="examples",
            overwrite=overwrite,
            n_jobs=n_jobs,
            # n_jobs_evaluation=n_jobs,
            cache_config=cache_config,
        )
    elif evaluation_type == "crosssubject":
        evaluation_P300 = CrossSubjectEvaluation(
            paradigm=paradigm_P300,
            datasets=datasets_P300,
            suffix="examples",
            overwrite=overwrite,
            n_jobs=n_jobs,
            # n_jobs_evaluation=n_jobs,
            cache_config=cache_config,
        )
        evaluation_LR = CrossSubjectEvaluation(
            paradigm=paradigm_LR,
            datasets=datasets_LR,
            suffix="examples",
            overwrite=overwrite,
            n_jobs=n_jobs,
            # n_jobs_evaluation=n_jobs,
            cache_config=cache_config,
        )
    else:
        raise ValueError("Unknown evaluation type!")

    if not skip_P300:
        results_P300 = evaluation_P300.process(pipelines, param_grid=params_grid)

    if not skip_MI and replace_x_dawn_cov_par_cov_for_MI:
        # replace XDawnCovariances with Covariances when using MI or LeftRightMI
        for pipe_name in pipelines:
            pipeline = pipelines[pipe_name]
            if pipeline.steps[0][0] == "xdawncovariances":
                pipeline.steps.pop(0)
                pipeline.steps.insert(0, ["covariances", Covariances("oas")])
                print("xdawncovariances transformer replaced by covariances")

        results_LR = evaluation_LR.process(pipelines, param_grid=params_grid)

        if not skip_P300:
            results = pd.concat([results_P300, results_LR], ignore_index=True)
        else:
            results = results_LR

        # each MI dataset uses its own configured MI paradigm
        if len(datasets_MI) > 0:
            for paradigm_MI, dataset_MI in zip(paradigms_MI, datasets_MI):
                if evaluation_type == "withinsession":
                    evaluation_MI = WithinSessionEvaluation(
                        paradigm=paradigm_MI,
                        datasets=[dataset_MI],
                        overwrite=overwrite,
                        n_jobs=n_jobs,
                        # n_jobs_evaluation=n_jobs,
                        cache_config=cache_config,
                    )
                elif evaluation_type == "crosssubject":
                    evaluation_MI = CrossSubjectEvaluation(
                        paradigm=paradigm_MI,
                        datasets=[dataset_MI],
                        overwrite=overwrite,
                        n_jobs=n_jobs,
                        # n_jobs_evaluation=n_jobs,
                        cache_config=cache_config,
                    )

                results_per_MI_pardigm = evaluation_MI.process(
                    pipelines, param_grid=params_grid
                )
                results = pd.concat(
                    [results, results_per_MI_pardigm], ignore_index=True
                )
    else:
        results = results_P300

    return results


def _AdjustDF(df, removeP300=False, removeMI=False):
    """
    Allows the results to contain only P300 databases or only Motor Imagery databases.
    Adds "P" and "M" to each database name for each P300 and MI result.

    Parameters
    ----------
    df : Pandas dataframe
        A dataframe with results from the benchrmark.
    removeP300 : bool, default = False
        P300 results will be removed from the dataframe.
    removeMI_LR : bool, default = False
        Motor Imagery results will be removed from the dataframe.

    Returns
    -------
    df : Pandas dataframe
        Returns a dataframe with filtered results.

    """

    datasets_P300 = [
        "BrainInvaders2013a",
        "BNCI2014-008",
        "BNCI2014-009",
        "BNCI2015-003",
        "BrainInvaders2015a",
        "BrainInvaders2015b",
        # 'Sosulski2019',
        "BrainInvaders2014a",
        "BrainInvaders2014b",
        # 'EPFLP300'
    ]
    datasets_MI = [
        "BNCI2015-004",  # 5 classes,
        "BNCI2015-001",  # 2 classes
        "BNCI2014-002",  # 2 classes
        "AlexMI",  # 3 classes
    ]
    datasets_LR = [
        "BNCI2014-001",
        "BNCI2014-004",
        "Cho2017",  # 49 subjects
        "GrosseWentrup2009",
        "PhysionetMotorImagery",  # 109 subjects
        "Shin2017A",
        "Weibo2014",
        "Zhou2016",
        "Lee2019_MI",  # new
        "Schirrmeister2017",  # new
    ]
    for ind in df.index:
        dataset_classified = False
        if df["dataset"][ind] in datasets_P300:
            df["dataset"][ind] = df["dataset"][ind] + "_P"
            dataset_classified = True

        elif df["dataset"][ind] in datasets_MI or df["dataset"][ind] in datasets_LR:
            df["dataset"][ind] = df["dataset"][ind] + "_M"
            dataset_classified = True
        if not dataset_classified:
            print("This dataset was not classified:", df["dataset"][ind])

    if removeP300:
        df = df.drop(df[df["dataset"].str.endswith("_P", na=None)].index)

    if removeMI:
        df = df.drop(df[df["dataset"].str.endswith("_M", na=None)].index)

    return df


def plot_stat(results, removeP300=False, removeMI=False):
    """
    Generates a point plot for each pipeline.
    Generate statistical plots by comparing every 2 pipelines. Test if the
    difference is significant by using SMD. It does that per database and overall
    with the "Meta-effect" line.
    Generates a summary plot - a significance matrix to compare the pipelines.
    It uses as a heatmap
    with green/grey/red for significantly higher/significantly lower.

    Parameters
    ----------
    results : Pandas dataframe
        A dataframe with results from a benchmark
    removeMI : default = False
        Do not icnclude Motor Imagery datasets in the statistics plots.
    removeP300 : default = False
        Do not icnclude P300 datasets in the statistics plots.

    Returns
    -------
    None.

    """
    results = _AdjustDF(results, removeP300=removeP300, removeMI=removeMI)

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

    # Generate statistics for the summary plot
    # Computes matrices of p-values and effects for all algorithms
    # over all datasets via combined p-values and
    # combined effects methods
    stats = compute_dataset_statistics(results)
    P, T = find_significant_differences(stats)
    print(stats.to_string())  # not all datasets are in stats

    # Negative SMD value favors the first(left) algorithm,
    # postive SMD the second(right)
    # A meta-analysis style plot that shows the
    # standardized effect with confidence intervals over
    # all datasets for two algorithms.
    # Hypothesis is that alg1 is larger than alg2
    pipelines = results["pipeline"].unique()
    pipelines_sorted = sorted(pipelines)
    for i in range(0, len(pipelines_sorted)):
        for j in range(i + 1, len(pipelines_sorted)):
            moabb_plt.meta_analysis_plot(
                stats, pipelines_sorted[i], pipelines_sorted[j]
            )
            plt.show()

    # Summary Plot - provides a significance matrix to compare pipelines.
    # Visualizes the significances as a heatmap with green/grey/red
    # for significantly higher/significantly lower.
    moabb_plt.summary_plot(P, T)
    plt.show()

    print("Evaluation in %:")
    print(results.groupby("pipeline").mean("score")[["score", "time"]])

    print("Evaluation in % per database:")
    print(results.groupby(["dataset", "pipeline"]).mean("score")[["score", "time"]])
    # print(results.groupby(["dataset","pipeline"]).describe())
