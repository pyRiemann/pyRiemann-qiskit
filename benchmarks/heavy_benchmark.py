"""
====================================================================
Benchmark Alpha
====================================================================

A benchmark on a predefined list of databases for P300 and Motor Imagery (LR).

Currently it requires the latest version of MOABB (from GIT) where:
    - cache_config is availabe for WithinSessionEvaluation|()
    - this bug is fixed: https://github.com/NeuroTechX/moabb/issues/514

Adapts both the pipeline and the paradigms depending on the evaluated databaase.
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


def benchmark_alpha(pipelines, max_n_subjects=-1, overwrite=False, n_jobs=12, skip_MR_LR = False):
    """

    Parameters
    ----------
    pipelines :
        Pipelines to test. The pipelines are expected to be configured for P300.
        When switching from P300 to Motor Imagery and if the first transformer
        is XdawnCovariances then it will be automatically replaced by Covariances()
    max_n_subjects : int, default = -1
        The maxmium number of subjects to be used per database.  
    overwrite : bool, optional
        Set to True if we want to overwrite cached results.
    n_jobs : int, default=12
        The number of jobs to use for the computation. It is used in WithinSessionEvaluation().
    skip_MR_LR : default = False
        Only P300 ERP databases will be used for this benchmark.
    
    Returns
    -------
    df : Pandas dataframe
        Returns a dataframe with results from the tests.
    

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

    if skip_MR_LR == False: 
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
    else:
        results = results_P300

    return results



def _AdjustDF(df, removeP300  = False, removeMI_LR = False):
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
    
    datasets_P300 = ['BrainInvaders2013a', 
                     'BNCI2014-008', 
                     'BNCI2014-009', 
                     'BNCI2015-003', 
                     'BrainInvaders2015a', 
                     'BrainInvaders2015b', 
                     #'Sosulski2019', 
                     'BrainInvaders2014a', 
                     'BrainInvaders2014b', 
                      #'EPFLP300'
                     ]
    datasets_MI = [ 'BNCI2015-004',  #5 classes, 
                    'BNCI2015-001',  #2 classes
                    'BNCI2014-002',  #2 classes
                    #'AlexMI',        #3 classes, Error: Classification metrics can't handle a mix of multiclass and continuous targets
                  ]
    datasets_LR = [ 'BNCI2014-001',
                    'BNCI2014-004',
                    'Cho2017',      #49 subjects
                    'GrosseWentrup2009',
                    'PhysionetMotorImagery',  #109 subjects
                    'Shin2017A', 
                    'Weibo2014', 
                    'Zhou2016',
                  ]
    for ind in df.index:
        dataset_classified = False
        if (df['dataset'][ind] in datasets_P300):
            df['dataset'][ind] = df['dataset'][ind] + "_P"
            dataset_classified = True
            
        elif (df['dataset'][ind] in datasets_MI or df['dataset'][ind] in datasets_LR): 
             df['dataset'][ind] = df['dataset'][ind] + "_M"
             dataset_classified = True
        if dataset_classified == False:
            print("This dataset was not classified:", df['dataset'][ind])
    
    if (removeP300):
        df = df.drop(df[df['dataset'].str.endswith('_P', na=None)].index)
            
    if (removeMI_LR):
        df = df.drop(df[df['dataset'].str.endswith('_M', na=None)].index)
            
    return df

def plot_stat(results, removeP300  = False, removeMI_LR = False):
    """
    Generates a point plot for each pipeline.
    Generate statistical plots by comparing every 2 pipelines. Test if the 
    difference is significant by using SMD. It does that per database and overall
    with the "Meta-effect" line. 
    Generates a summary plot - a significance matrix to compare the pipelines. It uses as a heatmap 
    with green/grey/red for significantly higher/significantly lower.

    Parameters
    ----------
    results : Pandas dataframe
        A dataframe with results from the benchmark
    removeP300 : TYPE, optional
        DESCRIPTION. The default is False.
    removeMI_LR : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    results = _AdjustDF(results, removeP300 = removeP300, removeMI_LR = removeMI_LR)
    
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
    
    print("Evaluation in %:")
    print(results.groupby("pipeline").mean("score")[["score", "time"]])
