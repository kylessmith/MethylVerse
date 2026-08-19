import numpy as np
import pandas as pd
from projectframe import ProjectFrame
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import fdrcorrection
from typing import List

# Local imports
from ...core.utilities import mean_methyl


def merge_peaks(peaks1: IntervalFrame,
                    peaks2: IntervalFrame,
                    gap: int = 1) -> IntervalFrame:
    """
    """

    new_peaks = LabeledIntervalArray()
    new_peaks.append(peaks1.index)
    new_peaks.append(peaks2.index)
    new_peaks = new_peaks.merge(gap)

    new_intervals = IntervalFrame(intervals = new_peaks)
    
    return new_intervals



def standardize_peaks(peaks: IntervalFrame,
                      standard_size: int) -> IntervalFrame:
    """
    """

    new_peaks = LabeledIntervalArray()
    half_size = int(standard_size / 2)
    midpoints = (peaks.starts + ((peaks.ends - peaks.starts) / 2)).astype(int)
    new_peaks.add(midpoints - half_size, midpoints + half_size, peaks.index.labels)
    new_peaks = new_peaks.merge()

    peaks = IntervalFrame(intervals = new_peaks)

    return peaks


def concensus_peaks(peaks: List[IntervalFrame],
                    gap: int = 1,
                    standard_size: int = None):
    """
    """

    # Get peaks
    total_peaks = peaks[0]

    # Iterate over observations
    for i, obs_peaks in enumerate(peaks):
        if i == 0:
            continue
        total_peaks = merge_peaks(total_peaks, obs_peaks, gap)

    # Standardize
    if standard_size is not None:
        total_peaks = standardize_peaks(total_peaks, standard_size)
        total_peaks = standardize_peaks(total_peaks, standard_size)
        ## Twice to make sure
        lengths = total_peaks.ends - total_peaks.starts
        total_peaks = total_peaks.iloc[lengths == standard_size,:]

    return total_peaks


def total_concensus_peaks(pf: ProjectFrame,
                            key: str = "hypomethylated_binary_regions",
                            anno_key: str = "binary_betas",
                            min_overlap: int = 1,
                            standard_size: int = None) -> ProjectFrame:
    """
    """

    # Get peaks
    obs = list(pf.obs)
    peaks = pf.obs_intervals[obs[0]][key]

    # Iterate over observations
    for o in obs[1:]:
        obs_peaks = pf.obs_intervals[o][key]
        peaks = merge_peaks(peaks, obs_peaks, min_overlap)

    # Standardize
    if standard_size is not None:
        peaks = standardize_peaks(peaks, standard_size)
        peaks = standardize_peaks(peaks, standard_size)
        ## Twice to make sure
        lengths = peaks.ends - peaks.starts
        peaks = peaks.iloc[lengths == standard_size,:]

    # Annotate peaks
    peaks = mean_methyl(peaks, pf.intervals[anno_key])

    # Assign to projectframe
    pf.add_intervals("concensus_peaks", peaks)

    return pf


def prop_test(pf: ProjectFrame,
              intervals: IntervalFrame,
              obs1: str,
              obs2: str,
              key: str = "binary_betas",
              alpha: float = 0.05,
              min_n: int = 3) -> IntervalFrame:
    """
    """

    new_peaks = IntervalFrame(intervals = intervals.index)
    new_peaks.df.loc[:,"stat"] = 0.0
    new_peaks.df.loc[:,obs1+"_n"] = 0
    new_peaks.df.loc[:,obs2+"_n"] = 0
    new_peaks.df.loc[:,obs1+"_methylated"] = 0
    new_peaks.df.loc[:,obs2+"_methylated"] = 0

    starts = new_peaks.starts
    ends = new_peaks.ends
    chroms = new_peaks.index.labels
    pvals = np.ones(new_peaks.shape[0])
    #stats = np.zeros(new_peaks.shape[0])
    for i in range(new_peaks.shape[0]):
        start = int(starts[i])
        end = int(ends[i])
        chrom = str(chroms[i])
        count = np.sum(pf.intervals[key].intersect(start, end, chrom).df.loc[:,[obs1, obs2]].values==1, axis=0)
        nobs = np.sum(pf.intervals[key].intersect(start, end, chrom).df.loc[:,[obs1, obs2]].values!=0, axis=0)
        #count = np.array([np.sum(pf1.intervals["betas"].intersect(start, end, chrom).df.loc[:,[obs1, obs2]].values==1), np.sum(signal_file2.intersect(start, end, chrom).df.loc[:,"betas"].values==1)])
        #nobs = np.array([signal_file.intersect(start, end, chrom).shape[0], signal_file2.intersect(start, end, chrom).shape[0]])
        stat, pval = proportions_ztest(count, nobs)
        pvals[i] = pval
        new_peaks.df.iloc[i,0] = stat
        new_peaks.df.iloc[i,1] = count[0]
        new_peaks.df.iloc[i,2] = count[1]
        new_peaks.df.iloc[i,3] = nobs[0]
        new_peaks.df.iloc[i,4] = nobs[1]

    # Correct
    pvals[~pd.isnull(pvals)] = fdrcorrection(pvals[~pd.isnull(pvals)])[1]

    # Add to results
    new_peaks.df.loc[:,"fdr_pval"] = pvals

    # Filter
    new_peaks = new_peaks.iloc[new_peaks.df.loc[:,"fdr_pval"].values<alpha,:]
    new_peaks = new_peaks.iloc[new_peaks.df.loc[:,"stat"].values>0,:]
    new_peaks = new_peaks.iloc[new_peaks.df.loc[:,obs1+"_n"].values>=min_n,:]

    return new_peaks


def compare_peaks(pf: ProjectFrame,
                  obs1: str,
                  obs2: str,
                  key: str = "hypomethylated_binary_regions",
                  min_overlap: int = 1,
                  alpha: float = 0.05,
                  min_n: int = 3) -> ProjectFrame:
    """
    """

    # Get peaks
    peaks1 = pf.obs_intervals[obs1][key]
    peaks2 = pf.obs_intervals[obs2][key]

    # Compare
    new_peaks = concensus_peaks(peaks1, peaks2, min_overlap)
    new_peaks.df.loc[:,"stat"] = 0.0
    new_peaks.df.loc[:,obs1+"_n"] = 0
    new_peaks.df.loc[:,obs2+"_n"] = 0
    new_peaks.df.loc[:,obs1+"_methylated"] = 0
    new_peaks.df.loc[:,obs2+"_methylated"] = 0

    starts = new_peaks.starts
    ends = new_peaks.ends
    chroms = new_peaks.index.labels
    pvals = np.zeros(new_peaks.shape[0])
    #stats = np.zeros(new_peaks.shape[0])
    for i in range(new_peaks.shape[0]):
        start = int(starts[i])
        end = int(ends[i])
        chrom = str(chroms[i])
        count = np.sum(pf.intervals["binary_betas"].intersect(start, end, chrom).df.loc[:,[obs1, obs2]].values==1, axis=0)
        nobs = np.sum(pf.intervals["binary_betas"].intersect(start, end, chrom).df.loc[:,[obs1, obs2]].values!=0, axis=0)
        #count = np.array([np.sum(pf1.intervals["betas"].intersect(start, end, chrom).df.loc[:,[obs1, obs2]].values==1), np.sum(signal_file2.intersect(start, end, chrom).df.loc[:,"betas"].values==1)])
        #nobs = np.array([signal_file.intersect(start, end, chrom).shape[0], signal_file2.intersect(start, end, chrom).shape[0]])
        stat, pval = proportions_ztest(count, nobs)
        pvals[i] = pval
        new_peaks.df.iloc[i,0] = stat
        new_peaks.df.iloc[i,1] = count[0]
        new_peaks.df.iloc[i,2] = count[1]
        new_peaks.df.iloc[i,3] = nobs[0]
        new_peaks.df.iloc[i,4] = nobs[1]

    # Correct
    pvals[~pd.isnull(pvals)] = fdrcorrection(pvals[~pd.isnull(pvals)])[1]

    # Add to results
    new_peaks.df.loc[:,"fdr_pval"] = pvals

    # Filter
    new_peaks = new_peaks.iloc[new_peaks.df.loc[:,"fdr_pval"].values<alpha,:]
    new_peaks = new_peaks.iloc[new_peaks.df.loc[:,"stat"].values>0,:]
    new_peaks = new_peaks.iloc[new_peaks.df.loc[:,obs1+"_n"].values>=min_n,:]

    # Assign to projectframe
    pf.add_obs_intervals(obs1, obs2+"_diff_peaks", new_peaks)

    return pf