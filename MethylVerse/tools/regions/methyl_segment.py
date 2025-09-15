import numpy as np
import pandas as pd
from projectframe import ProjectFrame
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray
import ngsfragments as ngs


# Local imports
from ..merge_regions.merge_regions import merge_less_than, merge_greater_than
from ...data.import_data import get_data_file


def get_solo_cpgs(betas: IntervalFrame):
    """
    """

    # Get island file
    file = get_data_file("CpGislands_hg38.parquet")
    islands = IntervalFrame.read_parquet(file)
    shelfs = LabeledIntervalArray()
    shelfs.add(islands.starts-4000, islands.ends+4000, islands.index.labels)
    shelfs = shelfs.merge()

    n = shelfs.nhits_from_LabeledIntervalArray(betas.index)
    solo = betas.iloc[n == 0,:]

    return solo


def hypomethylated_regions(betas: IntervalFrame,
                           column: str,
                           sort: bool = True,
                           n_cutoff: int = 5,
                           max_length: int = 500,
                           smooth: int = 0,
                           beta_cutoff: float = 0.6):
    """
    """

    # Sort
    if sort:
        iframe = betas.iloc[betas.index.sorted_index(),:]
    else:
        # Copy
        iframe = betas.copy()

    # Smooth
    if smooth > 0:
        iframe.df.loc[:,column] = iframe.df.loc[:,column].groupby(iframe.index.labels).rolling(smooth, center=True, win_type="gaussian").mean().values

    # Remove nans
    iframe = iframe.iloc[~pd.isnull(iframe.df.loc[:,column].values),:]
    
    # Merge
    merged_intervals = merge_less_than(iframe.index, iframe.df.loc[:,column].values, beta_cutoff)
    n = betas.index.nhits_from_LabeledIntervalArray(merged_intervals)
    merged_iframe = IntervalFrame(intervals=merged_intervals)
    merged_iframe.df.loc[:,"n"] = n
    merged_iframe.annotate(betas, column, method="mean")

    # Filter
    chosen = np.logical_and((merged_iframe.ends-merged_iframe.starts) < max_length, n > n_cutoff)
    merged_iframe = merged_iframe.iloc[chosen,:]

    return merged_iframe


def hypermethylated_regions(betas: IntervalFrame,
                           column: str,
                           sort: bool = True,
                           n_cutoff: int = 5,
                           max_length: int = 500,
                           smooth: int = 0,
                           beta_cutoff: float = 0.6):
    """
    """

    # Sort
    if sort:
        iframe = betas.iloc[betas.index.sorted_index(),:]
    else:
        # Copy
        iframe = betas.copy()

    # Smooth
    if smooth > 0:
        iframe.df.loc[:,column] = iframe.df.loc[:,column].groupby(iframe.index.labels).rolling(smooth, center=True, win_type="gaussian").mean().values

    # Remove nans
    iframe = iframe.iloc[~pd.isnull(iframe.df.loc[:,column].values),:]
    
    # Merge
    merged_intervals = merge_greater_than(iframe.index, iframe.df.loc[:,column].values, beta_cutoff)
    n = betas.index.nhits_from_LabeledIntervalArray(merged_intervals)
    merged_iframe = IntervalFrame(intervals=merged_intervals)
    merged_iframe.df.loc[:,"n"] = n
    merged_iframe.annotate(betas, column, method="mean")

    # Filter
    chosen = np.logical_and((merged_iframe.ends-merged_iframe.starts) < max_length, n > n_cutoff)
    merged_iframe = merged_iframe.iloc[chosen,:]

    return merged_iframe


def get_PMDs(betas: IntervalFrame,
            column: str,
            min_length: int = 50000,
            max_length: int = 10000000,
            beta_cutoff: int = 0.4):
    """
    """

    # Remove islands, shores, and shelves
    solo = get_solo_cpgs(betas)
    #mean_methyl = np.mean(betas.df.loc[:,column].values)

    # Determine pmds
    pmds = hypomethylated_regions(solo,
                                  column = column,
                                  sort=True,
                                  smooth=101,
                                  max_length = max_length,
                                  beta_cutoff = beta_cutoff)
    
    # Filter smaller regions
    pmds = pmds.iloc[(pmds.ends - pmds.starts) > min_length,:]

    return pmds


def get_LMRs(betas: IntervalFrame,
            column: str,
            n_cutoff: int = 5,
            min_length: int = 100,
            max_length: int = 1000,
            beta_cutoff: int = 0.4):
    """
    """

    # Determine LMRs
    lmrs = hypomethylated_regions(betas,
                                  column = column,
                                  sort = True,
                                  smooth = 3,
                                  n_cutoff = n_cutoff,
                                  max_length = max_length,
                                  beta_cutoff = beta_cutoff)
    
    # Filter UMRs
    lmrs = lmrs.iloc[lmrs.df.loc[:,"mean"].values > 0.1,:]
    
    # Filter smaller regions
    lmrs = lmrs.iloc[(lmrs.ends - lmrs.starts) > min_length,:]

    return lmrs


def hypermethylated_binary_regions(pf: ProjectFrame,
                            obs: str,
                            key: str = "binary_betas",
                            n_cutoff: int = 5,
                            verbose: bool = False) -> ProjectFrame:
    """
    """

    # Get signal
    signal_file = pf.intervals[key].loc[:,[obs]]

    # Iterate over chromosomes
    chroms = signal_file.index.unique_labels
    total_peaks = LabeledIntervalArray()
    for chrom in chroms:
        if verbose: print(chrom, flush=True)
        chrom_signal = signal_file.loc[chrom,:]
        position = chrom_signal.starts
        values = np.zeros(position[-1] + 1)
        values[position] = chrom_signal.df.loc[:,obs].values
        values = pd.Series(values, index = np.arange(values.shape[0]))
        #values = signal_file.loc[chrom,:].df.loc[:,obs].astype(float)
        #values.index = np.arange(values.shape[0])
        if pd.isnull(values).sum() > 0:
            continue
        peaks = ngs.peak_calling.CallPeaks.call_peaks(values, str(chrom), min_length=10, max_length=100)

        if peaks.size > 0:
            # Re-index segments
            #peaks.index_with_aiarray(signal_file.loc[chrom,:].index)
            total_peaks.append(peaks)
            #new_peaks = LabeledIntervalArray()
            #new_peaks.add(signal_file.starts[peaks.starts], signal_file.ends[peaks.ends], np.repeat(chrom, peaks.size))
            #total_peaks.append(new_peaks)

    # Create IntervalFrame
    hypermethylated_regions = IntervalFrame(total_peaks)
    hypermethylated_regions.annotate(signal_file, column=obs, method="mean")
    hypermethylated_regions.annotate(signal_file, column=obs, method="n")

    # Filter
    hypermethylated_regions = hypermethylated_regions.iloc[hypermethylated_regions.df.loc[:,"n"].values > n_cutoff,:]

    # Assign to projectframe
    pf.add_obs_intervals(obs, "hypermethylated_binary_regions", hypermethylated_regions)
    
    return pf


def hypomethylated_binary_regions(pf: ProjectFrame,
                            obs: str,
                            key: str = "binary_betas",
                            n_cutoff: int = 5,
                            mean_cutoff: float = -0.5,
                            min_length: int = 5,
                            max_length: int = 1000,
                            gap: int = 25,
                            verbose: bool = False) -> ProjectFrame:
    """
    """

    # Get signal
    signal_file = pf.intervals[key].loc[:,[obs]]

    # Iterate over chromosomes
    chroms = signal_file.index.unique_labels
    total_peaks = LabeledIntervalArray()
    for chrom in chroms:
        if verbose: print(chrom, flush=True)
        chrom_signal = signal_file.loc[chrom,:]
        position = chrom_signal.starts
        values = np.zeros(position[-1] + 1)
        values[position] = chrom_signal.df.loc[:,obs].values * -1
        values = pd.Series(values, index = np.arange(values.shape[0]))
        #values = signal_file.loc[chrom,:].df.loc[:,obs].astype(float)
        #values.index = np.arange(values.shape[0])
        #values = values * -1
        if pd.isnull(values).sum() > 0:
            continue
        peaks = ngs.peak_calling.CallPeaks.call_peaks(values, str(chrom), min_length=min_length, max_length=max_length, merge_distance=gap)

        if peaks.size > 0:
            # Re-index segments
            #peaks.index_with_aiarray(signal_file.loc[chrom,:].index)
            peaks = peaks.merge(gap)
            total_peaks.append(peaks)
            #new_peaks = LabeledIntervalArray()
            #new_peaks.add(signal_file.starts[peaks.starts], signal_file.ends[peaks.ends], np.repeat(chrom, peaks.size))
            #total_peaks.append(new_peaks)

    # Create IntervalFrame
    hypomethylated_regions = IntervalFrame(total_peaks)
    hypomethylated_regions.annotate(signal_file, column=obs, method="mean")
    hypomethylated_regions.annotate(signal_file, column=obs, method="n")

    # Filter
    hypomethylated_regions = hypomethylated_regions.iloc[hypomethylated_regions.df.loc[:,"n"].values > n_cutoff,:]
    hypomethylated_regions = hypomethylated_regions.iloc[hypomethylated_regions.df.loc[:,"mean"].values < mean_cutoff,:]

    # Assign to projectframe
    pf.add_obs_intervals(obs, "hypomethylated_binary_regions", hypomethylated_regions)
    
    return pf


def bin_data(signal_iframe: IntervalFrame,
             chromosome: str,
             key: str = "betas",
             genome_version : str = "hg38",
             bin_size: int = 10,
             inverse: bool = False) -> np.ndarray:
    """
    Bin data.

    Parameters
    ----------
        signal_iframe : IntervalFrame
            IntervalFrame of signal.
        chromosome : str
            Chromosome to bin.
        key : str
            Key to bin.
        genome_version : str
            Genome version.
        bin_size : int
            Bin size.
    
    Returns
    -------
        V : np.ndarray
            Binned signal.
    """

    # Get chromosome length
    import genome_info
    genome = genome_info.GenomeInfo(genome_version)
    try:
        chrom_length = genome["chrom_sizes"][chromosome]
    except KeyError:
        return None
    bin_results = IntervalFrame.from_dict_range({chromosome:chrom_length}, bin_size=bin_size)
    bin_scores = np.zeros(round(int(chrom_length) / bin_size) + 1)
    n = np.zeros(round(int(chrom_length) / bin_size) + 1)

    # Get signal
    signal_iframe = signal_iframe.loc[chromosome,:]
    starts = signal_iframe.starts
    ends = signal_iframe.ends
    if inverse:
        scores = signal_iframe.df.loc[:,key].values * -1
    else:
        scores = signal_iframe.df.loc[:,key].values

    # Bin
    for i in range(len(starts)):
        start = int(starts[i])
        end = int(ends[i])
        start_bin = int(start / bin_size)
        end_bin = int(end / bin_size)
        bins = np.arange(start_bin, end_bin + 1)
        bin_scores[bins] += scores[i]
        n[bins] += 1

    bin_scores[n > 0] = bin_scores[n > 0] / n[n > 0]
    if bin_results.shape[0] != len(bin_scores):
        bin_results.df.loc[:,"score"] = bin_scores[:-1]
    else:
        bin_results.df.loc[:,"score"] = bin_scores
    
    return bin_results


