import numpy as np
import pandas as pd
from intervalframe import IntervalFrame
from ailist import AIList, LabeledIntervalArray
import genome_info

# Local imports
from ..stats.beta_dist2 import *
from ...data.import_data import get_data_file
from ...core.utilities import bin_cpgs
    
    
def hypomethyl_segment(data, alpha=0.75, beta=0.5, threshold_mean=0.3):
    
    segments = AIList()
    n_regions = data.shape[0]
    start = 0
    n = 0
    i = 0
    j = 1
    llr_last = -np.inf
    while i < n_regions and j < n_regions:
        x = data[i:j, :].flatten()
        mean, llr = beta_model_test(x, alpha=alpha, beta=beta)
        
        if llr > llr_last and mean < threshold_mean:
            if n == 0:
                start = i
            j += 1
            llr_last = llr
            n += 1

        else:
            if n > 0:
                segments.add(start, j-1)
                
            i = j
            j += 1
            start = i
            n = 0
            llr_last = -np.inf
    if n > 0:
        segments.add(start, j-1)
    
    return segments


def hypermethyl_segment(data, alpha=0.5, beta=0.75, threshold_mean=0.7):
    
    segments = AIList()
    n_regions = data.shape[0]
    start = 0
    n = 0
    i = 0
    j = 1
    llr_last = -np.inf
    while i < n_regions and j < n_regions:
        x = data[i:j, :].flatten()
        mean, llr = beta_model_test(x, alpha=alpha, beta=beta)
        
        if llr > llr_last and mean > threshold_mean:
            if n == 0:
                start = i
            j += 1
            llr_last = llr
            n += 1

        else:
            if n > 0:
                segments.add(start, j-1)
                
            i = j
            j += 1
            start = i
            n = 0
            llr_last = -np.inf
    if n > 0:
        segments.add(start, j-1)
    
    return segments


def bin_fit_beta(data: IntervalFrame,
                 bins: LabeledIntervalArray = None,
                 genome_version: str = "hg38",
                 bin_size: int = 100000,
                 min_n: int = 10,
                 update: bool = False) -> IntervalFrame:
    """
    """

    # Default parameters
    x = data.df.values.flatten()
    x = x[~np.isnan(x)]
    alpha = (1-np.mean(x)) * (np.mean(x)**2) / np.var(x) - np.mean(x)
    beta = alpha * (1 / np.mean(x) - 1)

    # Get bins
    if bins is None:
        genome = genome_info.GenomeInfo(genome_version)
        bins = LabeledIntervalArray.create_bin(genome["chrom_sizes"], bin_size=bin_size)

    params = np.zeros((len(bins),2))
    for i, interval in enumerate(bins):
        methyl_betas = data.intersect(interval.start, interval.end, interval.label)
        betas = methyl_betas.df.values.flatten()
        # Remove nans
        betas = betas[~np.isnan(betas)]
        if len(betas) > min_n:
            if update:
                fitted = update_beta_prior(alpha, beta, betas)
            else:
                try:
                    fitted = estimate_beta_params(betas)
                except RuntimeError:
                    fitted = (alpha, beta)
            params[i,0] = fitted[0]
            params[i,1] = fitted[1]
        else:
            params[i,0] = alpha
            params[i,1] = beta

    # Create IntervalFrame
    params = pd.DataFrame(params, columns=["alpha", "beta"])
    iframe = IntervalFrame(df=params, intervals=bins)
        
    return iframe


def bin_beta_mean_per_sample(data: IntervalFrame,
                    bins: LabeledIntervalArray = None,
                    genome_version: str = "hg38",
                    bin_size: int = 100000,
                    min_n: int = 10,
                    update: bool = False) -> IntervalFrame:
    """
    """

    # Get samples
    samples = data.df.columns
    n_samples = len(samples)

    # Default parameters
    default_params = {}
    for sample in samples:
        x = data.df.loc[:,sample].values.flatten()
        x = x[~np.isnan(x)]
        alpha = (1-np.mean(x)) * (np.mean(x)**2) / np.var(x) - np.mean(x)
        beta = alpha * (1 / np.mean(x) - 1)
        #mean = beta_mean(alpha, beta)
        mean = np.median(np.random.beta(alpha, beta, 10000))
        default_params[sample] = (alpha, beta, mean)

    # Get bins
    if bins is None:
        genome = genome_info.GenomeInfo(genome_version)
        bins = LabeledIntervalArray.create_bin(genome["chrom_sizes"], bin_size=bin_size)

    params = np.zeros((len(bins), n_samples))
    for i, interval in enumerate(bins):
        methyl_betas = data.intersect(interval.start, interval.end, interval.label)
        
        # Get sample means
        betas = methyl_betas.df.values
        for j, sample in enumerate(samples):
            x = betas[:,j]
            # Remove nans
            x = x[~np.isnan(x)]
            if len(x) > min_n:
                if update:
                    fitted = update_beta_prior(default_params[sample][0], default_params[sample][1], x)
                else:
                    try:
                        fitted = estimate_beta_params(x)
                    except RuntimeError:
                        fitted = (default_params[sample][0], default_params[sample][1])
                #params[i,j] = beta_mean(fitted[0], fitted[1])
                params[i,j] = np.median(np.random.beta(fitted[0], fitted[1], 10000))
            else:
                params[i,j] = default_params[sample][2]

    # Create IntervalFrame
    params = pd.DataFrame(params, columns=samples)
    iframe = IntervalFrame(df=params, intervals=bins)
        
    return iframe


def methyl_blocks(data: IntervalFrame,
                  blocks: LabeledIntervalArray | IntervalFrame | str = "MethylBlocks_hg38.parquet",
                  impute_nan: bool = False,
                  method: str = "median") -> IntervalFrame:
    """
    """

    # Get blocks
    if blocks == "MethylBlocks_hg38.parquet":
        blocks = IntervalFrame.read_parquet(get_data_file("MethylBlocks_hg38.parquet"))
    elif isinstance(blocks, str):
        blocks = IntervalFrame.read_parquet(blocks)
    elif isinstance(blocks, LabeledIntervalArray):
        blocks = IntervalFrame(intervals=blocks)
    
    # Calculate block values
    overlap = data.index.intersect_from_LabeledIntervalArray(blocks.index, return_intervals=False, return_index=True)
    means = data.df.iloc[overlap[1],:].groupby(overlap[0]).mean()
    new_blocks = pd.DataFrame(np.zeros((blocks.shape[0], data.shape[1])), columns = data.df.columns)
    new_blocks[:] = np.nan
    new_blocks.iloc[means.index.values,:] = means.values
    blocks = IntervalFrame(intervals=blocks.index, df=new_blocks)

    # Impute nan
    if impute_nan:
        # Fit beta per sample
        for sample in blocks.df.columns.values:
            x = data.df.loc[:,sample].values.flatten()
            x = x[~np.isnan(x)]
            alpha = (1-np.mean(x)) * (np.mean(x)**2) / np.var(x) - np.mean(x)
            beta = alpha * (1 / np.mean(x) - 1)
            if method == "median":
                default = beta_median(alpha, beta)
            elif method == "mean":
                default = beta_mean(alpha, beta)
            else:
                default = beta_median(alpha, beta)
            
            # Impute
            blocks.df.loc[blocks.df.loc[:,sample].isna(), sample] = default

    return blocks