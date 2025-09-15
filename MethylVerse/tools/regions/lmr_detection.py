import numpy as np
import pandas as pd
from scipy import stats
from hmmlearn import hmm
from ailist import LabeledIntervalArray
from intervalframe import IntervalFrame

from ...data.import_data import get_data_file


def detect_fhrs_hmm_based(beta_values, positions, chromosome,
                          min_sites=3, max_length=1000, n_states=3,
                          pvalue_threshold=0.05, smooth_window=3):
    """
    Detect focally hypomethylated regions using HMM for initial segmentation
    and then applying additional filtering criteria.
    
    Parameters:
    -----------
    beta_values : array-like
        Array of methylation beta values (0-1)
    positions : LabeledIntervalArray
        Genomic positions corresponding to beta values
    min_sites : int
        Minimum number of sites required in a hypomethylated region
    max_length : int
        Maximum length of a hypomethylated region in base pairs
    n_states : int
        Number of methylation states for HMM
    pvalue_threshold : float
        Threshold for statistical significance when comparing to surrounding regions
        
    Returns:
    --------
    list
        List of tuples (start_idx, end_idx, pvalue, avg_methylation, length) for each FHR
    """
    
    starts = positions.starts
    ends = positions.ends

    # Smooth the beta values
    smooth_values = pd.Series(beta_values).rolling(window=smooth_window, center=True).mean().values
    smooth_values[pd.isnull(smooth_values)] = 0.5  # Handle NaNs after smoothing
    
    # 1. Use HMM to segment the methylation data
    X = np.array(smooth_values).reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", 
                          n_iter=100, random_state=42)
    model.fit(X)
    state_sequence = model.predict(X)
    
    # 2. Determine which state represents hypomethylated regions
    state_means = np.array([model.means_[i][0] for i in range(n_states)])
    hypomethylated_state = np.argmin(state_means)
    
    # 3. Find contiguous regions of the hypomethylated state
    candidates = []
    start_idx = None
    
    for i, state in enumerate(state_sequence):
        if state == hypomethylated_state and start_idx is None:
            start_idx = i
        elif state != hypomethylated_state and start_idx is not None:
            candidates.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where the last region is hypomethylated
    if start_idx is not None:
        candidates.append((start_idx, len(state_sequence)-1))
    
    # 4. Filter candidates based on criteria
    fhrs = []
    fnhrs_intervals = LabeledIntervalArray()
    
    for start_idx, end_idx in candidates:
        # Check region length constraint
        if positions is not None:
            region_length = ends[end_idx] - starts[start_idx]
            if region_length > max_length:
                continue
        
        # Check minimum number of sites
        site_count = end_idx - start_idx + 1
        if site_count < min_sites:
            continue
        
        # Statistical test: compare with surrounding regions
        # Define surrounding regions (equal total length to focal region)
        region_size = end_idx - start_idx + 1
        
        # Left surrounding region
        left_surround_start = max(0, start_idx - region_size)
        left_surround_end = start_idx
        left_surround = beta_values[left_surround_start:left_surround_end]
        
        # Right surrounding region
        right_surround_start = end_idx + 1
        right_surround_end = min(len(beta_values), end_idx + 1 + region_size)
        right_surround = beta_values[right_surround_start:right_surround_end]
        
        # Combine surrounding regions
        surrounding = np.concatenate([left_surround, right_surround])
        
        # Focal region
        focal = beta_values[start_idx:end_idx+1]
        
        # Statistical test (Mann-Whitney U test)
        _, pvalue = stats.mannwhitneyu(surrounding, focal, alternative='greater')
        
        if pvalue < pvalue_threshold:
            # Calculate average methylation in the region
            avg_methylation = np.mean(focal)
            
            fhrs.append((pvalue, avg_methylation, site_count))
            fnhrs_intervals.add(starts[start_idx], ends[end_idx] + 1, chromosome)

    # Convert to IntervalFrame
    if len(fhrs) == 0:
        return IntervalFrame()
    
    fhrs = pd.DataFrame(np.array(fhrs))
    #print(fhrs, flush=True)
    fhrs = IntervalFrame(df=fhrs, intervals=fnhrs_intervals)
    fhrs.df.columns = ["pvalue", "avg_methylation", "length"]
    
    return fhrs


def detect_lmrs(betas: IntervalFrame,
                sample: str,
                pmds: IntervalFrame = None,
                umrs: IntervalFrame = None,
                min_sites: int = 3,
                max_length: int = 1000,
                n_states: int = 3,
                pvalue_threshold: float = 0.05,
                smooth_window: int = 3) -> IntervalFrame:
    """
    Detect focally hypermethylated regions (LMRs) using HMM for initial segmentation
    and then applying additional filtering criteria.
    
    Parameters:
    -----------
    betas : IntervalFrame
        Methylation data
    sample : str
        Sample name
    pmds : IntervalFrame, optional
        PMD regions to exclude from LMR detection
    min_sites : int
        Minimum number of sites required in a hypermethylated region
    max_length : int
        Maximum length of a hypermethylated region in base pairs
    n_states : int
        Number of methylation states for HMM
    pvalue_threshold : float
        Threshold for statistical significance when comparing to surrounding regions
    smooth_window : int
        Window size for smoothing beta values before HMM
        
    Returns:
    --------
    IntervalFrame
        List of tuples (start_idx, end_idx, pvalue, avg_methylation, length) for each LMR
    """

    # Get sample beta values
    betas = betas.loc[:,[sample]]

    # Remove nan values
    betas = betas.iloc[~pd.isnull(betas.df.values[:,0]),:]
    
    if pmds is not None and pmds.shape[0] > 0:
        # Exclude PMD regions from beta values
        print("Excluding PMD regions from LMR detection", flush=True)
        pmd_overlap = betas.index.percent_coverage(pmds.index)
        betas = betas.iloc[pmd_overlap < 0.1,:]

    # Remove CpG islands
    islands = IntervalFrame.read_parquet(mv.data.get_data_file("CpGislands_hg38.parquet"))
    island_overlap = betas.index.percent_coverage(islands.index)
    betas = betas.iloc[island_overlap < 0.5,:]

    # Extract beta values and positions from the IntervalFrame
    lmr_results = LabeledIntervalArray()
    chromosomes = betas.index.unique_labels
    for chrom in chromosomes:
        print("Processing chromosome:", chrom, flush=True)
        
        # Extract beta values and positions
        beta_values = betas.loc[chrom,:].df.loc[:,sample].values
        positions = betas.loc[chrom,:].index
        
        if len(beta_values) < 100:
            print("Skipping chromosome", chrom, "due to insufficient data", flush=True)
            continue
        # Call the detect_fhrs_hmm_based function to find LMRs
        lmrs = detect_fhrs_hmm_based(beta_values, positions, chrom,
                                                 min_sites, max_length, n_states,
                                                 pvalue_threshold, smooth_window)
        if lmrs.shape[0] > 0:
            lmr_results.append(lmrs.index)
    
    # Call the detect_fhrs_hmm_based function to find LMRs
    lmr_results = IntervalFrame(intervals=lmr_results)
    lmr_results.annotate(betas, sample, method='mean')
    lmr_results.annotate(betas, sample, method='n')
    lmr_results.df.columns = ["avg_methylation", "nCpGs"]

    # Filter LMRs based on avg methylation
    keep = lmr_results.df.loc[:,"avg_methylation"].values < 0.5
    lmr_results = lmr_results.iloc[keep,:]

    # Remove UMRs
    if umrs is not None and umrs.shape[0] > 0:
        umrs_overlap = lmr_results.index.percent_coverage(umrs.index)
        lmr_results = lmr_results.iloc[umrs_overlap == 0,:]

    return lmr_results