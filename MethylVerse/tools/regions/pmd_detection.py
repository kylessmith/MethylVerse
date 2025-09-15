import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import signal
import pandas as pd
from typing import List, Tuple
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray

from ...data.import_data import get_data_file


def calculate_distance_weights(window_positions: np.ndarray, all_positions: np.ndarray) -> np.ndarray:
    """
    Calculate weights for CpGs based on distance to second-next CpGs.
    
    Parameters:
    -----------
    window_positions : np.ndarray
        Positions of CpGs in the current window
    all_positions : np.ndarray
        All CpG positions (sorted)
    
    Returns:
    --------
    np.ndarray
        Weights for each CpG in the window
    """
    weights = np.ones(len(window_positions))
    
    for i, pos in enumerate(window_positions):
        # Find index of this position in the full array
        pos_idx = np.searchsorted(all_positions, pos)
        
        distances = []
        
        # Distance to second-previous CpG (index - 2)
        if pos_idx >= 2:
            dist_prev = pos - all_positions[pos_idx - 2]
            distances.append(dist_prev)
        
        # Distance to second-next CpG (index + 2)
        if pos_idx + 2 < len(all_positions):
            dist_next = all_positions[pos_idx + 2] - pos
            distances.append(dist_next)
        
        # Use inverse of average distance as weight (closer CpGs get higher weight)
        if distances:
            avg_distance = np.mean(distances)
            weights[i] = 1.0 / (avg_distance + 1)  # Add 1 to avoid division by zero
        else:
            weights[i] = 1.0  # Default weight for edge cases
    
    return weights


def merge_overlapping_regions(regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping genomic regions.
    
    Parameters:
    -----------
    regions : List[Tuple[int, int]]
        List of (start, end) tuples representing genomic regions
    
    Returns:
    --------
    List[Tuple[int, int]]
        List of merged non-overlapping regions
    """
    if not regions:
        return []
    
    # Sort regions by start position
    regions = sorted(regions)
    merged = [regions[0]]
    
    for current_start, current_end in regions[1:]:
        last_start, last_end = merged[-1]
        
        # If regions overlap, merge them
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    return merged


def call_pmds(beta_values: np.ndarray, positions: np.ndarray, 
              window_size: int = 10000, step_size: int = 1, 
              methylation_threshold: float = 0.6, min_pmd_size: int = 100000) -> List[Tuple[int, int]]:
    """
    Identify Partially Methylated Domains (PMDs) from methylation data.
    
    Parameters:
    -----------
    beta_values : np.ndarray
        Array of methylation beta values (0-1)
    positions : np.ndarray
        Array of genomic positions corresponding to beta values
    window_size : int
        Size of sliding window in bp (default: 10000)
    step_size : int
        Step size for sliding window in bp (default: 1)
    methylation_threshold : float
        Methylation threshold below which regions are considered hypomethylated (default: 0.6)
    min_pmd_size : int
        Minimum size for a region to be called a PMD in bp (default: 100000)
    
    Returns:
    --------
    List[Tuple[int, int]]
        List of tuples containing (start, end) positions of identified PMDs
    """
    
    # Sort data by position
    #sort_idx = np.argsort(positions)
    #positions = positions[sort_idx]
    #beta_values = beta_values[sort_idx]
    
    # Calculate weighted methylation for sliding windows
    hypomethylated_windows = []
    
    # Determine the range for sliding windows
    start_pos = positions[0]
    end_pos = positions[-1]
    
    print(f"Processing {len(positions)} CpGs from position {start_pos} to {end_pos}")
    
    # Slide window across the genome
    current_pos = start_pos
    while current_pos <= end_pos - window_size:
        window_start = current_pos
        window_end = current_pos + window_size
        
        # Find CpGs within the current window
        in_window = (positions >= window_start) & (positions < window_end)
        window_positions = positions[in_window]
        window_betas = beta_values[in_window]
        
        if len(window_betas) > 0:
            # Calculate weights based on distance to second-next CpGs
            weights = calculate_distance_weights(window_positions, positions)
            
            # Calculate weighted average methylation
            if np.sum(weights) > 0:
                weighted_avg = np.average(window_betas, weights=weights)
                
                # Check if window is hypomethylated
                if weighted_avg < methylation_threshold:
                    hypomethylated_windows.append((window_start, window_end))
        
        current_pos += step_size
    
    print(f"Found {len(hypomethylated_windows)} hypomethylated windows")
    
    # Merge overlapping windows
    merged_regions = merge_overlapping_regions(hypomethylated_windows)
    print(f"After merging: {len(merged_regions)} regions")
    
    # Filter for regions larger than minimum PMD size
    pmds = [(start, end) for start, end in merged_regions if end - start >= min_pmd_size]
    print(f"Final PMDs (>= {min_pmd_size} bp): {len(pmds)}")
    
    return pmds


def detect_pmds(beta_values, sample_name, gap=100, min_length=50000,
                threshold=(0.3, 0.7), window_size=101):
    """
    Detect PMDs (Partial Methylation Domains) in methylation data using a statistical approach.
    """

    # Isolate sample beta values
    beta_values = beta_values.loc[:,[sample_name]]

    # Remove nan values
    beta_values = beta_values.iloc[~pd.isnull(beta_values.df.values[:,0]),:]

    # Remove CpG islands
    islands = IntervalFrame.read_parquet(get_data_file("CpGislands_hg38.parquet"))
    island_overlap = beta_values.index.percent_coverage(islands.index)
    beta_values = beta_values.iloc[island_overlap < 0.5,:]
    print(beta_values.shape, flush=True)

    # Ensure beta_values is a pandas Series with positions as index
    #beta_values = pd.Series(beta_values, index=positions)

    # Initialize an empty LabeledIntervalArray to store PMD candidates
    ail = LabeledIntervalArray()

    # Iterate through each chromosome
    for chrom in beta_values.index.unique_labels:
        print("Processing chromosome:", chrom, flush=True)

        chrom_beta_values = beta_values.loc[chrom,:]
        positions = chrom_beta_values.index.starts
        chrom_beta_values = pd.Series(chrom_beta_values.df.values[:,0], index=positions)

        # 1. Smooth the beta values using a rolling window
        smoothed_values = chrom_beta_values.rolling(window=window_size, center=True).mean()

        # 2. Identify potential PMD regions based on threshold
        chosen = np.logical_and(smoothed_values > threshold[0], smoothed_values < threshold[1])

        # 3. Find contiguous segments of PMD candidates
        x1 = np.hstack([ [False], chosen, [False] ])  # padding
        d = np.diff(x1.astype(int))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            continue
        
        # Create segments
        ail.add(positions[starts], positions[ends-1],np.repeat(chrom,len(starts)))
    
    ail = ail.merge(gap)

    # 4. Filter segments based on length
    ail = ail[(ail.ends - ail.starts) >= min_length]

    # Create an IntervalFrame from the LabeledIntervalArray
    if ail.size > 0:
        iframe = IntervalFrame(intervals=ail)
    else:
        return IntervalFrame()
    
    # 5. Calculate average methylation in the PMD segments
    iframe.annotate(beta_values, sample_name, method='mean')
    keep = np.logical_and(iframe.df.loc[:,"mean"].values > threshold[0], iframe.df.loc[:,"mean"].values < threshold[1])
    iframe = iframe.iloc[keep,:]

    return iframe
