import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import signal
import pandas as pd
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray

from ...data.import_data import get_data_file


def detect_umrs(beta_values, sample_name, pmds=None, gap=10, min_length=1000,
                threshold=0.4, window_size=10):
    """
    Detect UMRs (Unmethylated Regions) in methylation data using a statistical approach.
    """

    # Isolate sample beta values
    beta_values = beta_values.loc[:,[sample_name]]

    # Remove nan values
    beta_values = beta_values.iloc[~pd.isnull(beta_values.df.values[:,0]),:]

    # Remove CpG islands
    islands = IntervalFrame.read_parquet(get_data_file("CpGislands_hg38.parquet"))
    island_overlap = beta_values.index.percent_coverage(islands.index)
    beta_values = beta_values.iloc[island_overlap < 0.5,:]

    # Remove PMDs if provided
    if pmds is not None:
        pmd_overlap = beta_values.index.percent_coverage(pmds.index)
        beta_values = beta_values.iloc[pmd_overlap < 0.1,:]

    # Ensure beta_values is a pandas Series with positions as index
    #beta_values = pd.Series(beta_values, index=positions)

    # Initialize an empty LabeledIntervalArray to store UMR candidates
    ail = LabeledIntervalArray()

    # Iterate through each chromosome
    for chrom in beta_values.index.unique_labels:
        print("Processing chromosome:", chrom, flush=True)

        chrom_beta_values = beta_values.loc[chrom,:]
        positions = chrom_beta_values.index.starts
        chrom_beta_values = pd.Series(chrom_beta_values.df.values[:,0], index=positions)

        # 1. Smooth the beta values using a rolling window
        smoothed_values = chrom_beta_values.rolling(window=window_size, center=True).mean()

        # 2. Identify potential UMR regions based on threshold
        chosen = smoothed_values < threshold

        # 3. Find contiguous segments of UMR candidates
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
    
    # 5. Calculate average methylation in the UMR segments
    iframe.annotate(beta_values, sample_name, method='mean')
    keep = iframe.df.loc[:,"mean"].values < threshold
    iframe = iframe.iloc[keep,:]

    return iframe
