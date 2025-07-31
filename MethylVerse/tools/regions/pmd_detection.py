import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import signal
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray


def adaptive_window_pmds(beta_values,
                         window_sizes=[155, 255, 995],
                         threshold=(0.3, 0.7), 
                         min_segment_length=750):
    """
    Use adaptive sliding windows to directly identify PMD segments based on beta value ranges.
    This method doesn't rely on HMM pre-classification.
    
    Parameters:
    -----------
    beta_values : array-like
        Array of methylation beta values (0-1)
    window_sizes : list of int
        Different window sizes to use for multi-scale analysis
    threshold : float or tuple
        If float: threshold for classifying a region as PMD (regions with averages between
        threshold and 1-threshold are considered PMDs)
        If tuple: (low_threshold, high_threshold) for PMD range
    min_segment_length : int
        Minimum length of segments to retain
        
    Returns:
    --------
    tuple
        (pmd_segments, segment_scores)
        pmd_segments: list of tuples (start_index, end_index) for each PMD
        segment_scores: array with PMD likelihood score for each position
    """
    if isinstance(threshold, (int, float)):
        # Symmetric threshold
        low_threshold = threshold
        high_threshold = 1 - threshold
    else:
        # Asymmetric threshold provided as tuple
        low_threshold, high_threshold = threshold
    
    # Multi-scale smoothing
    smoothed_values = np.zeros((len(window_sizes), len(beta_values)))
    
    for i, window in enumerate(window_sizes):
        # Edge-preserving smoothing using median filter
        smoothed_values[i] = signal.medfilt(beta_values, window)
    
    # Combine multi-scale information (average across scales)
    combined_smooth = np.mean(smoothed_values, axis=0)
    
    # Score each position by how likely it is to be a PMD
    # 1.0 = definitely PMD, 0.0 = definitely not PMD
    segment_scores = np.zeros(len(beta_values))
    
    for i, val in enumerate(combined_smooth):
        # Calculate distance from fully methylated or unmethylated
        if val < low_threshold:
            # Unmethylated region
            segment_scores[i] = 0.0
        elif val > high_threshold:
            # Fully methylated region
            segment_scores[i] = 0.0
        else:
            # PMD region - score based on how central it is in the PMD range
            # Highest score at the center of the PMD range
            mid_point = (low_threshold + high_threshold) / 2
            distance_from_mid = abs(val - mid_point)
            max_distance = (high_threshold - low_threshold) / 2
            segment_scores[i] = 1.0 - (distance_from_mid / max_distance)
    
    # Binarize scores with threshold for segment identification
    binary_mask = segment_scores > 0.5
    
    # Find contiguous segments
    pmd_segments = []
    in_segment = False
    start_pos = 0
    
    for i in range(len(binary_mask)):
        if binary_mask[i] and not in_segment:
            # Start of a new segment
            start_pos = i
            in_segment = True
        elif not binary_mask[i] and in_segment:
            # End of a segment
            if i - start_pos >= min_segment_length:
                pmd_segments.append((start_pos, i-1))
            in_segment = False
    
    # Handle case where the last segment extends to the end
    if in_segment and len(binary_mask) - start_pos >= min_segment_length:
        pmd_segments.append((start_pos, len(binary_mask)-1))
    
    return pmd_segments, segment_scores


def refine_segment_boundaries(beta_values,
                              segments,
                              window_size=5,
                              threshold=0.5):
    """
    Refine the boundaries of identified segments by looking for sharp transitions
    in methylation levels.
    
    Parameters:
    -----------
    beta_values : array-like
        Array of methylation beta values
    segments : list of tuples
        List of (start, end) tuples defining segments
    window_size : int
        Size of window to look for transitions
    threshold : float
        Threshold for change in beta value to consider a boundary
        
    Returns:
    --------
    list
        Refined segments as (start, end) tuples
    """
    refined_segments = []
    
    # Calculate derivative of beta values to detect transitions
    beta_gradient = np.abs(np.gradient(beta_values))
    
    for start, end in segments:
        # Look for a more precise start boundary
        # Search within window_size of original boundary
        search_start = max(0, start - window_size)
        search_end = min(len(beta_values) - 1, start + window_size)
        
        # Find position with maximum gradient in the search window
        max_gradient_pos = search_start + np.argmax(beta_gradient[search_start:search_end+1])
        refined_start = max_gradient_pos
        
        # Look for a more precise end boundary
        search_start = max(0, end - window_size)
        search_end = min(len(beta_values) - 1, end + window_size)
        
        max_gradient_pos = search_start + np.argmax(beta_gradient[search_start:search_end+1])
        refined_end = max_gradient_pos
        
        refined_segments.append((refined_start, refined_end))
    
    return refined_segments