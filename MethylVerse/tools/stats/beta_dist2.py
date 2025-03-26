import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta, norm, combine_pvalues
from scipy.special import logit
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.stats.multitest import multipletests
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from ailist import LabeledIntervalArray
from intervalframe import IntervalFrame


def beta_mean(a, b):
    """
    Compute the mean of a Beta distribution.
    """
    return a / (a + b)


def beta_var(a, b):
    """
    Compute the variance of a Beta distribution.
    """
    return (a * b) / ((a + b)**2 * (a + b + 1))


def beta_std(a, b):
    """
    Compute the standard deviation of a Beta distribution.
    """
    return np.sqrt(beta_var(a, b))


def beta_median(a, b):
    """
    Compute the median of a Beta distribution.
    """
    return beta.ppf(0.5, a, b)


def update_beta_prior(alpha, beta, x):
    """
    Update the prior parameters of a Beta distribution based on observed data.
    alpha, beta: initial parameters of the Beta distribution.
    x: array of observed values (assumed to be in (0,1)).
    """
    # Count successes and failures
    successes = np.sum(x)
    failures = len(x) - successes
    # Update parameters
    new_alpha = alpha + successes
    new_beta = beta + failures
    return new_alpha, new_beta


def beta_neg_log_likelihood(params, x):
    """
    Compute the negative log likelihood for the Beta distribution.
    x: array of methylation values (assumed to be in (0,1)).
    params: tuple (a, b) for the Beta distribution parameters.
    """
    a, b = params
    # enforce a, b > 0
    if a <= 0 or b <= 0:
        return np.inf
    n = len(x)
    # The log likelihood for a Beta distribution
    ll = np.sum((a - 1) * np.log(x) + (b - 1) * np.log(1 - x)) - n * betaln(a, b)
    return -ll  # return negative log-likelihood for minimization


def estimate_beta_params(x, init_params=(2.0, 2.0)):
    """
    Estimate Beta distribution parameters from data using maximum likelihood.
    x: array of methylation values (must be in (0,1); values are clipped to avoid 0,1 issues).
    init_params: initial guess for (a, b).
    """
    # Handle edge cases where all values are close to 0 or 1
    if np.all(x <= 0) or np.all(x >= 1):
        raise ValueError("All values must be within the open interval (0, 1).")

    # Clip x slightly to avoid issues with log(0)
    x = np.clip(x, 1e-6, 1 - 1e-6)
    result = minimize(beta_neg_log_likelihood, x0=init_params, args=(x,),
                      bounds=[(1e-3, None), (1e-3, None)])
    if result.success:
        return result.x  # returns (a, b)
    else:
        raise RuntimeError("Beta parameter estimation failed.")


def log_likelihood_beta(x, a, b):
    """
    Compute the log likelihood of data x under a Beta(a, b) model.
    """
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.sum((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b))


def beta_model_test(x, alpha=2, beta=1):
    """
    Compare fitted Beta model to a reference high methylation Beta model.

    Parameters:
    - x: array of methylation values (assumed to be in (0,1)).
    - alpha, beta: parameters of the reference Beta distribution.

    Returns:
    - mle_mean: Mean of the estimated Beta distribution.
    - llr: Log-likelihood ratio comparing fitted model to reference.
    """
    try:
        # Estimate Beta parameters (MLE) for this region
        a_mle, b_mle = estimate_beta_params(x)
    except (RuntimeError, ValueError):
        print("Parameter estimation failed.")
        return None, None

    # Compute the estimated mean methylation level for the region
    mle_mean = a_mle / (a_mle + b_mle)

    # Compute log likelihoods under the estimated model and under the fixed high methylation model
    ll_mle = log_likelihood_beta(x, a_mle, b_mle)
    ll_high = log_likelihood_beta(x, alpha, beta)

    # Log-likelihood ratio
    llr = ll_mle - ll_high

    return mle_mean, llr


def beta_zscore(x_obs, alpha=2, beta_param=1):
    """
    Compute a standardized score for observed methylation values under a Beta distribution.
    """
    mean_beta = beta_mean(alpha, beta_param)
    std_beta = beta_std(alpha, beta_param)

    # Compute "Beta z-score"
    z_beta = (x_obs - mean_beta) / std_beta

    # Compute percentile-based score (p-value equivalent)
    p_value = beta.cdf(x_obs, alpha, beta_param)

    # Convert percentile to standard normal z-score equivalent
    z_p = norm.ppf(p_value)

    return z_beta, z_p, p_value


def beta_tscore(x, y):
    """
    Compute a t-like statistic comparing the means of two Beta-distributed datasets
    using the Delta method for more accurate variance estimation.

    Parameters:
    - x: First dataset (array of methylation values in (0, 1)).
    - y: Second dataset (array of methylation values in (0, 1)).

    Returns:
    - t_beta: The t-like statistic comparing the means of the two datasets.
    - p_value: Two-tailed p-value for the difference in means.
    """

    # Remove NaNs
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Fit Beta distribution to data
    x_alpha, x_beta = estimate_beta_params(x)
    y_alpha, y_beta = estimate_beta_params(y)

    # Means of the Beta distributions
    x_mean = beta_mean(x_alpha, x_beta)
    y_mean = beta_mean(y_alpha, y_beta)

    # Variances using the delta method
    x_var = (x_alpha * x_beta) / ((x_alpha + x_beta) ** 2 * (x_alpha + x_beta + 1))
    y_var = (y_alpha * y_beta) / ((y_alpha + y_beta) ** 2 * (y_alpha + y_beta + 1))

    # Standard error of the difference in means
    se_diff = np.sqrt(x_var / len(x) + y_var / len(y))

    # T-like statistic
    t_beta = (x_mean - y_mean) / se_diff

    # Compute two-tailed p-value using normal approximation
    p_value = 2 * (1 - norm.cdf(np.abs(t_beta)))

    return t_beta, p_value


def calc_autocorrelation(methylation_matrix, cpg_positions, max_lag=500):
    """
    Calculate spatial autocorrelation between CpG sites.

    Parameters:
    - methylation_matrix: 2D numpy array (CpGs as rows, samples as columns) with methylation values (0-1 range).
    - cpg_positions: 1D array of genomic positions corresponding to each CpG row.
    - max_lag: Maximum genomic distance to consider for autocorrelation.

    Returns:
    - autocorrelations: List of autocorrelation values for each lag distance.
    """
    # Ensure correct input shapes
    assert methylation_matrix.shape[0] == len(cpg_positions), "Mismatch between rows and CpG positions."

    # Number of CpG sites
    n_cpgs, n_samples = methylation_matrix.shape

    # Mean-center the methylation values across samples
    centered_methylation = methylation_matrix - np.mean(methylation_matrix, axis=1, keepdims=True)

    autocorrelations = []

    # Loop over possible lags (spatial distances between CpGs)
    for lag in range(1, max_lag + 1):
        # Track correlation at this lag
        valid_pairs = []
        correlations = []

        # Compare each CpG with its neighbors within the lag range
        for i in range(n_cpgs - lag):
            # Find the next CpG within the lag
            j = i + 1
            while j < n_cpgs and (cpg_positions[j] - cpg_positions[i]) <= lag:
                # Calculate Pearson correlation between methylation levels
                corr = np.corrcoef(centered_methylation[i], centered_methylation[j])[0, 1]
                correlations.append(corr)
                valid_pairs.append((i, j))
                j += 1

        # Average correlation for this lag
        if correlations:
            autocorrelations.append(np.mean(correlations))
        else:
            autocorrelations.append(np.nan)  # No valid pairs for this lag

    return np.array(autocorrelations)


def beta_regression_pvalues(methylation_matrix, group_labels):
    """
    Perform Beta regression for each CpG site to calculate p-values for group differences.

    Parameters:
    - methylation_matrix: 2D numpy array (CpGs as rows, samples as columns) with methylation values in (0, 1).
    - group_labels: 1D array-like with sample group labels (categorical).

    Returns:
    - p_values: Array of p-values for each CpG site.
    """
    n_cpgs, n_samples = methylation_matrix.shape
    group_labels = np.array(group_labels)

    # Ensure the matrix and labels are aligned
    assert len(group_labels) == n_samples, "Mismatch between samples and labels."

    # Preprocess methylation values (clip to avoid log(0) issues)
    methylation_matrix = np.clip(methylation_matrix, 1e-6, 1 - 1e-6)

    p_values = []

    for cpg_idx in range(n_cpgs):
        # Extract methylation values for this CpG site
        methylation_values = methylation_matrix[cpg_idx, :]

        # Prepare the data for regression
        df = pd.DataFrame({
            'methylation': methylation_values,
            'group': group_labels
        })

        # Apply logit transformation to map (0, 1) → (-inf, inf)
        df['methylation_logit'] = logit(df['methylation'])

        # Fit a Beta regression model (using logit link function for the mean)
        model = glm("methylation_logit ~ C(group)", data=df,
                    family=sm.families.Binomial()).fit()

        # Extract p-value for the group effect
        p_value = model.pvalues["C(group)[T.1]"]
        p_values.append(p_value)

    return np.array(p_values)


def smooth_pvalues(pvalues, positions, max_dist=500):
    """
    Smooth p-values using spatial correlation between CpG sites.

    Parameters:
    - pvalues: Array of raw p-values for each CpG site.
    - positions: Array of genomic positions for each CpG site.
    - max_dist: Maximum genomic distance to consider for smoothing.

    Returns:
    - smoothed_pvalues: Smoothed p-values accounting for spatial correlation.
    """
    n = len(pvalues)
    smoothed_pvalues = np.zeros(n)

    for i in range(n):
        # Identify neighboring CpGs within the specified distance
        neighbors = np.where(np.abs(positions - positions[i]) <= max_dist)[0]

        # Use Fisher’s method to combine p-values of neighboring CpGs
        combined_pval = combine_pvalues(pvalues[neighbors], method='fisher')[1]
        smoothed_pvalues[i] = combined_pval

    return smoothed_pvalues


def correct_pvalues_autocorr(pvalues, positions, max_dist=500, method='fdr_bh'):
    """
    Perform p-value correction accounting for spatial correlation.

    Parameters:
    - pvalues: Array of raw p-values for each CpG site.
    - positions: Array of genomic positions for each CpG site.
    - max_dist: Maximum genomic distance for smoothing.
    - method: Multiple testing correction method (default: 'fdr_bh').

    Returns:
    - adjusted_pvalues: P-values adjusted for multiple testing.
    """
    # Step 1: Smooth p-values using spatial autocorrelation
    smoothed_pvalues = smooth_pvalues(pvalues, positions, max_dist)

    # Step 2: Apply multiple testing correction (Benjamini-Hochberg or others)
    _, adjusted_pvalues, _, _ = multipletests(smoothed_pvalues, method=method)

    return adjusted_pvalues


def smooth_pvalues_gp(pvalues, positions, length_scale=1000.0, noise_level=1e-4):
    positions = np.array(positions).reshape(-1, 1)

    # Transform p-values to z-scores
    z_scores = -norm.ppf(np.clip(pvalues, 1e-10, 1 - 1e-10))

    # Gaussian Process with RBF kernel (for smoothing) and noise (to prevent overfitting)
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

    # Fit GP to z-scores
    gp.fit(positions, z_scores)

    # Predict smoothed z-scores
    smoothed_z = gp.predict(positions)

    # Convert back to p-values
    smoothed_pvalues = 2 * (1 - norm.cdf(np.abs(smoothed_z)))

    return smoothed_pvalues

def identify_clusters(positions, pvalues, threshold=0.05, max_gap=500):
    significant_indices = np.where(pvalues < threshold)[0]
    clusters = []
    current_cluster = [significant_indices[0]]

    for i in range(1, len(significant_indices)):
        if positions[significant_indices[i]] - positions[significant_indices[i - 1]] <= max_gap:
            current_cluster.append(significant_indices[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [significant_indices[i]]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def combine_cluster_pvalues(pvalues, clusters):
    combined_pvalues = []

    for cluster in clusters:
        cluster_pvalues = pvalues[cluster]
        if len(cluster_pvalues) == 1:
            combined_pvalues.append(cluster_pvalues[0])
        else:
            stat, combined_p = combine_pvalues(cluster_pvalues, method='fisher')
            combined_pvalues.append(combined_p)

    return np.array(combined_pvalues)

def detect_dmrs(pvalues, positions, length_scale=1000, noise_level=1e-4, pvalue_threshold=0.05, max_gap=500, min_cpgs=3):
    # Step 1: Smooth p-values using GP regression
    smoothed_pvalues = smooth_pvalues_gp(pvalues, positions, length_scale, noise_level)
    #smoothed_pvalues = smooth_pvalues(pvalues, positions, max_dist=max_gap)

    # Step 2: Identify CpG clusters
    clusters = identify_clusters(positions, smoothed_pvalues, pvalue_threshold, max_gap)

    # Step 3: Filter clusters by minimum CpG count
    clusters = [c for c in clusters if len(c) >= min_cpgs]

    # Step 4: Combine p-values within each cluster
    combined_pvalues = combine_cluster_pvalues(smoothed_pvalues, clusters)

    # Step 5: Adjust cluster p-values for multiple testing
    _, adjusted_pvalues, _, _ = multipletests(combined_pvalues, method='fdr_bh')

    # Prepare DMR results
    dmr_results = pd.DataFrame({
        'cluster_id': range(len(clusters)),
        'start_position': [positions[c[0]] for c in clusters],
        'end_position': [positions[c[-1]] for c in clusters],
        'num_cpgs': [len(c) for c in clusters],
        'combined_pvalue': combined_pvalues,
        'adjusted_pvalue': adjusted_pvalues
    })

    # Create IntervalFrame for DMRs
    intervals = LabeledIntervalArray()
    intervals.add(dmr_results['start_position'].values, dmr_results['end_position'].values, np.repeat('chr1',dmr_results.shape[0]))
    dmr_intervals = IntervalFrame(df=dmr_results, intervals=intervals)
    dmr_intervals.df = dmr_intervals.df.drop(columns=['start_position', 'end_position'])

    return dmr_intervals