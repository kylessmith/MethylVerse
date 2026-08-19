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
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.formula.api as smf


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
    unique_groups = pd.unique(group_labels)

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
        p_value = model.pvalues["C(group)[T." + str(unique_groups[1]) + "]"]
        p_values.append(p_value)

    return np.array(p_values)


def smooth_pvalues_rolling(pvalues, positions, max_dist=500):
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


def smooth_pvalues_window(pvalues, positions, window_size=1000):
    z_scores = -norm.ppf(np.clip(pvalues, 1e-10, 1 - 1e-10))
    smoothed_z = np.zeros_like(z_scores)
    
    # Sort by position
    idx = np.argsort(positions)
    sorted_pos = positions[idx]
    sorted_z = z_scores[idx]
    
    # Sliding window average
    for i in range(len(sorted_pos)):
        window = np.abs(sorted_pos - sorted_pos[i]) <= window_size
        smoothed_z[idx[i]] = np.mean(sorted_z[window])
    
    # Convert back to p-values
    return 2 * (1 - norm.cdf(np.abs(smoothed_z)))


def smooth_pvalues_spline(pvalues, positions, smoothing=0.5):
    z_scores = -norm.ppf(np.clip(pvalues, 1e-10, 1 - 1e-10))
    
    # Sort by position
    idx = np.argsort(positions)
    sorted_pos = positions[idx]
    sorted_z = z_scores[idx]
    
    # Fit spline (s controls smoothing strength)
    spline = UnivariateSpline(sorted_pos, sorted_z, s=smoothing*len(sorted_pos))
    smoothed_z = spline(positions)
    
    # Convert back to p-values
    return 2 * (1 - norm.cdf(np.abs(smoothed_z)))


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
    if len(significant_indices) == 0:
        return clusters
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

def call_dmps(data, group_labels, method='ttest', 
             covariates=None, alpha=0.05, correction='fdr_bh', 
             logit_transform=True, min_delta=0.05):
    """
    Find differentially methylated positions (DMPs) between groups using various statistical methods.
    
    Parameters:
    -----------
   data : IntervalFrame
        2D array with methylation beta values (shape: n_cpgs × n_samples)
    group_labels : numpy.ndarray
        1D array with group assignments for each sample
    method : str
        Statistical method to use: 'ttest', 'mannwhitney', 'logistic', or 'beta'
    covariates : numpy.ndarray or pandas.DataFrame, optional
        2D array of covariates (shape: n_samples × n_covariates)
    alpha : float
        Significance level for adjusted p-values
    correction : str
        Method for multiple testing correction (see statsmodels.stats.multitest)
    logit_transform : bool
        Whether to logit transform methylation values for t-tests (recommended)
    min_delta : float
        Minimum absolute methylation difference to consider a DMP significant
        
    Returns:
    --------
    pandas.DataFrame
        Results with statistics, p-values, and significance calls
    """
    methylation_data = data.df.values
    unique_groups = np.unique(group_labels)
    if len(unique_groups) != 2:
        raise ValueError("This function currently supports only two groups")
    
    n_cpgs = methylation_data.shape[0]
    n_samples = methylation_data.shape[1]
    
    if len(group_labels) != n_samples:
        raise ValueError(f"group_labels length ({len(group_labels)}) must match the number of samples ({n_samples})")
    
    # Convert group labels to binary (0/1)
    group_binary = np.zeros(n_samples)
    group_binary[group_labels == unique_groups[1]] = 1
    
    group1_idx = group_labels == unique_groups[0]
    group2_idx = group_labels == unique_groups[1]
    
    # Initialize results storage
    pvals = []
    stats_values = []
    delta_betas = []
    coef_values = []
    
    # Process each CpG site
    for cpg_idx in range(n_cpgs):
        # Extract methylation values for this CpG
        cpg_values = methylation_data[cpg_idx, :]
        
        # Calculate mean difference between groups
        mean_group1 = np.mean(cpg_values[group1_idx])
        mean_group2 = np.mean(cpg_values[group2_idx])
        delta = mean_group2 - mean_group1
        delta_betas.append(delta)
        
        # Apply statistical test based on selected method
        if method == 'ttest':
            values_group1 = cpg_values[group1_idx]
            values_group2 = cpg_values[group2_idx]
            
            # Apply logit transform if requested (handles bounds better)
            if logit_transform:
                # Avoid log(0) and log(1) issues
                epsilon = 1e-6
                values_group1 = np.clip(values_group1, epsilon, 1-epsilon)
                values_group2 = np.clip(values_group2, epsilon, 1-epsilon)
                values_group1 = np.log(values_group1 / (1 - values_group1))
                values_group2 = np.log(values_group2 / (1 - values_group2))
            
            t_stat, p_val = stats.ttest_ind(values_group1, values_group2, equal_var=False)
            pvals.append(p_val)
            stats_values.append(t_stat)
            coef_values.append(delta)
            
        elif method == 'mannwhitney':
            values_group1 = cpg_values[group1_idx]
            values_group2 = cpg_values[group2_idx]
            u_stat, p_val = stats.mannwhitneyu(values_group1, values_group2)
            pvals.append(p_val)
            stats_values.append(u_stat)
            coef_values.append(delta)
            
        elif method == 'logistic':
            # For logistic regression, methylation is predictor, group is outcome
            X = cpg_values.reshape(-1, 1)
            y = group_binary
            
            # Add intercept and standardize methylation values
            X = StandardScaler().fit_transform(X)
            X = sm.add_constant(X)
            
            # Add covariates if provided
            if covariates is not None:
                if isinstance(covariates, np.ndarray):
                    X = np.column_stack((X, covariates))
                else:
                    X = np.column_stack((X, covariates.values))
            
            try:
                # Fit logistic regression model
                model = sm.Logit(y, X)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = model.fit(disp=0, method='bfgs')
                
                # Extract coefficient and p-value for methylation effect (index 1)
                pvals.append(results.pvalues[1])
                stats_values.append(results.tvalues[1])
                coef_values.append(results.params[1])
            except:
                # Handle convergence issues
                pvals.append(1.0)
                stats_values.append(0.0)
                coef_values.append(0.0)
                
        elif method == 'beta':
            # Proper implementation for beta regression
            # Uses a linear model with logit-transformed methylation values
            
            # Avoid exact 0 and 1 values
            epsilon = 1e-6
            cpg_values_adj = np.clip(cpg_values, epsilon, 1-epsilon)
            
            # Logit transform methylation values - necessary for beta regression approximation
            logit_values = np.log(cpg_values_adj / (1 - cpg_values_adj))
            
            # Create design matrix with group and covariates
            X = sm.add_constant(group_binary.reshape(-1, 1))
            
            # Add covariates if provided
            if covariates is not None:
                if isinstance(covariates, np.ndarray):
                    X = np.column_stack((X, covariates))
                else:
                    X = np.column_stack((X, covariates.values))
            
            try:
                # Fit linear model on logit-transformed values
                model = sm.OLS(logit_values, X)
                results = model.fit()
                
                # Extract coefficient and p-value for group effect (index 1)
                pvals.append(results.pvalues[1])
                stats_values.append(results.tvalues[1])
                coef_values.append(results.params[1])
            except Exception as e:
                # Handle fitting issues
                pvals.append(1.0)
                stats_values.append(0.0)
                coef_values.append(0.0)
        else:
            raise ValueError(f"Method '{method}' not recognized. Use 'ttest', 'mannwhitney', 'logistic', or 'beta'")
    
    # Multiple testing correction
    pvals = np.array(pvals)
    pvals[pd.isnull(pvals)] = 1.0  # Replace NaNs with 1.0
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=correction)
    
    # Create results DataFrame
    results = IntervalFrame(intervals=data.index,
                            df = pd.DataFrame({
                            'cpg_index': range(n_cpgs),
                            'p_value': pvals,
                            'adjusted_p_value': pvals_corrected,
                            'statistic': stats_values,
                            'effect_size': coef_values,
                            'delta_beta': delta_betas,
                            'is_significant': reject & (np.abs(np.array(delta_betas)) >= min_delta)
                        }))
    
    return results


def detect_chrom_dmrs(pvalues,
                        positions,
                        chromosome,
                        length_scale=1000,
                        noise_level=1e-4,
                        pvalue_threshold=0.05,
                        max_gap=500,
                        min_cpgs=3,
                        smooth_method="spline"):
    # Step 1: Smooth p-values using GP regression
    if smooth_method == "spline":
        smoothed_pvalues = smooth_pvalues_spline(pvalues, positions)
    elif smooth_method == "window":
        smoothed_pvalues = smooth_pvalues_window(pvalues, positions)
    elif smooth_method == "gp":
        smoothed_pvalues = smooth_pvalues_gp(pvalues, positions, length_scale, noise_level)
    elif smooth_method == "rolling":
        smoothed_pvalues = smooth_pvalues_rolling(pvalues, positions, max_gap)
    else:
        raise ValueError("Invalid smoothing method. Choose 'spline', 'window', or 'gp'.")

    # Step 2: Identify CpG clusters
    clusters = identify_clusters(positions, smoothed_pvalues, pvalue_threshold, max_gap)

    if len(clusters) == 0:
        # No clusters found, return empty IntervalFrame
        empty_df = pd.DataFrame(columns=['cluster_id', 'start_position', 'end_position', 'num_cpgs', 'combined_pvalue', 'adjusted_pvalue'])
        empty_intervals = LabeledIntervalArray()
        return IntervalFrame(df=empty_df, intervals=empty_intervals)

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
    intervals.add(dmr_results['start_position'].values, dmr_results['end_position'].values, np.repeat(chromosome,dmr_results.shape[0]))
    dmr_intervals = IntervalFrame(df=dmr_results, intervals=intervals)
    dmr_intervals.df = dmr_intervals.df.drop(columns=['start_position', 'end_position'])

    return dmr_intervals


def detect_dmrs(pvalues,
                length_scale=1000,
                noise_level=1e-4,
                pvalue_threshold=0.05,
                max_gap=500,
                min_cpgs=3,
                smooth_method="spline",
                verbose=False):
    """
    Detect differentially methylated regions (DMRs) based on p-values and genomic positions.
    """

    # Iterate through chromosomes
    dmr_results = []
    unique_chromosomes = pvalues.index.unique_labels
    for chrom in unique_chromosomes:
        if verbose:
            print(f"Processing chromosome: {chrom}", flush=True)

        # Extract p-values and positions for this chromosome
        chrom_pvalues = pvalues.loc[chrom, :].df.loc[:,"p_value"].values
        chrom_positions = pvalues.loc[chrom, :].index.starts

        # Detect DMRs for this chromosome
        if len(chrom_pvalues) >= 10:
            dmr_intervals = detect_chrom_dmrs(chrom_pvalues,
                                            chrom_positions,
                                            chrom,
                                            length_scale=length_scale,
                                            noise_level=noise_level,
                                            pvalue_threshold=pvalue_threshold,
                                            max_gap=max_gap,
                                            min_cpgs=min_cpgs,
                                            smooth_method=smooth_method)
            dmr_results.append(dmr_intervals)

    # Combine results from all chromosomes
    dmrs = dmr_results[0].concat(dmr_results[1:])
    dmrs.df.loc[:,"cluster_id"] = np.arange(dmrs.df.shape[0])

    return dmrs