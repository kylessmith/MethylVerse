import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist
from kneed import KneeLocator
from joblib import Parallel, delayed
import warnings
import time
from scipy.signal import argrelmax


def standardize_methylation_nmf(X, H, W=None, hist_bins=50, method='peaks'):
    """
    Standardize NMF Component ranges to match original
    """
    # Scale H to match the range of original data
    # Using peak normalization method
    if method == 'peaks':
        # Calculate histogram of original data mean across samples
        original_range = X.mean(axis=0)
        hist = np.histogram(original_range, bins=hist_bins)
        m = argrelmax(hist[0], order=3)[0]

        if len(m) > 1:
            minimum = hist[1][m[0]]
            global_maximum = hist[1][m[-1]]
        else:
            global_maximum = 0.85

        for i in range(H.shape[0]):
            
            if W is not None:
                original_range = np.average(X, axis=0, weights=W[:, i])
                hist = np.histogram(original_range, bins=hist_bins)
                m = argrelmax(hist[0], order=3)[0]
        
                if len(m) > 1:
                    minimum = hist[1][m[0]]
                    maximum = hist[1][m[-1]]
                else:
                    maximum = global_maximum
            else:
                maximum = global_maximum

            hist = np.histogram(H[i,:], bins=50)
            peaks = argrelmax(hist[0], order=3)[0]
            if len(peaks) == 0:
                continue
            m = peaks[-1]
            new_h = H[i,:] / (hist[1][m] / maximum)
            new_h = np.clip(new_h, 0, 1)
            H[i,:] = new_h

    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        original_min = X.min()
        original_max = X.max()
        
        for i in range(H.shape[0]):
            h_min = H[i,:].min()
            h_max = H[i,:].max()
            if h_max - h_min > 0:
                H[i,:] = (H[i,:] - h_min) / (h_max - h_min) * (original_max - original_min) + original_min
            else:
                H[i,:] = np.clip(H[i,:], original_min, original_max)

    else:
        raise ValueError("Method must be 'peaks' or 'minmax'")
    
    # Recalculate W
    H1_inv = np.linalg.pinv(H)
    W = np.array(np.dot(X, H1_inv))
        
    return W, H
    

def _fit_nmf_single_component(X, n_components, n_runs, base_random_state, 
                             init, max_iter, standardize_method, hist_bins, verbose_level):
    """
    Fit NMF with a specific number of components using multiple random initializations.
    
    Returns:
    --------
    dict: Results for this component count including averaged metrics
    """
    cosine_distances_runs = []
    reconstruction_errors_runs = []
    
    for run in range(n_runs):
        random_state = base_random_state + run if base_random_state is not None else None
        
        # Fit NMF
        nmf = NMF(n_components=n_components, 
                  init=init, 
                  random_state=random_state,
                  max_iter=max_iter)
        
        W = nmf.fit_transform(X)  # Sample loadings
        H = nmf.components_      # Feature loadings (components × features)
        
        # Standardize components
        W, H = standardize_methylation_nmf(X, H, W=W, hist_bins=hist_bins, method=standardize_method)
        
        # Calculate reconstruction error
        reconstruction_error = nmf.reconstruction_err_
        reconstruction_errors_runs.append(reconstruction_error)
        
        # Calculate average cosine distance between components
        if n_components > 1:
            cosine_dist_matrix = cosine_distances(H)
            upper_triangle_indices = np.triu_indices_from(cosine_dist_matrix, k=1)
            pairwise_distances = cosine_dist_matrix[upper_triangle_indices]
            avg_cosine_distance = np.mean(pairwise_distances)
        else:
            avg_cosine_distance = 0.0  # Single component case
            
        cosine_distances_runs.append(avg_cosine_distance)
    
    # Average across runs
    avg_cosine_distance = np.mean(cosine_distances_runs)
    avg_reconstruction_error = np.mean(reconstruction_errors_runs)
    std_cosine_distance = np.std(cosine_distances_runs)
    std_reconstruction_error = np.std(reconstruction_errors_runs)
    
    if verbose_level >= 2:
        print(f"    {n_components} components: "
              f"Cosine dist = {avg_cosine_distance:.4f} ± {std_cosine_distance:.4f}, "
              f"Recon err = {avg_reconstruction_error:.2f} ± {std_reconstruction_error:.2f}")
    elif verbose_level >= 1:
        print(f"  {n_components} components: {avg_cosine_distance:.4f} ± {std_cosine_distance:.4f}")
    
    return {
        'n_components': n_components,
        'avg_cosine_distance': avg_cosine_distance,
        'std_cosine_distance': std_cosine_distance,
        'avg_reconstruction_error': avg_reconstruction_error,
        'std_reconstruction_error': std_reconstruction_error,
        'cosine_distances_runs': cosine_distances_runs,
        'reconstruction_errors_runs': reconstruction_errors_runs
    }

def find_optimal_nmf_components(X, max_components=20, min_components=2, 
                              random_state=42, init='nndsvd', max_iter=1000,
                              standardize_method='peaks', hist_bins=50,
                              plot_results=True, knee_sensitivity=1.0,
                              n_runs=5, n_jobs=-1, verbose=1):
    """
    Find optimal number of NMF components using knee detection on average cosine distances.
    Uses multiple random initializations and parallel processing for robustness and speed.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data matrix
    max_components : int, default=20
        Maximum number of components to test
    min_components : int, default=2
        Minimum number of components to test
    random_state : int or None, default=42
        Base random seed for reproducibility. If None, uses random initialization.
    init : str, default='nndsvd'
        Initialization method for NMF
    max_iter : int, default=1000
        Maximum iterations for NMF
    plot_results : bool, default=True
        Whether to plot the results
    knee_sensitivity : float, default=1.0
        Sensitivity parameter for knee detection
    n_runs : int, default=5
        Number of random initializations to average over for each component count
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores
    verbose : int, default=1
        Verbosity level: 0=silent, 1=progress, 2=detailed
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'optimal_components': int, optimal number of components
        - 'components_range': list, range of components tested
        - 'avg_cosine_distances': list, average cosine distances for each component count
        - 'std_cosine_distances': list, standard deviations of cosine distances
        - 'reconstruction_errors': list, average reconstruction errors for each component count
        - 'std_reconstruction_errors': list, standard deviations of reconstruction errors
        - 'knee_point': int, detected knee point
        - 'detailed_results': list, detailed results for each component count
        - 'computation_time': float, total computation time in seconds
    """
    
    start_time = time.time()
    
    # Validate inputs
    if min_components < 1:
        raise ValueError("min_components must be at least 1")
    if max_components <= min_components:
        raise ValueError("max_components must be greater than min_components")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    
    # Ensure X is non-negative for NMF
    if np.any(X < 0):
        warnings.warn("Input matrix contains negative values. Taking absolute values.")
        X = np.abs(X)
    
    components_range = list(range(min_components, max_components + 1))
    
    if verbose >= 1:
        print(f"Testing NMF with {min_components} to {max_components} components...")
        print(f"Using {n_runs} random initializations per component count")
        print(f"Parallel processing with {n_jobs if n_jobs != -1 else 'all available'} cores")
    
    # Parallel processing across different component counts
    detailed_results = Parallel(n_jobs=n_jobs, verbose=max(0, verbose-1))(
        delayed(_fit_nmf_single_component)(
            X, n_components, n_runs, random_state, init, max_iter, standardize_method, hist_bins, verbose
        ) for n_components in components_range
    )
    
    # Extract aggregated results
    avg_cosine_distances = [result['avg_cosine_distance'] for result in detailed_results]
    std_cosine_distances = [result['std_cosine_distance'] for result in detailed_results]
    reconstruction_errors = [result['avg_reconstruction_error'] for result in detailed_results]
    std_reconstruction_errors = [result['std_reconstruction_error'] for result in detailed_results]
    
    # Find knee point using kneed library
    try:
        knee_locator = KneeLocator(components_range, reconstruction_errors, 
                                  curve='convex', direction='decreasing',
                                  S=knee_sensitivity)
        optimal_components = knee_locator.knee
        
        if optimal_components is None:
            # Fallback: find maximum second derivative
            if len(reconstruction_errors) >= 3:
                second_derivatives = np.diff(reconstruction_errors, 2)
                optimal_components = components_range[np.argmax(second_derivatives) + 1]
            else:
                optimal_components = min_components
                
    except Exception as e:
        if verbose >= 1:
            print(f"Knee detection failed: {e}")
        # Fallback to maximum second derivative
        if len(reconstruction_errors) >= 3:
            second_derivatives = np.diff(reconstruction_errors, 2)
            optimal_components = components_range[np.argmax(second_derivatives) + 1]
        else:
            optimal_components = min_components
    
    computation_time = time.time() - start_time
    
    # Create results dictionary
    results = {
        'optimal_components': optimal_components,
        'components_range': components_range,
        'avg_cosine_distances': avg_cosine_distances,
        'std_cosine_distances': std_cosine_distances,
        'reconstruction_errors': reconstruction_errors,
        'std_reconstruction_errors': std_reconstruction_errors,
        'knee_point': optimal_components,
        'detailed_results': detailed_results,
        'computation_time': computation_time,
        'n_runs': n_runs
    }
    
    # Plot results if requested
    if plot_results:
        plot_nmf_results(results)
    
    if verbose >= 1:
        print(f"\nOptimal number of components: {optimal_components}")
        print(f"Total computation time: {computation_time:.2f} seconds")
        if verbose >= 2:
            print(f"Average cosine distance at optimal: "
                  f"{avg_cosine_distances[components_range.index(optimal_components)]:.4f} ± "
                  f"{std_cosine_distances[components_range.index(optimal_components)]:.4f}")
    
    return results

def plot_nmf_results(results):
    """Plot the NMF analysis results with error bars from multiple initializations."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    components_range = results['components_range']
    avg_cosine_distances = results['avg_cosine_distances']
    std_cosine_distances = results['std_cosine_distances']
    reconstruction_errors = results['reconstruction_errors']
    std_reconstruction_errors = results['std_reconstruction_errors']
    optimal_components = results['optimal_components']
    n_runs = results['n_runs']
    
    # Plot 1: Average cosine distances with error bars
    ax1.errorbar(components_range, avg_cosine_distances, yerr=std_cosine_distances,
                 fmt='bo-', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax1.axvline(x=optimal_components, color='red', linestyle='--', 
                label=f'Optimal: {optimal_components} components')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Average Cosine Distance')
    ax1.set_title(f'Average Cosine Distance Between Components\n({n_runs} initializations per point)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Reconstruction errors with error bars
    ax2.errorbar(components_range, reconstruction_errors, yerr=std_reconstruction_errors,
                 fmt='go-', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax2.axvline(x=optimal_components, color='red', linestyle='--', 
                label=f'Optimal: {optimal_components} components')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title(f'NMF Reconstruction Error\n({n_runs} initializations per point)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_component_stability(results):
    """
    Analyze the stability of component selection across multiple initializations.
    
    Parameters:
    -----------
    results : dict
        Results from find_optimal_nmf_components
    
    Returns:
    --------
    dict : Stability analysis including coefficient of variation
    """
    stability_analysis = {}
    
    for result in results['detailed_results']:
        n_comp = result['n_components']
        cosine_distances = result['cosine_distances_runs']
        
        # Coefficient of variation (CV = std/mean)
        cv = result['std_cosine_distance'] / result['avg_cosine_distance'] if result['avg_cosine_distance'] > 0 else 0
        
        stability_analysis[n_comp] = {
            'coefficient_variation': cv,
            'min_cosine_distance': np.min(cosine_distances),
            'max_cosine_distance': np.max(cosine_distances),
            'range_cosine_distance': np.max(cosine_distances) - np.min(cosine_distances)
        }
    
    # Overall stability metrics
    all_cvs = [stability_analysis[n]['coefficient_variation'] for n in stability_analysis]
    
    return {
        'per_component': stability_analysis,
        'mean_coefficient_variation': np.mean(all_cvs),
        'max_coefficient_variation': np.max(all_cvs),
        'stability_score': 1 / (1 + np.mean(all_cvs))  # Higher = more stable
    }


def run_optimal_nmf(X, optimal_components, standardize_method='peaks', hist_bins=50, random_state=42, n_runs=1, **kwargs):
    """
    Run NMF with the optimal number of components, optionally with multiple initializations.
    
    Parameters:
    -----------
    X : array-like
        Input data matrix
    optimal_components : int
        Number of components to use
    standardize_method : str, default='peaks'
        Method for standardizing components ('peaks' or 'minmax')
    hist_bins : int, default=50
        Number of histogram bins for standardization
    random_state : int
        Base random seed
    n_runs : int, default=1
        Number of random initializations to try (returns best by reconstruction error)
    **kwargs : additional arguments for NMF
    
    Returns:
    --------
    dict : Dictionary containing W, H matrices and fitted NMF model
    """
    
    if n_runs == 1:
        # Single run case
        nmf = NMF(n_components=optimal_components, random_state=random_state, **kwargs)
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        W, H = standardize_methylation_nmf(X, H, W=W, hist_bins=hist_bins, method=standardize_method)
        
        return {
            'W': W,  # Sample loadings (n_samples × n_components)
            'H': H,  # Component patterns (n_components × n_features)
            'model': nmf,
            'reconstruction_error': nmf.reconstruction_err_
        }
    
    else:
        # Multiple runs - return best model by reconstruction error
        best_nmf = None
        best_error = float('inf')
        best_W = None
        best_H = None
        
        for run in range(n_runs):
            run_random_state = random_state + run if random_state is not None else None
            nmf = NMF(n_components=optimal_components, random_state=run_random_state, **kwargs)
            W = nmf.fit_transform(X)
            H = nmf.components_
            
            W, H = standardize_methylation_nmf(X, H, hist_bins=hist_bins, method=standardize_method)
            
            if nmf.reconstruction_err_ < best_error:
                best_error = nmf.reconstruction_err_
                best_nmf = nmf
                best_W = W
                best_H = H
        
        return {
            'W': best_W,
            'H': best_H,
            'model': best_nmf,
            'reconstruction_error': best_error,
            'n_runs_performed': n_runs
        }
    
def predict_H(X, W, standardize_method='peaks', hist_bins=50, n_jobs=-1, **kwargs):
    """
    Given fixed component patterns H, predict sample loadings W for new data X.
    
    Parameters:
    -----------
    X : array-like
        New input data matrix (n_samples × n_features)
    W : array-like
        Fixed sample loadings (n_samples × n_components)
    standardize_method : str, default='peaks'
        Method for standardizing components ('peaks' or 'minmax')
    hist_bins : int, default=50
        Number of histogram bins for standardization
    random_state : int
        Random seed for reproducibility
    **kwargs : additional arguments for NMF

    Returns:
    --------
    H : array-like
        Predicted component patterns (n_components × n_features)
    """

    from scipy.optimize import nnls

    n_features = X.shape[1]
    n_components = W.shape[1]

    def solve_column(i):
        h_col, _ = nnls(W, X[:, i])
        return h_col
    
    # Parallel computation
    H_columns = Parallel(n_jobs=n_jobs)(
        delayed(solve_column)(i) for i in range(n_features)
    )
    
    # Stack columns to form H
    H = np.column_stack(H_columns)

    # Standardize H to match original data distribution
    if standardize_method is not None:
        _, H_std = standardize_methylation_nmf(X, H, W=W, hist_bins=hist_bins, method=standardize_method)
    else:
        H_std = H

    return H_std  # Component patterns (n_components × n_features)


def impute_missing_iterative(X, W, missing_mask, max_iter=50, tol=1e-4):
    """
    Iterative imputation using alternating optimization
    """
    X_imputed = X.copy()
    
    # Initial imputation (mean imputation)
    for j in range(X.shape[1]):
        if np.any(missing_mask[:, j]):
            observed_mean = np.nanmean(X[:, j])
            X_imputed[missing_mask[:, j], j] = observed_mean
    
    prev_imputed_values = X_imputed[missing_mask].copy()
    
    for iteration in range(max_iter):
        # Step 1: Compute H given current X_imputed
        H = predict_H(X_imputed, W)
        
        # Step 2: Update missing values with reconstruction
        X_reconstructed = W @ H
        X_imputed[missing_mask] = X_reconstructed[missing_mask]
        
        # Check convergence
        current_imputed_values = X_imputed[missing_mask]
        change = np.linalg.norm(current_imputed_values - prev_imputed_values)
        
        if change < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
            
        prev_imputed_values = current_imputed_values.copy()
    
    return X_imputed, H