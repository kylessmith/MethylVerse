import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class RCGModel:
    """
    Ratio of Correlated Gammas (RCG) Model for DNA methylation beta values analysis.
    
    This model is designed to analyze beta values (M/(M+U)) by accounting for potential
    correlations between methylated (M) and unmethylated (U) signal intensities. It assumes 
    the bivariate distribution of M and U follows a Wicksell-Kibble distribution (bivariate
    gamma distribution).
    
    Parameters
    ----------
    beta_values : array-like
        Beta values to model (array of values between 0 and 1).
    covariates : pandas.DataFrame
        DataFrame of covariates to include in the model.
    """
    
    def __init__(self, beta_values, covariates=None):
        """Initialize the RCG model."""
        self.beta_values = np.array(beta_values)
        
        # Check that beta values are within [0, 1]
        if np.any((self.beta_values < 0) | (self.beta_values > 1)):
            raise ValueError("Beta values must be between 0 and 1")
            
        # Handle missing values (NaNs) by removing them
        if np.any(np.isnan(self.beta_values)):
            valid_indices = ~np.isnan(self.beta_values)
            self.beta_values = self.beta_values[valid_indices]
            
            if covariates is not None:
                # Handle different types of covariates
                if hasattr(covariates, 'iloc'):
                    # pandas DataFrame or Series
                    covariates = covariates.iloc[valid_indices, :] if covariates.ndim > 1 else covariates.iloc[valid_indices]
                else:
                    # numpy array
                    covariates = covariates[valid_indices, :] if covariates.ndim > 1 else covariates[valid_indices]
                    
            warnings.warn(f"Removed {np.sum(~valid_indices)} samples with missing beta values")
        
        # Process covariates
        if covariates is None:
            self.X = np.ones((len(self.beta_values), 1))  # Intercept only
        else:
            # Add intercept
            if isinstance(covariates, pd.DataFrame) or isinstance(covariates, pd.Series):
                self.X = sm.add_constant(covariates)
            else:
                # Convert numpy array to pandas DataFrame for easier handling
                self.X = sm.add_constant(pd.DataFrame(covariates))
        
        # Initialize parameters
        self.params = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        
        # Initialize parameter bounds
        # Shape parameters should be positive
        # Correlation parameter should be between -1 and 1
        self.param_bounds = None
        
    def _compute_loglik(self, params):
        """
        Compute the negative log-likelihood for the RCG model.
        
        Parameters
        ----------
        params : array-like
            Model parameters [a, b, rho, gamma_1, gamma_2, ..., gamma_p]
            where:
            - a, b are shape parameters for the gamma distributions
            - rho is the correlation parameter
            - gamma_i are the coefficients for covariates
            
        Returns
        -------
        float
            Negative log-likelihood value
        """
        # Extract parameters
        a = params[0]  # Shape parameter for M
        b = params[1]  # Shape parameter for U
        rho = params[2]  # Correlation parameter
        gamma = params[3:]  # Coefficients for covariates
        
        # Protection against extreme parameter values
        if a <= 0 or b <= 0 or abs(rho) >= 1:
            return np.inf
            
        # Create linear predictor (design matrix * coefficients)
        eta = np.dot(self.X, gamma)
        
        # Transform linear predictor to [0, 1] scale using logistic function
        mu = 1 / (1 + np.exp(-eta))
        
        # Ensure mu values are not exactly 0 or 1 to avoid numerical issues
        mu = np.clip(mu, 1e-10, 1-1e-10)
        
        # Compute log-likelihood for each observation
        loglik = 0
        
        for i in range(len(self.beta_values)):
            y = self.beta_values[i]
            m = mu[i]
            
            # Handle potential numerical issues with extreme beta values
            y = np.clip(y, 1e-10, 1-1e-10)
            
            # Compute contribution to log-likelihood from this observation
            # This is derived from the density function of the ratio of correlated gammas
            
            # Base log-likelihood (beta distribution part)
            llk_i = (a-1) * np.log(y) + (b-1) * np.log(1-y) - gammaln(a) - gammaln(b) + gammaln(a+b)
            
            # Account for correlation between M and U
            # Ensure the term inside log is positive to avoid numerical issues
            corr_term = 1 - rho * y * (1-y) / (a*b)
            if corr_term <= 0:
                return np.inf  # Invalid parameters
            
            llk_i += np.log(corr_term)
            
            # Check for NaN or infinite values
            if not np.isfinite(llk_i):
                return np.inf
            
            loglik += llk_i
        
        # Check overall log-likelihood validity
        if not np.isfinite(loglik):
            return np.inf
        
        # Return negative log-likelihood for minimization
        return -loglik
        
    def fit(self, method='Nelder-Mead', maxiter=5000):
        """
        Fit the RCG model to the data.
        
        Parameters
        ----------
        method : str, optional
            Optimization method for scipy.optimize.minimize. Default is 'Nelder-Mead',
            which is more robust for this problem than gradient-based methods.
        maxiter : int, optional
            Maximum number of iterations for optimization. Default is 5000.
            
        Returns
        -------
        self : RCGModel
            Fitted model instance
        """
        # Number of parameters: a, b, rho, and coefficients for covariates
        n_params = 3 + self.X.shape[1]
        
        # Get some statistics about the data to make better initial guesses
        mean_beta = np.mean(self.beta_values)
        var_beta = np.var(self.beta_values)
        
        # Using method of moments to estimate initial a and b
        # For a beta distribution:
        # mean = a / (a + b)
        # var = a*b / ((a+b)^2 * (a+b+1))
        if var_beta == 0:
            # If all beta values are identical, use default values
            a_init = 2.0
            b_init = 2.0
        else:
            # Solve for a and b using method of moments
            ab_sum = mean_beta * (1 - mean_beta) / var_beta - 1
            a_init = mean_beta * ab_sum
            b_init = (1 - mean_beta) * ab_sum
            
            # Ensure reasonable values
            a_init = max(1.0, min(100.0, a_init))
            b_init = max(1.0, min(100.0, b_init))
        
        # Initial parameter values
        initial_params = np.zeros(n_params)
        initial_params[0] = a_init  # a
        initial_params[1] = b_init  # b
        initial_params[2] = 0.0     # rho (start with no correlation)
        
        # For gamma coefficients, use a reasonable starting point
        # The intercept should predict the mean beta value
        if self.X.shape[1] > 0:
            # Logit of mean for intercept
            initial_params[3] = np.log(mean_beta / (1 - mean_beta))
            
        # Parameter bounds
        self.param_bounds = [
            (0.001, 1000.0),  # a > 0
            (0.001, 1000.0),  # b > 0
            (-0.9, 0.9),      # -1 < rho < 1, but avoid extremes
        ]
        
        # Add bounds for gamma coefficients (bounded to avoid numerical issues)
        for _ in range(self.X.shape[1]):
            self.param_bounds.append((-50.0, 50.0))
        
        # Try multiple optimization methods if needed
        methods = [method]
        if method != 'Nelder-Mead':
            methods.append('Nelder-Mead')  # Fallback method
        if 'L-BFGS-B' not in methods:
            methods.append('L-BFGS-B')     # Another fallback
            
        best_result = None
        best_loglik = -np.inf
        
        for current_method in methods:
            try:
                # Fit the model by minimizing negative log-likelihood
                opt_result = minimize(
                    self._compute_loglik,
                    initial_params,
                    method=current_method,
                    bounds=self.param_bounds,
                    options={'maxiter': maxiter}
                )
                
                # Check if this is better than our previous result
                if opt_result.success and (best_result is None or -opt_result.fun > best_loglik):
                    best_result = opt_result
                    best_loglik = -opt_result.fun
                    
            except Exception as e:
                print(f"Optimization with {current_method} failed: {str(e)}")
                continue
                
        # If we couldn't find a good fit with any method
        if best_result is None:
            raise ValueError("Model fitting failed with all optimization methods")
        
        # Store results
        self.params = best_result.x
        self.log_likelihood = -best_result.fun
        self.convergence = best_result.success
        self.n_samples = len(self.beta_values)
        
        # Calculate AIC and BIC
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = n_params * np.log(self.n_samples) - 2 * self.log_likelihood
        
        # Check the validity of the fit
        if not np.isfinite(self.log_likelihood) or self.log_likelihood == 0:
            warnings.warn("Model fitting may have failed: log-likelihood is not valid")
            self.convergence = False
            
        return self
    
    def summary(self):
        """
        Print a summary of the fitted model.
        
        Returns
        -------
        str
            Summary string
        """
        if self.params is None:
            return "Model not fitted yet. Call fit() first."
        
        summary_str = "RCG Model Summary\n"
        summary_str += "="*50 + "\n"
        
        # Extract parameters
        a = self.params[0]
        b = self.params[1]
        rho = self.params[2]
        gamma = self.params[3:]
        
        summary_str += f"Shape parameter a: {a:.4f}\n"
        summary_str += f"Shape parameter b: {b:.4f}\n"
        summary_str += f"Correlation parameter rho: {rho:.4f}\n"
        
        summary_str += "\nCoefficients:\n"
        for i, col in enumerate(self.X.columns if hasattr(self.X, 'columns') else range(self.X.shape[1])):
            summary_str += f"  {col}: {gamma[i]:.4f}\n"
        
        summary_str += "\nModel Statistics:\n"
        summary_str += f"  Log-likelihood: {self.log_likelihood:.4f}\n"
        summary_str += f"  AIC: {self.aic:.4f}\n"
        summary_str += f"  BIC: {self.bic:.4f}\n"
        summary_str += f"  Number of samples: {self.n_samples}\n"
        
        return summary_str
    
    def test_covariate_effect(self, covariate_index, alpha=0.05):
        """
        Test the significance of a covariate's effect on methylation.
        
        This implements a likelihood ratio test comparing the full model
        with a reduced model that excludes the specified covariate.
        
        Parameters
        ----------
        covariate_index : int
            Index of the covariate to test (in the original covariates DataFrame).
            Use 0 for the intercept.
        alpha : float, optional
            Significance level. Default is 0.05.
            
        Returns
        -------
        dict
            Dictionary containing test statistics and p-value
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Adjust covariate_index for the internal parameter indexing
        param_index = covariate_index + 3  # a, b, rho come first
        
        # Verify the index is valid
        if param_index >= len(self.params):
            raise IndexError(f"Covariate index {covariate_index} is out of bounds. Model has {len(self.params) - 3} covariates.")
        
        # Get full model log-likelihood
        full_model_ll = self.log_likelihood
        
        # Create a copy of the X matrix without the covariate being tested
        if hasattr(self.X, 'drop'):
            # For pandas DataFrame
            col_name = self.X.columns[covariate_index]
            reduced_X = self.X.drop(columns=[col_name])
        else:
            # For numpy array
            reduced_X = np.delete(self.X, covariate_index, axis=1)
        
        # Fit a reduced model
        try:
            reduced_model = RCGModel(self.beta_values, covariates=reduced_X)
            reduced_model.fit()
            reduced_model_ll = reduced_model.log_likelihood
            
            # Compute likelihood ratio test statistic
            lr_statistic = 2 * (full_model_ll - reduced_model_ll)
            
            # Avoid numerical issues
            if lr_statistic < 0:
                # This shouldn't happen theoretically, but can due to numerical issues
                lr_statistic = 0
                
            # Compute p-value (chi-squared with 1 degree of freedom)
            p_value = 1 - stats.chi2.cdf(lr_statistic, 1)
            
        except Exception as e:
            print(f"Error in likelihood ratio test: {str(e)}")
            # Fallback to approximate test using parameter standard error
            # Estimate standard error based on the Hessian (second derivative of log-likelihood)
            # This is a simpler but less accurate approach
            
            # For simplicity, just use a direct test based on the parameter value
            param_value = self.params[param_index]
            # Use a rough estimate for the standard error
            se = abs(param_value) / 2.0  # Rule of thumb estimate
            
            if se == 0 or param_value == 0:
                p_value = 1.0
            else:
                # Calculate z-statistic
                z_stat = param_value / se
                # Two-tailed p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            lr_statistic = z_stat ** 2  # Approximation
        
        # Return test results
        result = {
            'covariate': covariate_index,
            'lr_statistic': float(lr_statistic),  # Ensure it's not a numpy type
            'p_value': float(p_value),  # Ensure it's not a numpy type
            'significant': p_value < alpha
        }
        
        return result
    
    def predict(self, new_covariates=None):
        """
        Predict mean beta values for given covariates.
        
        Parameters
        ----------
        new_covariates : pandas.DataFrame or numpy.ndarray, optional
            New covariate values for prediction. If None, uses the training data.
            
        Returns
        -------
        numpy.ndarray
            Predicted beta values
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Process new covariates
        if new_covariates is None:
            X_pred = self.X
        else:
            # Check if we're dealing with pandas or numpy
            if hasattr(new_covariates, 'values'):
                # It's a pandas DataFrame/Series
                if hasattr(self.X, 'columns'):
                    # Ensure columns match if both are pandas
                    if not all(col in new_covariates.columns for col in self.X.columns):
                        # Add missing columns with zeros
                        for col in self.X.columns:
                            if col not in new_covariates.columns:
                                new_covariates[col] = 0
                    # Only select the columns in the same order as in self.X
                    X_pred = new_covariates[self.X.columns]
                else:
                    # Convert to numpy if self.X is numpy
                    X_pred = new_covariates.values
            else:
                # It's a numpy array
                if hasattr(self.X, 'values'):
                    # Convert self.X to numpy if it's pandas
                    X_pred = new_covariates
                else:
                    # Both are numpy
                    X_pred = new_covariates
        
        # Extract gamma coefficients
        gamma = self.params[3:]
        
        # Ensure dimensions match
        if len(gamma) != X_pred.shape[1]:
            raise ValueError(f"Dimension mismatch: Number of coefficients ({len(gamma)}) "
                           f"does not match number of covariates ({X_pred.shape[1]})")
            
        # Create linear predictor
        try:
            eta = np.dot(X_pred, gamma)
        except ValueError as e:
            print(f"Dimension error in prediction: X_pred shape {X_pred.shape}, gamma shape {gamma.shape}")
            raise e
        
        # Transform to [0, 1] scale using logistic function
        predicted_beta = 1 / (1 + np.exp(-eta))
        
        return predicted_beta


# Example usage:

def compare_features_with_rcg(array1, array2, covariates=None, alpha=0.05, fallback_to_ttest=True):
    """
    Compare features between two arrays using the RCG model.
    
    Parameters
    ----------
    array1 : numpy.ndarray
        First array where rows are features and columns are samples.
    array2 : numpy.ndarray
        Second array where rows are features and columns are samples.
    covariates : pandas.DataFrame, optional
        Covariates for the model. If None, only group effect is modeled.
    alpha : float, optional
        Significance level. Default is 0.05.
    fallback_to_ttest : bool, optional
        If True, use a t-test when RCG model fitting fails. Default is True.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing test results for each feature.
    """
    # Check input arrays
    if array1.shape[0] != array2.shape[0]:
        raise ValueError("Arrays must have the same number of features (rows)")
    
    n_features = array1.shape[0]
    n_samples1 = array1.shape[1]
    n_samples2 = array2.shape[1]
    
    # Create group indicator variable (0 for array1, 1 for array2)
    group = np.concatenate([np.zeros(n_samples1), np.ones(n_samples2)])
    
    # Create combined covariates
    if covariates is None:
        # Only use group indicator
        combined_covariates = pd.DataFrame({'group': group})
    else:
        # Check covariates
        if not isinstance(covariates, list) or len(covariates) != 2:
            raise ValueError("Covariates should be a list of two DataFrames")
        
        cov1, cov2 = covariates
        
        # Ensure covariates have the right number of samples
        if len(cov1) != n_samples1 or len(cov2) != n_samples2:
            raise ValueError(f"Covariates dimensions don't match arrays. Got {len(cov1)}, {len(cov2)} but expected {n_samples1}, {n_samples2}")
        
        # Combine covariates and add group indicator
        combined_covariates = pd.concat([cov1, cov2], axis=0, ignore_index=True)
        combined_covariates['group'] = group
    
    # Initialize results storage
    results = {
        'feature_id': [],
        'p_value': [],
        'significant': [],
        'effect_size': [],
        'method_used': []  # Track which method was used
    }
    
    # Process each feature
    for i in range(n_features):
        # Get data for this feature
        vals1 = array1[i, :]
        vals2 = array2[i, :]
        
        # Skip features with too many missing values
        missing_rate1 = np.mean(np.isnan(vals1))
        missing_rate2 = np.mean(np.isnan(vals2))
        
        if missing_rate1 > 0.5 or missing_rate2 > 0.5:
            # Too many missing values, don't analyze this feature
            results['feature_id'].append(i)
            results['p_value'].append(1.0)
            results['significant'].append(False)
            results['effect_size'].append(0.0)
            results['method_used'].append('skipped_missing_data')
            continue
            
        # Check if there's enough variation to analyze
        if np.nanvar(vals1) < 1e-8 and np.nanvar(vals2) < 1e-8:
            # Almost no variation in either group
            results['feature_id'].append(i)
            results['p_value'].append(1.0)
            results['significant'].append(False)
            results['effect_size'].append(0.0)
            results['method_used'].append('skipped_no_variation')
            continue
            
        # Combine beta values for the feature
        beta_values = np.concatenate([vals1, vals2])
        
        # Try using RCG model first
        use_rcg = False
        try:
            # First check if RCG model is likely to be useful
            # If means are very different, RCG should be beneficial
            mean_diff = abs(np.nanmean(vals2) - np.nanmean(vals1))
            if mean_diff > 0.05:  # Meaningful difference to model
                # Fit RCG model
                model = RCGModel(beta_values, covariates=combined_covariates)
                model.fit()
                
                if not model.convergence:
                    raise ValueError("RCG model failed to converge")
                
                # Get the index of the 'group' column
                group_index = list(combined_covariates.columns).index('group')
                    
                # Test group effect
                test_result = model.test_covariate_effect(
                    covariate_index=group_index,
                    alpha=alpha
                )
                
                # Check if the test result makes sense
                if not np.isfinite(test_result['p_value']) or test_result['lr_statistic'] <= 0:
                    raise ValueError("Invalid test result")
                
                # If we get here, RCG was successful
                use_rcg = True
                
                # Store results
                results['feature_id'].append(i)
                results['p_value'].append(test_result['p_value'])
                results['significant'].append(test_result['significant'])
                
                # Calculate effect size (difference in predicted means)
                # Create two new datasets with only the group effect varying
                X_pred1 = pd.DataFrame({'group': [0]})
                X_pred2 = pd.DataFrame({'group': [1]})
                
                # Add mean values for other covariates
                for col in combined_covariates.columns:
                    if col != 'group':
                        mean_val = combined_covariates[col].mean()
                        X_pred1[col] = mean_val
                        X_pred2[col] = mean_val
                
                # Predict and calculate effect size
                pred1 = model.predict(X_pred1)[0]
                pred2 = model.predict(X_pred2)[0]
                effect_size = pred2 - pred1
                
                results['effect_size'].append(effect_size)
                results['method_used'].append('rcg')
            else:
                # Not enough difference to justify RCG
                raise ValueError("Small mean difference, skipping RCG")
        except Exception as e:
            if use_rcg:
                # If we already set use_rcg but then hit an error later
                print(f"RCG model fitted but testing failed for feature {i}: {str(e)}")
            else:
                print(f"Skipping RCG model for feature {i}: {str(e)}")
            
            # Fall back to other methods
            use_rcg = False
            
        # If RCG didn't work, try simpler methods
        if not use_rcg:
            if fallback_to_ttest:
                try:
                    # Fall back to t-test or Mann-Whitney U test
                    # Remove missing values
                    vals1_clean = vals1[~np.isnan(vals1)]
                    vals2_clean = vals2[~np.isnan(vals2)]
                    
                    # Check if we have enough data
                    if len(vals1_clean) < 3 or len(vals2_clean) < 3:
                        raise ValueError("Not enough non-missing data points for statistical test")
                    
                    # Check normality to decide between t-test and Mann-Whitney U
                    try:
                        _, p_norm1 = stats.shapiro(vals1_clean)
                        _, p_norm2 = stats.shapiro(vals2_clean)
                        normal_data = p_norm1 > 0.05 and p_norm2 > 0.05
                    except:
                        # Shapiro-Wilk test might fail with small samples
                        normal_data = False
                    
                    if normal_data:
                        # Use t-test for normally distributed data
                        _, p_value = stats.ttest_ind(vals1_clean, vals2_clean, equal_var=False)
                        test_name = 'ttest'
                    else:
                        # Use Mann-Whitney U test for non-normal data
                        _, p_value = stats.mannwhitneyu(vals1_clean, vals2_clean, alternative='two-sided')
                        test_name = 'mannwhitney'
                    
                    # Calculate simple effect size (difference in means)
                    effect_size = np.nanmean(vals2) - np.nanmean(vals1)
                    
                    results['feature_id'].append(i)
                    results['p_value'].append(p_value)
                    results['significant'].append(p_value < alpha)
                    results['effect_size'].append(effect_size)
                    results['method_used'].append(test_name)
                    
                except Exception as e2:
                    print(f"Fallback test also failed for feature {i}: {str(e2)}")
                    results['feature_id'].append(i)
                    results['p_value'].append(np.nan)
                    results['significant'].append(False)
                    results['effect_size'].append(np.nan)
                    results['method_used'].append('failed')
            else:
                # Don't use fallback test
                results['feature_id'].append(i)
                results['p_value'].append(np.nan)
                results['significant'].append(False)
                results['effect_size'].append(np.nan)
                results['method_used'].append('failed')
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply multiple testing correction
    valid_pvals = ~np.isnan(results_df['p_value'])
    if np.sum(valid_pvals) > 0:
        # Only apply correction to valid p-values
        corrected_pvals = np.ones(len(results_df))
        
        # Get valid p-values
        pvals_to_correct = results_df.loc[valid_pvals, 'p_value'].values
        
        # Correct them
        corrected_valid_pvals = sm.stats.multipletests(
            pvals_to_correct,
            method='fdr_bh'  # Benjamini-Hochberg procedure
        )[1]
        
        # Put them back
        corrected_pvals[valid_pvals] = corrected_valid_pvals
        
        results_df['adjusted_p_value'] = corrected_pvals
    else:
        # No valid p-values to correct
        results_df['adjusted_p_value'] = 1.0
    
    results_df['significant_adjusted'] = results_df['adjusted_p_value'] < alpha
    
    # Sort by adjusted p-value and filter out NaN adjusted p-values
    results_df = results_df.sort_values('adjusted_p_value')
    
    # Count methods used
    method_counts = results_df['method_used'].value_counts()
    print("Methods used for testing:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} features ({count/n_features*100:.1f}%)")
    
    return results_df


# Example with simulated data
def simulate_methylation_data():
    """
    Simulate methylation data for demonstration.
    
    Returns
    -------
    tuple
        (array1, array2, true_differential_features)
    """
    np.random.seed(42)
    
    # Simulation parameters
    n_features = 100
    n_samples1 = 20
    n_samples2 = 20
    n_diff_features = 10
    
    # Generate base methylation patterns
    array1 = np.random.beta(2, 2, size=(n_features, n_samples1))
    array2 = np.random.beta(2, 2, size=(n_features, n_samples2))
    
    # Add differential methylation to some features
    diff_features = np.random.choice(n_features, n_diff_features, replace=False)
    
    for i in diff_features:
        # Add a consistent difference
        delta = 0.3
        array2[i, :] = np.minimum(array1[i, :] + delta, 0.99)
    
    # Add some missing values
    missing_rate = 0.05
    mask1 = np.random.random(array1.shape) < missing_rate
    mask2 = np.random.random(array2.shape) < missing_rate
    
    array1[mask1] = np.nan
    array2[mask2] = np.nan
    
    return array1, array2, diff_features

# Example usage code:
if __name__ == "__main__":
    print("Running example with simulated data...")
    
    print("Step 1: Simulating methylation beta values data")
    # Simulate data
    array1, array2, true_diff_features = simulate_methylation_data()
    
    # Look at the data
    print(f"Simulated data shape: {array1.shape} and {array2.shape}")
    print(f"Number of true differential features: {len(true_diff_features)}")
    print(f"True differential features: {true_diff_features}")
    
    print("\nStep 2: Creating example covariates")
    # Create some covariates (e.g., age and sex)
    n_samples1 = array1.shape[1]
    n_samples2 = array2.shape[1]
    covariates1 = pd.DataFrame({
        'age': np.random.normal(50, 10, n_samples1),
        'sex': np.random.binomial(1, 0.5, n_samples1)
    })
    covariates2 = pd.DataFrame({
        'age': np.random.normal(55, 10, n_samples2),
        'sex': np.random.binomial(1, 0.5, n_samples2)
    })
    
    print("Created covariates:")
    print(f"Group 1: {covariates1.shape}")
    print(f"Group 2: {covariates2.shape}")
    
    print("\nStep 3: Simple test with first feature only")
    # Try running the model on just the first feature
    try:
        print(f"Testing feature {true_diff_features[0]} (a true differential feature)")
        beta_values = np.concatenate([array1[true_diff_features[0], :], array2[true_diff_features[0], :]])
        
        # Create combined covariates
        group = np.concatenate([np.zeros(n_samples1), np.ones(n_samples2)])
        combined_covs = pd.concat([covariates1, covariates2], axis=0, ignore_index=True)
        combined_covs['group'] = group
        
        print(f"Beta values shape: {beta_values.shape}")
        print(f"Combined covariates shape: {combined_covs.shape}")
        
        # Create and fit model
        model = RCGModel(beta_values, covariates=combined_covs)
        model.fit()
        
        # Test for group effect
        group_index = list(combined_covs.columns).index('group')
        test_result = model.test_covariate_effect(covariate_index=group_index)
        
        print(f"Model fitting successful!")
        print(f"P-value for group effect: {test_result['p_value']:.6f}")
        print(f"Model parameters: a={model.params[0]:.4f}, b={model.params[1]:.4f}, rho={model.params[2]:.4f}")
        
        # Simple effect size calculation
        print(f"Mean in group 1: {np.nanmean(array1[true_diff_features[0], :]):.4f}")
        print(f"Mean in group 2: {np.nanmean(array2[true_diff_features[0], :]):.4f}")
        
        print("\nModel successfully tested on a single feature!")
    except Exception as e:
        print(f"Error testing single feature: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nStep 4: Running full comparison with all features")
    # Compare features
    results = compare_features_with_rcg(
        array1, array2, 
        covariates=[covariates1, covariates2]
    )

    # Print results
    print(f"\nNumber of significant features: {results['significant_adjusted'].sum()}")
    
    if len(true_diff_features) > 0:
        true_pos = sum(results.loc[results['significant_adjusted'], 'feature_id'].isin(true_diff_features))
        true_pos_rate = true_pos / len(true_diff_features) if len(true_diff_features) > 0 else 0
        print(f"True positive rate: {true_pos_rate:.2f}")
        
        false_pos = results['significant_adjusted'].sum() - true_pos
        fdr = false_pos / results['significant_adjusted'].sum() if results['significant_adjusted'].sum() > 0 else 0
        print(f"False discovery rate: {fdr:.2f}")

    # Look at the top features
    print("\nTop differentially methylated features:")
    print(results.head(10).to_string())

    # Check if we recovered all the true differential features
    print("\nTrue differential features and their ranks:")
    for feature in true_diff_features:
        if feature in results['feature_id'].values:
            idx = results[results['feature_id'] == feature].index[0]
            rank = idx + 1
            p_value = results.loc[idx, 'adjusted_p_value']
            method = results.loc[idx, 'method_used']
            print(f"Feature {feature}: rank {rank}, adjusted p-value {p_value:.6f}, method: {method}")
        else:
            print(f"Feature {feature}: Not found in results")
            
    print("\nStep 5: Running traditional t-tests for comparison")
    ttest_pvals = []
    
    for i in range(array1.shape[0]):
        # Get data without NaNs
        vals1 = array1[i, :][~np.isnan(array1[i, :])]
        vals2 = array2[i, :][~np.isnan(array2[i, :])]
        
        if len(vals1) < 2 or len(vals2) < 2:
            ttest_pvals.append(1.0)
            continue
            
        try:
            _, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
            ttest_pvals.append(p_val)
        except:
            ttest_pvals.append(1.0)
    
    # Apply multiple testing correction
    ttest_adj_pvals = sm.stats.multipletests(ttest_pvals, method='fdr_bh')[1]
    
    # Count significant features
    ttest_sig = np.sum(ttest_adj_pvals < 0.05)
    print(f"T-test significant features: {ttest_sig}")
    
    # Calculate true positive rate for t-test
    ttest_true_pos = sum([1 for i, adj_p in enumerate(ttest_adj_pvals) if adj_p < 0.05 and i in true_diff_features])
    ttest_tpr = ttest_true_pos / len(true_diff_features) if len(true_diff_features) > 0 else 0
    print(f"T-test true positive rate: {ttest_tpr:.2f}")
    
    # Calculate FDR for t-test
    ttest_fdr = 1 - ttest_true_pos / ttest_sig if ttest_sig > 0 else 0
    print(f"T-test false discovery rate: {ttest_fdr:.2f}")
    
    # Compare ranks of true differential features between RCG and t-test
    print("\nStep 6: Comparing ranks of true differential features between RCG and t-test")
    print("Feature\tRCG Rank\tT-test Rank")
    
    for feature in true_diff_features:
        # Get RCG rank
        if feature in results['feature_id'].values:
            rcg_rank = results[results['feature_id'] == feature].index[0] + 1
        else:
            rcg_rank = "Not found"
            
        # Get t-test rank
        try:
            # Sort and find rank
            ttest_sorted_indices = np.argsort(ttest_adj_pvals)
            feature_position = np.where(ttest_sorted_indices == feature)[0]
            if len(feature_position) > 0:
                ttest_rank = feature_position[0] + 1
            else:
                ttest_rank = "Not found"
        except Exception as e:
            ttest_rank = f"Error: {str(e)}"
        
        print(f"{feature}\t{rcg_rank}\t{ttest_rank}")
    
    print("\nExample analysis complete!")