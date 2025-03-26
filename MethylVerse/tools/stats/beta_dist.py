import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta, norm


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
    # The log likelihood for a Beta distribution is:
    #   sum[(a-1)*log(x_i) + (b-1)*log(1-x_i)] - n * betaln(a, b)
    ll = np.sum((a - 1) * np.log(x) + (b - 1) * np.log(1 - x)) - n * betaln(a, b)
    return -ll  # return negative log-likelihood for minimization

def estimate_beta_params(x, init_params=(2.0, 2.0)):
    """
    Estimate Beta distribution parameters from data using maximum likelihood.
    x: array of methylation values (must be in (0,1); values are clipped to avoid 0,1 issues).
    init_params: initial guess for (a, b).
    """
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
    try:
        # Estimate Beta parameters (MLE) for this region
        a_mle, b_mle = estimate_beta_params(x)
    except RuntimeError:
        print("Parameter estimation failed.")
        return None, None
    # Compute the estimated mean methylation level for the region
    mle_mean = a_mle / (a_mle + b_mle)
    
    # Compute log likelihoods under the estimated model and under the fixed high methylation model
    ll_mle = log_likelihood_beta(x, a_mle, b_mle)
    ll_high = log_likelihood_beta(x, alpha, beta)
    
    # Log-likelihood ratio (a positive value indicates the estimated model fits better than the high methylation model)
    llr = ll_mle - ll_high
    
    return mle_mean, llr
    
def beta_zscore(x_obs, alpha=2, beta_param=1):
    """
    Compute a standardized score for observed methylation values under a Beta distribution.
    x_obs: observed methylation value (assumed to be in (0,1)).
    alpha, beta_param: parameters of the Beta distribution.
    """
    # Compute mean and standard deviation of the Beta distribution
    mean_beta = alpha / (alpha + beta_param)
    std_beta = np.sqrt((alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1)))

    # Compute "Beta z-score" (standardized beta score)
    z_beta = (x_obs - mean_beta) / std_beta

    # Compute percentile-based score (p-value equivalent)
    p_value = beta.cdf(x_obs, alpha, beta_param)

    # Convert percentile to standard normal z-score equivalent
    z_p = norm.ppf(p_value)
    
    return z_beta, z_p, p_value


def beta_tscore(x, y):
    """
    """

    # Fit Beta distribution to data
    x_alpha, x_beta = estimate_beta_params(x)
    y_alpha, y_beta = estimate_beta_params(y)

    # Compute mean and standard deviation of the Beta distribution
    x_mean = beta_mean(x_alpha, x_beta)
    x_std = beta_std(x_alpha, x_beta)
    y_mean = beta_mean(y_alpha, y_beta)
    y_std = beta_std(y_alpha, y_beta)

    # Compute "Beta z-score" (standardized beta score)
    t_beta = (x_mean - y_mean) / (x_std / np.sqrt(len(x)))

    return t_beta
    