import numpy as np
import scipy.stats as stats
import scipy.special as sp
import scipy.optimize as opt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def beta_binomial_methylation(beta_values, sample_labels, weights=None):
    """
    Perform beta-binomial regression to compare methylation values between two samples using statsmodels.

    Parameters:
    - methylated_counts: List or array of methylated read counts.
    - total_counts: List or array of total read counts.
    - sample_labels: Binary labels (0 or 1) indicating sample groups.

    Returns:
    - Summary of regression results.
    """

    # Convert to pandas DataFrame
    data = pd.DataFrame({
        "proportion": beta_values,
        "sample": sample_labels
    })

    # Compute proportion of methylation
    data["proportion"] = data["methylated"] / data["total"]

    # Fit a binomial logistic regression (as an approximation to beta-binomial)
    if weights is None:
        model = smf.glm("proportion ~ sample",
                        data=data,
                        family=sm.families.Binomial()).fit()
    else:
        model = smf.glm("proportion ~ sample",
                        data=data,
                        family=sm.families.Binomial(),
                        freq_weights=weights).fit()

    return model


def beta_binomial_log_likelihood(params, methylated_counts, total_counts, sample_labels):
    """
    Compute the log-likelihood of the Beta-Binomial regression model.

    Parameters:
    - params: Model parameters (alpha, beta, intercept, beta_coef)
    - methylated_counts: Array of methylated read counts
    - total_counts: Array of total read counts
    - sample_labels: Binary group labels (0 = control, 1 = treatment)

    Returns:
    - Negative log-likelihood (to minimize in optimization)
    """
    alpha, beta, intercept, beta_coef = params

    # Compute probability (logit transformation)
    logit_p = intercept + beta_coef * sample_labels
    p = 1 / (1 + np.exp(-logit_p))  # Sigmoid function

    # Compute the Beta-Binomial log-likelihood
    log_likelihood = (
        sp.betaln(methylated_counts + alpha * p, total_counts - methylated_counts + beta * (1 - p))
        - sp.betaln(alpha * p, beta * (1 - p))
        + (sp.gammaln(total_counts + 1) - sp.gammaln(methylated_counts + 1) - sp.gammaln(total_counts - methylated_counts + 1))
    )

    return -np.sum(log_likelihood)  # Negative log-likelihood for minimization


def beta_binomial_methylation(methylated_counts, total_counts, sample_labels):
    """
    Perform Beta-Binomial regression using Maximum Likelihood Estimation (MLE) and compute p-values.

    Parameters:
    - methylated_counts: List or array of methylated read counts
    - total_counts: List or array of total read counts
    - sample_labels: Binary labels (0 = control, 1 = treatment)

    Returns:
    - Optimized parameter estimates (alpha, beta, intercept, beta_coef)
    - p-value for beta_coef (statistical significance test)
    """
    # Convert to numpy arrays
    methylated_counts = np.array(methylated_counts)
    total_counts = np.array(total_counts)
    sample_labels = np.array(sample_labels)

    # Initial guesses for parameters (alpha, beta, intercept, beta_coef)
    init_params = [1.0, 1.0, 0.0, 0.0]

    # Optimize using MLE
    result = opt.minimize(
        beta_binomial_log_likelihood, 
        init_params, 
        args=(methylated_counts, total_counts, sample_labels),
        method="L-BFGS-B",
        bounds=[(1e-5, None), (1e-5, None), (None, None), (None, None)],  # Alpha & Beta must be > 0
        options={'disp': False}
    )

    # Extract estimated parameters
    alpha, beta, intercept, beta_coef = result.x

    # Compute standard errors from inverse Hessian matrix (approximate variance)
    if result.hess_inv is not None:
        variance_cov_matrix = result.hess_inv.todense()  # Convert Hessian to dense matrix
        standard_errors = np.sqrt(np.diag(variance_cov_matrix))  # Get standard errors
    else:
        standard_errors = [np.nan] * 4  # Fallback if Hessian is unavailable

    # Compute z-score and p-value for beta_coef
    beta_se = standard_errors[3]  # Standard error of beta_coef
    if not np.isnan(beta_se):
        z_score = beta_coef / beta_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed p-value
    else:
        p_value = np.nan  # If SE is not computable, return NaN

    return {
        "alpha": alpha,
        "beta": beta,
        "intercept": intercept,
        "beta_coef": beta_coef,
        "beta_se": beta_se,
        "z_score": z_score,
        "p_value": p_value,
        "log_likelihood": -result.fun  # Log-likelihood of the model
    }