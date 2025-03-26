import pandas as pd
import numpy as np
from sklearn.svm import NuSVR
from sklearn.linear_model import HuberRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import r2_score
from scipy.optimize import nnls
from scipy.optimize import least_squares
import h5py
from joblib import Parallel, delayed
from typing import List, Sequence, Tuple


def calculate_rmse(beta_m: pd.DataFrame,
                   ref_m: pd.DataFrame,
                   est_m: pd.DataFrame,
                   verbose: bool = False):
    """
    """

    # Iterate over samples
    rmse_v = np.zeros(beta_m.shape[0])
    for i in range(beta_m.shape[0]):
        if verbose: print(beta_m.index.values[i])
        sample_m = beta_m.iloc[i,:]
        selected = ~pd.isnull(sample_m)

        # Calculate RMSE
        reconst_m = np.matmul(ref_m.loc[:,selected].values.T, est_m.iloc[i,:].values)
        rmse_v[i] = np.sqrt(np.mean((sample_m.values[selected] - reconst_m)**2))

    return rmse_v


def rms(y, yfit):
    return np.sqrt(np.sum((y-yfit)**2))


def calculate_diff(beta_m: pd.DataFrame,
                   ref_m: pd.DataFrame,
                   est_m: pd.DataFrame,
                   verbose: bool = False):
    """
    """

    # Iterate over samples
    r2 = np.zeros(beta_m.shape[0])
    for i in range(beta_m.shape[0]):
        if verbose: print(beta_m.index.values[i])
        sample_m = beta_m.iloc[i,:]
        selected = ~pd.isnull(sample_m)

        # Calculate RMSE
        if selected.sum() > 0:
            reconst_m = np.matmul(ref_m.loc[:,selected].values.T, est_m.iloc[i,:].values)
            r2[i] = np.median(np.abs(sample_m.values[selected] - reconst_m))
        else:
            r2[i] = np.nan

    return r2


def run_cibertsort(beta_m: pd.DataFrame,
                   ref_m: pd.DataFrame,
                   n_features: int = 2000,
                   verbose: bool = False):
    """
    Parameters
    ----------
        beta_m : pd.DataFrame
            Beta values
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        verbose : bool
            If True, print progress
    
    Returns
    -------
        est_m : pd.DataFrame
            Estimated cell fractions
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Iterate over samples
    est_m = np.zeros((beta_m.shape[0], ref_m.shape[0]))
    for i in range(beta_m.shape[0]):
        if verbose: print(beta_m.index.values[i])
        sample_m = beta_m.iloc[i,:]

        est = cibersort(sample_m, ref_m, n_features=n_features)
        est_m[i,:] = est

    # Format DataFrame
    est_m = pd.DataFrame(est_m, index=beta_m.index.values, columns=ref_m.index.values)

    return est_m


def cibersort(sample_m: np.ndarray,
              ref_m: np.ndarray,
              n_features: int = 2000,
              nu_v: List[int] = [0.25, 0.5, 0.75]) -> np.ndarray:
    """
    Parameters
    ----------
        sample_m : np.ndarray
            Sample matrix
        ref_m : np.ndarray
            Reference matrix
        n_features : int
            Number of features to use
        nu_v : List[int]
            List of nu values to use

    Returns
    -------
        est : np.ndarray
            Estimated cell fractions
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]

    if len(selected) > 5:
        # Scale values
        sample_scaled = (sample_m.loc[selected].values - sample_m.loc[selected].values.mean()) / sample_m.loc[selected].values.std()
        ref_m_scaled = (ref_m.loc[:,selected].values - ref_m.loc[:,selected].values.mean()) / ref_m.loc[:,selected].values.std()

        est_lm = {}
        nui = 0

        # Iterate over nu values
        for nu in nu_v:
            
            # Fit model
            nusvr = NuSVR(C=1.0, nu=nu, kernel="linear")
            nusvr.fit(ref_m_scaled.T, sample_scaled)

            # Calculate coefficients
            coef_v = np.matmul(nusvr.dual_coef_, nusvr.support_vectors_)
            coef_v[coef_v<0] = 0
            total = np.sum(coef_v)
            coef_v = coef_v / total

            est_lm[nui] = coef_v
            nui += 1
        
        # Determin best nu
        rmse_m = np.zeros(len(nu_v))
        for nui in range(len(nu_v)):
            reconst_m = np.matmul(ref_m.loc[:,selected].values.T, est_lm[nui].T).T
            rmse_m[nui] = np.sqrt(np.mean((sample_m.loc[selected].values - reconst_m)**2, axis=1))
        nu_idx = np.argmin(rmse_m)

        est = est_lm[nu_idx][0]

    else:
        est = np.zeros(ref_m.shape[0])

    return est


def huber(sample_m: pd.DataFrame,
         ref_m: pd.DataFrame,
         n_features: int = 2000,
         return_rms: bool = False) -> np.ndarray:
    """
    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        
    Returns
    -------
        coef_v : np.ndarray
            Estimated cell fractions
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]

    if len(selected) > 5:
        # Fit model
        hr = HuberRegressor(max_iter=10000)
        hr.fit(ref_m.loc[:,selected].values.T, sample_m.loc[selected].values)

        # Calculate coefficients
        coef_v = hr.coef_
        coef_v[coef_v < 0] = 0
        total = np.sum(coef_v)
        coef_v = coef_v / total

    else:
        coef_v = np.zeros(ref_m.shape[0])

    return coef_v


def huber_weighted(sample_m: pd.DataFrame,
                ref_m: pd.DataFrame,
                n_features: int = 2000,
                return_rms: bool = False) -> np.ndarray:
    """
    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        
    Returns
    -------
        coef_v : np.ndarray
            Estimated cell fractions
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]
    weights = var[selected].values
    weights = weights / weights.sum()

    if len(selected) > 5:
        # Fit model
        hr = HuberRegressor(max_iter=10000)
        hr.fit(np.sqrt(weights)[:, None] * ref_m.loc[:,selected].values.T,
                      np.sqrt(weights) * sample_m.loc[selected].values)

        # Calculate coefficients
        coef_v = hr.coef_
        coef_v[coef_v < 0] = 0
        total = np.sum(coef_v)
        coef_v = coef_v / total

    else:
        coef_v = np.zeros(ref_m.shape[0])

    return coef_v


def normalize_coefs(coef_v: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
        coef_v : np.ndarray
            Coefficient vector
    
    Returns
    -------
        coef_v : np.ndarray
            Normalized coefficient vector
    """

    coef_v[coef_v < 0] = 0
    total = np.sum(coef_v)
    coef_v = coef_v / total

    return coef_v


def calculate_rmse(sample_m: pd.DataFrame,
                   ref_m: pd.DataFrame,
                   coef_v: np.ndarray) -> float:
    """

    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        coef_v : np.ndarray
            Coefficient vector
    
    Returns
    -------
        rmse : float
            Root mean squared error
    """

    reconst_m = np.matmul(ref_m, coef_v)
    rmse = np.sqrt(np.mean((sample_m - reconst_m)**2))

    return rmse


def huber_regress(ref_m: np.ndarray,
                  sample_m: np.ndarray,
                  weighted: bool = False) -> np.ndarray:
    """

    Parameters
    ----------
        ref_m : np.ndarray
            Reference matrix
        sample_m : np.ndarray
            Sample matrix
        weighted : bool
            If True, use weighted regression
    
    Returns
    -------
        rmse : float
    """

    # Fit model
    hr = HuberRegressor(max_iter=10000)
    if weighted:
        weights = ref_m.var(axis=1)
        weights = weights / weights.sum()
        hr.fit(np.sqrt(weights)[:, None] * ref_m,
                        np.sqrt(weights) * sample_m)
        score = hr.score(ref_m, sample_m, sample_weight=weights)
    else:
        hr.fit(ref_m, sample_m)
        score = hr.score(ref_m, sample_m)

    # Calculate coefficients
    if np.sum(hr.coef_ > 0) == 0:
        #rmse = np.nan
        coef_v = np.zeros(ref_m.shape[1])
    else:
        coef_v = normalize_coefs(hr.coef_)
        #rmse = calculate_rmse(sample_m, ref_m, coef_v)

    return coef_v, score


def nnls_regress(ref_m: np.ndarray,
                  sample_m: np.ndarray,
                  weighted: bool = False) -> Tuple[float,float]:
    """

    Parameters
    ----------
        ref_m : np.ndarray
            Reference matrix
        sample_m : np.ndarray
            Sample matrix
        weighted : bool
            If True, use weighted regression
    
    Returns
    -------
        rmse : float
    """

    # Fit model
    if weighted:
        weights = ref_m.var(axis=1)
        weights = weights / weights.sum()
        coef_v = nnls(np.sqrt(weights)[:, None] * ref_m,
                        np.sqrt(weights) * sample_m)[0]
    else:
        coef_v = nnls(ref_m, sample_m)[0]

    # Calculate coefficients
    if np.sum(coef_v > 0) == 0:
        rmse = np.nan
        coef_v = np.zeros(ref_m.shape[1])
    else:
        coef_v = normalize_coefs(coef_v)
        rmse = calculate_rmse(sample_m, ref_m, coef_v)

    return coef_v, rmse


def permute_regress(ref_m: np.ndarray,
                    sample_m: np.ndarray,
                    weighted: bool = False,
                    method: str = "huber") -> np.ndarray:
    """

    Parameters
    ----------
        ref_m : np.ndarray
            Reference matrix
        sample_m : np.ndarray
            Sample matrix
        weighted : bool
            If True, use weighted regression
    
    Returns
    -------
        rmse : float
    """

    # Selecte method
    if method == "huber":
        func = huber_regress
    elif method == "nnls":
        func = nnls_regress

    # Permute
    perm = np.random.permutation(sample_m)

    # Run regression
    coefs, r = func(ref_m, perm, weighted)

    return r


def permute(sample_m: pd.DataFrame,
            ref_m: pd.DataFrame,
            n_features: int = 2000,
            n: int = 1000,
            n_jobs: int = 1,
            method: str = "huber",
            weighted: bool = False) -> np.ndarray:
    """
    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        
    Returns
    -------
        coef_v : np.ndarray
            Estimated cell fractions
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]

    # Determine regression function
    if method == "huber":
        func = huber_regress
    elif method == "nnls":
        func = nnls_regress
    

    if len(selected) > 5:
        obs_coefs, obs_r = func(ref_m.loc[:,selected].values.T, sample_m.loc[selected].values, weighted)
        rmse = Parallel(n_jobs=n_jobs)(delayed(permute_regress)(ref_m.loc[:,selected].values.T, sample_m.loc[selected].values, weighted, method) for i in range(n))
        rmse = np.array(rmse)

    else:
        rmse = np.zeros(n)
        rmse[:] = np.nan

    # Calculate pvalue
    #pvalue = np.sum(rmse < obs_r) / n
    pvalue = np.sum(rmse > obs_r) / n
    pvalue = max(pvalue, 1/n)

    return obs_coefs, obs_r, rmse, pvalue


def NNLS(sample_m: pd.DataFrame,
         ref_m: pd.DataFrame,
         n_features: int = 2000) -> np.ndarray:
    """

    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
    
    Returns
    -------
        coef_v : np.ndarray
            Estimated cell fractions
    
    
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]

    if len(selected) > 5:
        # Fit model
        coef_v = nnls(ref_m.loc[:,selected].values.T, sample_m.loc[selected].values)[0]

        # Calculate coefficients
        coef_v[coef_v < 0] = 0
        total = np.sum(coef_v)
        coef_v = coef_v / total

    else:
        coef_v = np.zeros(ref_m.shape[0])

    return coef_v


def NNLS_weighted(sample_m: pd.DataFrame,
                  ref_m: pd.DataFrame,
                  n_features: int = 2000) -> np.ndarray:
    """

    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
    
    Returns
    -------
        coef_v : np.ndarray
            Estimated cell fractions
    
    
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]
    weights = var[selected].values
    weights = weights / weights.sum()

    if len(selected) > 5:
        # Fit model
        coef_v = nnls(np.sqrt(weights)[:, None] * ref_m.loc[:,selected].values.T,
                      np.sqrt(weights) * sample_m.loc[selected].values)[0]

        # Calculate coefficients
        coef_v[coef_v < 0] = 0
        total = np.sum(coef_v)
        coef_v = coef_v / total

    else:
        coef_v = np.zeros(ref_m.shape[0])

    return coef_v


def least_square(sample_m: pd.DataFrame,
                ref_m: pd.DataFrame,
                n_features: int = 2000) -> np.ndarray:
    """

    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        
    Returns
    -------
        coef_v : np.ndarray
            Estimated cell fractions
    """

    # Calculate variable features
    var = ref_m.var(axis=0)

    # Select sample features
    sample_var = var.loc[ref_m.columns.values[~pd.isnull(sample_m.values)]]
    n_used_features = min(n_features, len(sample_var))
    selected = sample_var.index.values[np.argsort(sample_var.values)[-n_used_features:]]

    if len(selected) > 10:
        # Fit model
        weights = np.zeros(ref_m.shape[0])
        obj_func = lambda weights: np.linalg.norm(sample_m.loc[selected].values - np.dot(weights,ref_m.loc[:,selected].values))
        coef_v = least_squares(obj_func, weights).x

        # Calculate coefficients
        coef_v[coef_v < 0] = 0
        total = np.sum(coef_v)
        coef_v = coef_v / total

    else:
        coef_v = np.zeros(ref_m.shape[0])

    return coef_v


def sample_iter_decon(sample_m: pd.DataFrame,
                        ref_m: pd.DataFrame,
                        n_features: int = 2000,
                        method: str = "huber") -> pd.Series:
    """

    Parameters
    ----------
        sample_m : pd.DataFrame
            Sample matrix
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        method : str
            Method to use for deconvolution
    
    Returns
    -------
        est_m : pd.Series
            Estimated cell fractions
    """

    est_m = pd.Series(np.zeros(ref_m.shape[0]),
                      index = ref_m.index.values)

    select = np.arange(ref_m.shape[0])
    n_select = 0

    while len(select) != n_select:
        n_select = len(select)
        if n_select == 0:
            break
        
        if method == "huber":
            coef_v = huber(sample_m,
                            ref_m.iloc[select,:],
                            n_features=n_features)
        elif method == "ciber":
            coef_v = cibersort(sample_m,
                                ref_m.iloc[select,:],
                                n_features=n_features)
        elif method == "nnls":
            coef_v = NNLS(sample_m,
                            ref_m.iloc[select,:],
                            n_features=n_features)
        elif method == "wnnls":
            coef_v = NNLS_weighted(sample_m,
                                   ref_m.iloc[select,:],
                                   n_features=n_features)
        elif method == "lstsq":
                coef_v = least_square(sample_m,
                              ref_m.iloc[select,:],
                              n_features=n_features)
        select = select[coef_v > 0.0]

    if len(select) > 0:
        est_m.values[select] = coef_v

    return est_m


def iter_decon(beta_m: pd.DataFrame,
               ref_m: pd.DataFrame,
               n_features: int = 2000,
               method: str = "huber",
               controls: Sequence[str] = np.array(["CONTR_INFLAM",
                                              "CONTR_REACT",
                                              "CONTR_CEBM",
                                              "PLASMA"]),
               verbose: bool = False) -> pd.DataFrame:
    """

    Parameters
    ----------
        beta_m : pd.DataFrame
            Beta values
        ref_m : pd.DataFrame
            Reference matrix
        n_features : int
            Number of features to use
        method : str
            Method to use for deconvolution
        controls : Sequence[str]
            List of control cell types
        verbose : bool
            If True, print progress

    Returns
    -------
        est_m : pd.DataFrame
            Estimated cell fractions
    """

    # Select controls
    is_control = np.in1d(ref_m.index.values, controls)

    # Initialize estimation matrix
    est_m = pd.DataFrame(np.zeros((beta_m.shape[0], ref_m.shape[0])),
                         index = beta_m.index.values,
                         columns = ref_m.index.values)
                     

    # Iterate over samples
    for i, s in enumerate(beta_m.index.values):
        if verbose: print(s)

        # Calculate first estimates
        sample_decon = sample_iter_decon(beta_m.iloc[i,:],
                                            ref_m,
                                            n_features=n_features,
                                            method=method)
        
        if np.max(sample_decon.values[~is_control]) > 0:
            top_hit = sample_decon.index.values[~is_control][np.argmax(sample_decon.values[~is_control])]
            selected = np.append(controls, top_hit)

            # Calculate estimates
            sample_decon = sample_iter_decon(beta_m.iloc[i,:],
                                                ref_m.loc[selected,:],
                                                n_features=n_features,
                                                method=method)
        # Set estimates
        est_m.loc[s, sample_decon.index.values] = sample_decon.values
        
    return est_m


def make_signature_matrix(ref_m: pd.DataFrame,
                          groups: Sequence[str],
                          method: str = "mean") -> pd.DataFrame:
    """

    Parameters
    ----------
        ref_m : pd.DataFrame
            Reference matrix
        groups : Sequence[str]
            List of groups
        method : str
            Method to use for calculating signature matrix

    Returns
    -------
        sig_matrix : pd.DataFrame
            Signature matrix
    """

    # Calculate matrix
    if method == "mean":
        sig_matrix = ref_m.groupby(groups).mean()
    elif method == "median":
        sig_matrix = ref_m.groupby(groups).median()
    else:
        raise ValueError("Method not implemented")
    
    return sig_matrix