# Description: Classify methylation data using MPACT model
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List
from joblib import Parallel, delayed
from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler

# Local imports
from ...data.import_data import get_data_file
from .decompose import huber_regress


def _normal_decomposition(sample, betas, reference, normal_fluids, normal_tissues):
    """
    Process a single sample for decomposition with logit transformation.
    """
    # Extract sample betas and remove NaNs
    sample_betas = betas.loc[sample, :].dropna()

    # Identify common probes
    common = reference.index.intersection(sample_betas.index).values
    if len(common) == 0:
        return sample, None

    # Filter reference and sample to common probes
    sample_ref = reference.loc[common, :]
    sample_betas = sample_betas.loc[common]

    # Apply logit transformation
    sample_betas = logit(np.clip(sample_betas, 1e-6, 1 - 1e-6))

    # Select top 4000 most variable probes
    max_iter = 5  # Prevent infinite looping
    iteration = 0
    coef_v = np.zeros(sample_ref.shape[1])
    while np.any(coef_v == 0) and sample_ref.shape[0] > 400 and iteration < max_iter:
        var_probes = sample_ref.var(axis=1).nlargest(4000).index
        X = sample_ref.loc[var_probes, :].values
        y = sample_betas.loc[var_probes].values
        
        coef_v, _ = huber_regress(X, y)
        
        # Ensure at least some non-zero coefficients remain
        if np.sum(coef_v != 0) == 0:
            break
        
        # Keep only features with nonzero coefficients
        sample_ref = sample_ref.loc[:, coef_v != 0]
        coef_v = coef_v[coef_v != 0]
        iteration += 1

    # Convert coefficients back (not needed for decomposition)
    coefs = pd.Series(coef_v, index=sample_ref.columns)
    decom_sample = pd.Series(index=normal_fluids + ["Neuron"], dtype=float)
    decom_sample[:] = 0  # Initialize to avoid missing values

    common_fluids = sample_ref.columns.intersection(normal_fluids)
    common_tissues = sample_ref.columns.intersection(normal_tissues)

    if len(common_fluids) > 0:
        decom_sample[common_fluids] = coefs.loc[common_fluids].values
    if len(common_tissues) > 0:
        decom_sample["Neuron"] = coefs.loc[common_tissues].values.sum()

    return sample, decom_sample


def normal_decomposition(betas: pd.DataFrame,
                         normal_fluids: list[str] = ['B', 'B-Mem',
                                            'Granulocytes', 'Monocytes',
                                            'NK', 'T-CD3', 'T-CD4',
                                            'T-CD8', 'T-CenMem-CD4',
                                            'T-Eff-CD8', 'T-EffMem-CD4',
                                            'T-EffMem-CD8', 'T-Naive-CD4',
                                            'T-Naive-CD8', 'Macrophages','Vein-Endothel'],
                         normal_tissues: list[str] = ['CONTR_CEBM', 'CONTR_HEMI'],
                         n_jobs: int = -1,
                         verbose: bool = False) -> pd.DataFrame:
    """
    Decomposes methylation beta values into contributions from normal tissues.
    Uses Huber regression for robustness and parallel processing for speedup.
    """
    # Load reference data
    file = get_data_file("BrainTumorDeconRef.parquet")
    reference = pd.read_parquet(file)

    # Combine normal fluids and tissues
    normals = normal_fluids + normal_tissues
    reference = reference.loc[:, normals]

    # Parallelize processing of each sample
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_normal_decomposition)(sample, betas, reference, normal_fluids, normal_tissues)
        for sample in betas.index
    )

    # Construct results DataFrame
    decom = pd.DataFrame(index=betas.index, columns=normal_fluids + ["Neuron"], dtype=float)
    for sample, values in results:
        if values is not None:
            decom.loc[sample] = values

    return decom


def _remove_normal_csf(name, sample, X, reference_fraction=1.0):
    """
    Process a single sample: perform regression and return the adjusted values.
    """
    # Identify non-missing features
    valid_mask = ~pd.isnull(sample)
    if valid_mask.sum() == 0:
        return name, None  # Skip empty samples

    sample = sample[valid_mask]
    X_valid = X[valid_mask, :]

    # Apply logit transformation
    methylation_beta = np.clip(sample, 1e-6, 1 - 1e-6)
    y_logit = logit(methylation_beta)

    # Fit GLM regression
    model = sm.GLM(y_logit, X_valid, family=sm.families.Gaussian()).fit()

    # Compute residuals
    residuals = y_logit - model.predict(X_valid)

    # Transform back to beta values
    residuals_beta = expit(residuals)

    # Adjust for reference fraction
    # residuals_beta = methylation_beta - residuals_beta
    if reference_fraction < 1.0:
        residuals_beta = (residuals_beta * reference_fraction) + (methylation_beta * (1 - reference_fraction))
    residuals_beta[residuals_beta < 0.5] = residuals_beta[residuals_beta < 0.5] - 1e-6
    residuals_beta[residuals_beta > 0.5] = residuals_beta[residuals_beta > 0.5] + 1e-6

    # Clip values to [0, 1]
    residuals_beta = np.clip(residuals_beta, 0, 1)

    # Scale
    scaler = MinMaxScaler()
    residuals_beta = scaler.fit_transform(residuals_beta.values.reshape(-1, 1)).flatten()

    return name, pd.Series(residuals_beta, index=sample.index)


def remove_normal_csf(samples: pd.DataFrame,
                      normals: list[str] = ['B', 'B-Mem',
                                            'Granulocytes', 'Monocytes',
                                            'NK', 'T-CD3', 'T-CD4',
                                            'T-CD8', 'T-CenMem-CD4',
                                            'T-Eff-CD8', 'T-EffMem-CD4',
                                            'T-EffMem-CD8', 'T-Naive-CD4',
                                            'T-Naive-CD8', 'Macrophages',
                                            'ControlCSF', 'CONTR_REACT', 'CONTR_INFLAM', 'PLASMA',
                                            'Blood', 'IMMUNE','Vein-Endothel'],
                      n_jobs: int = -1,
                      reference_fraction: float = 0.5,
                      verbose: bool = False):
    """
    Removes normal tissue influence using parallelized Gaussian GLM regression on methylation beta values.
    """
    # Load reference data
    file = get_data_file("BrainTumorDeconRef.parquet")
    reference = pd.read_parquet(file)

    # Select normal tissue reference columns
    reference = reference.loc[:, normals]

    # Find common probes
    common_probes = reference.index.intersection(samples.columns)
    if common_probes.empty:
        raise ValueError("No common probes found between reference and sample data.")

    reference = reference.loc[common_probes, :]
    samples = samples.loc[:, common_probes]

    # Prepare regression matrix
    X = sm.add_constant(reference).values

    # Parallel processing of samples
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_remove_normal_csf)(name, samples.loc[name, :], X, reference_fraction=reference_fraction)
        for name in samples.index
    )

    # Construct results DataFrame
    new_values = pd.DataFrame(index=samples.index, columns=samples.columns, dtype=float)
    for name, adjusted_values in results:
        if adjusted_values is not None:
            new_values.loc[name, adjusted_values.index] = adjusted_values

    return new_values


def tumor_decomposition(betas: pd.DataFrame,
                        tumor_types: np.ndarray,
                        normals: list[str] = ['B', 'B-Mem',
                                            'Granulocytes', 'Monocytes',
                                            'NK', 'T-CD3', 'T-CD4',
                                            'T-CD8', 'T-CenMem-CD4',
                                            'T-Eff-CD8', 'T-EffMem-CD4',
                                            'T-EffMem-CD8', 'T-Naive-CD4',
                                            'T-Naive-CD8', 'Macrophages',
                                            'ControlCSF', 'CONTR_REACT', 'CONTR_INFLAM', 'PLASMA',
                                            'Blood', 'IMMUNE','Vein-Endothel'],
                        n_features: int = 4000,
                      verbose: bool = False):
    """
    """

    # Read normal tissues
    file = get_data_file("BrainTumorDeconRef.parquet")
    reference = pd.read_parquet(file)

    # Iterate over samples
    tumor_purity = pd.Series(np.zeros(betas.shape[0]), index=betas.index)
    for i, sample in enumerate(betas.index.values):
        if verbose:
            print("Decomposing", sample, flush=True)
        # Match
        tumor_type = tumor_types[i]
        if tumor_type in normals or tumor_type == "Control":
            continue
        e = [tumor_type] + normals
        #sample_ref = reference.loc[:,normals+[tumor_type]]
        sample_betas = betas.loc[sample,:]
        # Remove nan
        sample_betas = sample_betas[~pd.isnull(sample_betas)]

        common = reference.index.intersection(sample_betas.index).values
        if len(common) == 0:
            continue
        sample_ref = reference.loc[common,:]
        sample_betas = sample_betas.loc[common]

        # Find variable probes
        var = np.argsort(sample_ref.var(axis=1).values)[-n_features:]
        coef_v, score = huber_regress(sample_ref.loc[:,e].values[var,:], sample_betas.values[var])
        if np.sum(coef_v[1:] == 0) and coef_v[0] > 0:
            e = [tumor_type] + list(np.array(normals)[coef_v[1:] > 0])
            coef_v, score = huber_regress(sample_ref.loc[:,e].values[var,:], sample_betas.values[var])

        tumor_purity.loc[sample] = coef_v[0]

    return tumor_purity





