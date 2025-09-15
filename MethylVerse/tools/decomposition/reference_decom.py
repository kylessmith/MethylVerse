# Description: Classify methylation data using MPACT model
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List
from joblib import Parallel, delayed
from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import nnls

# Local imports
from ...data.import_data import get_data_file
from .decompose import huber_regress


def _normal_decomposition(sample, betas, reference, normal_fluids, normal_tissues, tumor_type=None, n_features=4000):
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

    # Eliminate other tumors
    if tumor_type == "NonmalignantBackground":
        tumor_type = None
    if tumor_type is not None:
        columns = normal_fluids + normal_tissues + [tumor_type]
    else:
        columns = normal_fluids + normal_tissues
    sample_ref = sample_ref.loc[:, columns]

    # Select top 4000 most variable probes
    max_iter = 5  # Prevent infinite looping
    iteration = 0
    coef_v = np.zeros(sample_ref.shape[1])
    while np.any(coef_v == 0) and sample_ref.shape[0] > 400 and iteration < max_iter:
        var_probes = sample_ref.var(axis=1).nlargest(n_features).index
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
    if tumor_type is not None:
        decom_sample = pd.Series(index=normal_fluids + ["Neuron"], dtype=float)
    else:
        decom_sample = pd.Series(index=normal_fluids + ["Tumor"] + ["Neuron"], dtype=float)
    decom_sample[:] = 0  # Initialize to avoid missing values

    common_fluids = sample_ref.columns.intersection(normal_fluids)
    common_tissues = sample_ref.columns.intersection(normal_tissues)
    if tumor_type is not None:
        common_tumor = sample_ref.columns.intersection([tumor_type])


    if len(common_fluids) > 0:
        decom_sample[common_fluids] = coefs.loc[common_fluids].values
    if len(common_tissues) > 0:
        decom_sample["Neuron"] = coefs.loc[common_tissues].values.sum()
    if tumor_type is not None and len(common_tumor) > 0:
        decom_sample["Tumor"] = coefs.loc[common_tumor].values.sum()

    return sample, decom_sample


def normal_decomposition(betas: pd.DataFrame,
                         normal_fluids: list[str] = ['B', 'B-Mem',
                                            'Granulocytes', 'Monocytes',
                                            'NK', 'T-CD3', 'T-CD4',
                                            'T-CD8', 'T-CenMem-CD4',
                                            'T-Eff-CD8', 'T-EffMem-CD4',
                                            'T-EffMem-CD8', 'T-Naive-CD4',
                                            'T-Naive-CD8', 'Macrophages', "DendriticCells",
                                            'Vein-Endothel','Treg','Oligodendrocytes','Microglia'],
                         normal_tissues: list[str] = ['CONTR_CEBM', 'CONTR_HEMI'],
                         tumor_types: List[str] = None,
                         n_jobs: int = -1,
                         ref_file: str = "BrainTumorDeconRef.parquet",
                         n_features=4000,
                         verbose: bool = False) -> pd.DataFrame:
    """
    Decomposes methylation beta values into contributions from normal tissues.
    Uses Huber regression for robustness and parallel processing for speedup.
    """
    # Load reference data
    if ref_file == "BrainTumorDeconRef.parquet":
        file = get_data_file("BrainTumorDeconRef.parquet")
    else:
        file = ref_file
    reference = pd.read_parquet(file)

    # Combine normal fluids and tissues
    normals = normal_fluids + normal_tissues
    if tumor_types is not None:
        normals = normals + list(set(pd.unique(tumor_types)) - set(["NonmalignantBackground"]))
        tumor_types = {sample: tumor_types[i] for i, sample in enumerate(betas.index)}
    reference = reference.loc[:, normals]

    # Parallelize processing of each sample
    if tumor_types is None:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_normal_decomposition)(sample, betas, reference, normal_fluids, normal_tissues, None, n_features)
            for sample in betas.index
        )
    else:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_normal_decomposition)(sample, betas, reference, normal_fluids, normal_tissues, tumor_types[sample], n_features)
            for sample in betas.index
        )

    # Construct results DataFrame
    if tumor_types is not None:
        decom = pd.DataFrame(index=betas.index, columns=normal_fluids + ["Neuron"] + ["Tumor"], dtype=float)
    else:
        decom = pd.DataFrame(index=betas.index, columns=normal_fluids + ["Neuron"], dtype=float)
    for sample, values in results:
        if values is not None:
            decom.loc[sample] = values

    return decom


def _remove_normal_csf(name, sample, X, max_fraction=1.0):
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
    #methylation_beta = np.clip(sample, 1e-6, 1 - 1e-6)
    #y_logit = logit(methylation_beta)

    # Fit GLM regression
    #model = sm.GLM(y_logit, X_valid, family=sm.families.Gaussian()).fit()
    coeffs, residual = nnls(X_valid, sample.values)
    total_contamination = min(np.sum(coeffs), max_fraction)

    # Compute residuals
    #residuals = y_logit - model.predict(X_valid)

    # Transform back to beta values
    #residuals_beta = expit(residuals)

    # Calculate contamination contribution
    #contamination = X_valid @ coeffs
    contamination = np.average(X_valid, axis=1, weights=coeffs)

    if total_contamination > 0:
        # Estimate pure tumor signal
        if total_contamination < 1.0:
            pure_tumor = (sample.values - (total_contamination * contamination)) / (1 - total_contamination)
            pure_tumor = np.clip(pure_tumor, 0, 1)
        else:
            # Highly contaminated - use residuals
            pure_tumor = sample.values - contamination
            pure_tumor = np.clip(pure_tumor, 0, 1)
        
        # Check if decontamination makes sense
        decontaminated = pd.Series(pure_tumor, index=sample.index)
    else:
        # No contamination detected, return original sample
        decontaminated = pd.Series(sample, index=sample.index)

    # Adjust for reference fraction
    # residuals_beta = methylation_beta - residuals_beta
    #if reference_fraction < 1.0:
    #    residuals_beta = (residuals_beta * reference_fraction) + (methylation_beta * (1 - reference_fraction))
    #residuals_beta[residuals_beta < 0.5] = residuals_beta[residuals_beta < 0.5] - 1e-6
    #residuals_beta[residuals_beta > 0.5] = residuals_beta[residuals_beta > 0.5] + 1e-6

    # Clip values to [0, 1]
    #residuals_beta = np.clip(residuals_beta, 0, 1)

    # Scale
    #scaler = MinMaxScaler()
    #residuals_beta = scaler.fit_transform(residuals_beta.values.reshape(-1, 1)).flatten()

    return name, decontaminated


def remove_normal_csf(samples: pd.DataFrame,
                      normals: list[str] = ['B', 'B-Mem',
                                            'Granulocytes', 'Monocytes',
                                            'NK', 'T-CD3', 'T-CD4',
                                            'T-CD8', 'T-CenMem-CD4',
                                            'T-Eff-CD8', 'T-EffMem-CD4',
                                            'T-EffMem-CD8', 'T-Naive-CD4',
                                            'T-Naive-CD8', 'Macrophages',
                                            'ControlCSF', 'CONTR_REACT', 'CONTR_INFLAM', 'PLASMA',
                                            'Blood', 'IMMUNE'],
                      n_jobs: int = -1,
                      max_fraction: float = 0.5,
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
    #X = sm.add_constant(reference).values
    X = reference.values

    # Parallel processing of samples
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_remove_normal_csf)(name, samples.loc[name, :], X, max_fraction=max_fraction)
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
                                            'T-Naive-CD8', 'Macrophages', "DendriticCells",
                                            'Vein-Endothel','Treg','Oligodendrocytes','Microglia',
                                            'CSF', 'CONTR_REACT', 'CONTR_INFLAM', 'PLASMA',
                                            'Blood', 'IMMUNE','Astrocyte','CONTR_ADENOPIT',
                                            'CONTR_CEBM', 'CONTR_HEMI', 'CONTR_HYPTHAL',
                                            'CONTR_PINEAL', 'CONTR_PONS', 'CONTR_WM'],
                        n_features: int = 4000,
                        min_purity: float = 0.05,
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
        if tumor_type in normals or tumor_type == "Control" or tumor_type == "NonmalignantBackground":
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
        updated_normals = np.array(normals).copy()
        var = np.argsort(sample_ref.var(axis=1).values)[-n_features:]
        coef_v, score = huber_regress(sample_ref.loc[:,e].values[var,:], sample_betas.values[var])
        if coef_v[0] < min_purity:
            tumor_purity.loc[sample] = 0.0
            continue
        
        # Serially remove normals with zero coefficients
        while np.sum(coef_v[1:] == 0) > 0 and coef_v[0] > 0:
            # If all coefficients are zero, try again with only non-zero coefficients
            updated_normals = updated_normals[coef_v[1:] > 0]
            e = [tumor_type] + list(updated_normals)
            coef_v, score = huber_regress(sample_ref.loc[:,e].values[var,:], sample_betas.values[var])

        # Store tumor purity
        if len(coef_v) > 0:
            tumor_purity.loc[sample] = coef_v[0]

    return tumor_purity


def tumor_decomposition_search(betas: pd.DataFrame,
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
    tumor_types = reference.columns.difference(normals).values
    if len(tumor_types) == 0:
        raise ValueError("No tumor types found in the reference data.")
    tumor_purity = pd.DataFrame(np.zeros(betas.shape[0], len(tumor_types)), index=betas.index, columns=tumor_types)

    for tumor_type in tumor_types:
        if verbose:
            print("Decomposing tumor type:", tumor_type, flush=True)
        for i, sample in enumerate(betas.index.values):
            if verbose:
                print("Decomposing", sample, flush=True)
            # Match
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
            if np.sum(coef_v[1:] == 0) and coef_v[0] > 0:
                e = [tumor_type] + list(np.array(normals)[coef_v[1:] > 0])
                coef_v, score = huber_regress(sample_ref.loc[:,e].values[var,:], sample_betas.values[var])

            tumor_purity.loc[sample,tumor_type] = coef_v[0]

    # Determine the tumor type with the highest purity for each sample
    tumor_purity['MaxPurity'] = tumor_purity.max(axis=1)
    tumor_purity['PredictedTumorType'] = tumor_purity[tumor_types].idxmax(axis=1)

    return tumor_purity




