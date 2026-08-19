# Description: Classify methylation data using MPACT model
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Sequence, Tuple
from joblib import Parallel, delayed
from scipy.special import logit, expit
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import nnls

# Local imports
from multiverse_cache import get_data_file
from .decompose import huber_regress, nnls_regress


def _normalize_decomposition_result(values: pd.Series) -> pd.Series:
    values = values.fillna(0).clip(lower=0)
    total = values.sum()
    if total > 0:
        values = values / total
    return values


def _apply_transform(values, logit_transform: bool):
    """
    Put betas onto the regression scale, applied IDENTICALLY to the sample and
    to the reference so the two sides always live in the same space.

    NOTE on modelling: cell-type fractions mix linearly in beta (proportion)
    space, i.e. y_beta ~= X_beta @ fractions. A logit transform breaks that
    linearity, so logit_transform=True trades model-exactness for variance
    stabilisation at the extremes -- the important thing is that BOTH sides get
    the same treatment (the previous code logit-transformed only the sample).
    For a strictly mixing-model-correct fit, use logit_transform=False.

    Accepts a Series (sample) or DataFrame (reference) and returns the same type.
    """
    if not logit_transform:
        return values.astype(float)
    clipped = np.clip(values.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    transformed = logit(clipped)
    if isinstance(values, pd.DataFrame):
        return pd.DataFrame(transformed, index=values.index, columns=values.columns)
    return pd.Series(transformed, index=values.index)


def _load_decomposition_reference(ref_file: str = "BrainTumorDeconRef.parquet",
                                  custom_classes: str | None = None) -> pd.DataFrame:
    if ref_file == "BrainTumorDeconRef.parquet":
        file = get_data_file("BrainTumorDeconRef.parquet")
    else:
        file = ref_file
    reference = pd.read_parquet(file)

    if custom_classes is not None:
        custom_class_ref = pd.read_parquet(custom_classes)
        common_probes = reference.index.intersection(custom_class_ref.index)
        if len(common_probes) == 0:
            raise ValueError("No common probes found between original reference and custom class reference.")
        reference = reference.loc[common_probes, :]
        custom_class_ref = custom_class_ref.loc[common_probes, :]
        reference = reference.drop(columns=custom_class_ref.columns, errors='ignore').join(custom_class_ref)

    return reference


def _perturb_sample_betas(sample_betas: pd.Series,
                          rng: np.random.Generator,
                          sparsity: float = 0.0,
                          noise_scale: float = 0.0) -> pd.Series:
    perturbed = sample_betas.copy()

    if sparsity > 0 and len(perturbed) > 1:
        drop_count = min(int(np.floor(len(perturbed) * sparsity)), len(perturbed) - 1)
        if drop_count > 0:
            drop_idx = rng.choice(perturbed.index.to_numpy(), size=drop_count, replace=False)
            perturbed = perturbed.drop(index=drop_idx)

    if noise_scale > 0 and len(perturbed) > 0:
        noise = rng.normal(loc=0.0, scale=noise_scale, size=len(perturbed))
        perturbed = pd.Series(
            np.clip(perturbed.to_numpy(dtype=float) + noise, 1e-6, 1 - 1e-6),
            index=perturbed.index,
        )

    return perturbed


def _select_marker_features(sample_ref: pd.DataFrame, feature_count: int) -> pd.Index:
    """
    Select marker probes balanced across reference cell types using z-scores.
    For each column (cell type) the top markers_per_class probes by absolute
    z-score are collected; the union is returned.  This ensures every cell type
    contributes discriminative probes rather than the selection being dominated
    by the most variable components.
    """
    n_cols = sample_ref.shape[1]
    n_probes = sample_ref.shape[0]
    if n_cols == 0 or n_probes == 0:
        return sample_ref.index

    markers_per_class = max(1, feature_count // n_cols)

    col_mean = sample_ref.mean(axis=1)
    col_std = sample_ref.std(axis=1).replace(0, np.nan)
    z_scores = sample_ref.sub(col_mean, axis=0).div(col_std, axis=0).fillna(0)

    selected: set = set()
    for col in z_scores.columns:
        top_idx = z_scores[col].abs().nlargest(min(markers_per_class, n_probes)).index
        selected.update(top_idx.tolist())

    return sample_ref.index.intersection(list(selected))


def _select_features(sample_ref: pd.DataFrame,
                     feature_count: int,
                     feature_selection: str) -> pd.Index:
    """
    Pick the probes used for a single regression fit.

    Modes
    -----
    "marker"     Balanced z-score markers across cell types
                 (delegates to _select_marker_features). Default.
    "variance"   Highest cross-cell-type (row) variance probes.
    "hybrid"     ~50% markers + ~50% highest-variance, as a deduplicated union:
                 half the budget goes to balanced markers, the remainder is
                 filled with the highest-variance probes not already chosen, so
                 the union is ~feature_count. The marker share is configurable
                 inline, e.g. "hybrid:0.7" puts 70% of the budget into markers.

    Any unrecognized value falls back to "variance", matching the previous
    behaviour of the old `else` branch.

    Note: with a very small budget relative to the number of columns, the marker
    half can exceed its share (>=1 probe per cell type is always kept), leaving
    little room for the variance half. With the typical large feature_count and
    the small per-node child counts in the hierarchy this is not a concern.
    """
    n_probes = sample_ref.shape[0]
    if n_probes == 0:
        return sample_ref.index
    feature_count = min(int(feature_count), n_probes)

    mode = (feature_selection or "").strip().lower()

    if mode == "marker":
        return _select_marker_features(sample_ref, feature_count)

    if mode.startswith("hybrid"):
        marker_ratio = 0.5
        if ":" in mode:
            try:
                marker_ratio = float(mode.split(":", 1)[1])
            except ValueError:
                marker_ratio = 0.5
        marker_ratio = min(max(marker_ratio, 0.0), 1.0)

        marker_budget = int(round(feature_count * marker_ratio))
        marker_probes = (_select_marker_features(sample_ref, marker_budget)
                         if marker_budget > 0 else sample_ref.index[:0])

        remaining = feature_count - len(marker_probes)
        if remaining > 0:
            row_var = sample_ref.var(axis=1).drop(index=marker_probes, errors="ignore")
            variance_probes = row_var.nlargest(remaining).index
        else:
            variance_probes = sample_ref.index[:0]

        selected = marker_probes.union(variance_probes)
        return sample_ref.index.intersection(selected)

    # default / "variance"
    return sample_ref.var(axis=1).nlargest(feature_count).index


def _normal_decomposition(sample,
                          betas,
                          reference,
                          normal_fluids,
                          normal_tissues,
                          tumor_type=None,
                          n_features=4000,
                          method="huber",
                          logit_transform: bool = True,
                          feature_selection: str = "marker",
                          n_rounds: int = 1,
                          sparsity: float = 0.0,
                          noise_scale: float = 0.0,
                          feature_counts: Sequence[int] | None = None,
                          random_seed: int | None = None,
                          other_ref: pd.Series | None = None):
    """
    Process a single sample for decomposition with optional logit transformation.
    """
    # Extract sample betas and remove NaNs
    base_sample_betas = betas.loc[sample, :].dropna()

    if feature_counts is None:
        feature_counts = [n_features]
    feature_counts = [int(feature_count) for feature_count in feature_counts if int(feature_count) > 0]
    if len(feature_counts) == 0:
        raise ValueError("feature_counts must contain at least one positive integer.")

    rng = np.random.default_rng(random_seed)

    # Identify common probes
    common = reference.index.intersection(base_sample_betas.index).values
    if len(common) == 0:
        return sample, None

    # Eliminate other tumors
    if tumor_type == "NonmalignantBackground":
        tumor_type = None
    if tumor_type is not None:
        columns = normal_fluids + normal_tissues + [tumor_type]
    else:
        columns = normal_fluids + normal_tissues

    other_index = ["Other"] if other_ref is not None else []
    if tumor_type is not None:
        decom_sample = pd.Series(index=normal_fluids + ["Neuron"] + other_index, dtype=float)
    else:
        decom_sample = pd.Series(index=normal_fluids + ["Tumor"] + ["Neuron"] + other_index, dtype=float)
    decom_sample[:] = 0  # Initialize to avoid missing values

    round_results = []
    total_rounds = max(1, int(n_rounds))
    for _ in range(total_rounds):
        perturbed_betas = _perturb_sample_betas(base_sample_betas.loc[common], rng, sparsity=sparsity, noise_scale=noise_scale)
        if len(perturbed_betas) == 0:
            continue

        sample_betas = _apply_transform(perturbed_betas, logit_transform)

        for feature_count in feature_counts:
            sample_ref = reference.loc[perturbed_betas.index, columns].copy()
            if other_ref is not None:
                other_aligned = other_ref.reindex(perturbed_betas.index)
                other_aligned = other_aligned.fillna(other_aligned.median())
                sample_ref["Other"] = other_aligned.values
            # Transform the reference into the SAME space as the sample so the
            # linear fit is consistent (fixes the prior logit-sample /
            # beta-reference mismatch). Marker selection and the regression then
            # both operate in this space.
            sample_ref = _apply_transform(sample_ref, logit_transform)

            max_iter = 5  # Prevent infinite looping
            iteration = 0
            coef_v = np.zeros(sample_ref.shape[1])
            while np.any(coef_v <= 0.001) and sample_ref.shape[0] > 400 and iteration < max_iter:
                var_probes = _select_features(sample_ref, feature_count, feature_selection)
                X = sample_ref.loc[var_probes, :].values
                y = sample_betas.loc[var_probes].values
                if method == "huber":
                    coef_v, _ = huber_regress(X, y)
                elif method == "nnls":
                    coef_v, _ = nnls_regress(X, y)
                    iteration = max_iter  # No need to iterate for NNLS since it already enforces non-negativity
                else:
                    raise ValueError(f"Unknown method: {method}")

                if np.sum(coef_v > 0.001) == 0:
                    break

                sample_ref = sample_ref.loc[:, coef_v > 0.001]
                coef_v = coef_v[coef_v > 0.001]
                iteration += 1

            coefs = pd.Series(coef_v, index=sample_ref.columns)
            decom_round = decom_sample.copy()

            common_fluids = sample_ref.columns.intersection(normal_fluids)
            common_tissues = sample_ref.columns.intersection(normal_tissues)
            if tumor_type is not None:
                common_tumor = sample_ref.columns.intersection([tumor_type])

            if len(common_fluids) > 0:
                decom_round[common_fluids] = coefs.loc[common_fluids].values
            if len(common_tissues) > 0:
                decom_round["Neuron"] = coefs.loc[common_tissues].values.sum()
            if tumor_type is not None and len(common_tumor) > 0:
                decom_round["Tumor"] = coefs.loc[common_tumor].values.sum()
            if other_ref is not None and "Other" in coefs.index:
                decom_round["Other"] = coefs["Other"]

            round_results.append(decom_round)

    if len(round_results) == 0:
        return sample, None

    #averaged = pd.concat(round_results, axis=1).mean(axis=1)
    medians = pd.concat(round_results, axis=1).median(axis=1)
    return sample, _normalize_decomposition_result(medians)


def _tumor_decomposition(sample,
                         betas,
                         reference,
                         normals,
                         tumor_type,
                         n_features=4000,
                         min_purity: float = 0.05,
                         method: str = "huber",
                         logit_transform: bool = True,
                         feature_selection: str = "marker",
                         n_rounds: int = 1,
                         sparsity: float = 0.0,
                         noise_scale: float = 0.0,
                         feature_counts: Sequence[int] | None = None,
                         random_seed: int | None = None,
                         other_ref: pd.Series | None = None):
    base_sample_betas = betas.loc[sample, :].dropna()

    if tumor_type in normals or tumor_type in {"Control", "NonmalignantBackground"}:
        return sample, 0.0
    if tumor_type not in reference.columns:
        return sample, 0.0

    if feature_counts is None:
        feature_counts = [n_features]
    feature_counts = [int(feature_count) for feature_count in feature_counts if int(feature_count) > 0]
    if len(feature_counts) == 0:
        raise ValueError("feature_counts must contain at least one positive integer.")

    columns = [tumor_type] + list(normals)
    if other_ref is not None:
        columns = columns + ["Other"]

    common = reference.index.intersection(base_sample_betas.index)
    if len(common) == 0:
        return sample, 0.0

    rng = np.random.default_rng(random_seed)
    round_results = []
    total_rounds = max(1, int(n_rounds))
    for _ in range(total_rounds):
        perturbed_betas = _perturb_sample_betas(base_sample_betas.loc[common], rng, sparsity=sparsity, noise_scale=noise_scale)
        if len(perturbed_betas) == 0:
            continue

        sample_betas = _apply_transform(perturbed_betas, logit_transform)

        for feature_count in feature_counts:
            sample_ref = reference.loc[perturbed_betas.index, [tumor_type] + list(normals)].copy()
            if other_ref is not None:
                other_aligned = other_ref.reindex(perturbed_betas.index)
                other_aligned = other_aligned.fillna(other_aligned.median())
                sample_ref["Other"] = other_aligned.values
            # Transform the reference into the SAME space as the sample
            # (fixes the prior logit-sample / beta-reference mismatch).
            sample_ref = _apply_transform(sample_ref, logit_transform)

            max_iter = 5
            iteration = 0
            coef_v = np.zeros(sample_ref.shape[1])
            while np.any(coef_v <= 0.001) and sample_ref.shape[0] > 400 and iteration < max_iter:
                var_probes = _select_features(sample_ref, feature_count, feature_selection)

                X = sample_ref.loc[var_probes, :].values
                y = sample_betas.loc[var_probes].values
                if method == "huber":
                    coef_v, _ = huber_regress(X, y)
                elif method == "nnls":
                    coef_v, _ = nnls_regress(X, y)
                    iteration = max_iter
                else:
                    raise ValueError(f"Unknown method: {method}")

                if np.sum(coef_v > 0.001) == 0:
                    break

                sample_ref = sample_ref.loc[:, coef_v > 0.001]
                coef_v = coef_v[coef_v > 0.001]
                iteration += 1

            if sample_ref.shape[1] == 0:
                round_results.append(0.0)
                continue

            coefs = pd.Series(coef_v, index=sample_ref.columns)
            tumor_coef = float(max(coefs.get(tumor_type, 0.0), 0.0))
            round_results.append(tumor_coef if tumor_coef >= min_purity else 0.0)

    if len(round_results) == 0:
        return sample, 0.0

    return sample, float(np.median(round_results))


def normal_decomposition(betas: pd.DataFrame,
                         normal_fluids: list[str] = ['B', 'B-Mem',
                                            'Granulocytes', 'Monocytes',
                                            'NK', 'T-CD3', 'T-CD4',
                                            'T-CD8', 'T-CenMem-CD4',
                                            'T-Eff-CD8', 'T-EffMem-CD4',
                                            'T-EffMem-CD8', 'T-Naive-CD4',
                                            'T-Naive-CD8', 'Macrophages', "DendriticCells",
                                            'Vein-Endothel','Treg','Oligodendrocytes','Microglia',
                                            'Astrocyte'],
                         normal_tissues: list[str] = ['CONTR_CEBM', 'CONTR_HEMI'],
                         tumor_types: List[str] = None,
                         custom_classes: str = None,
                         n_jobs: int = -1,
                         ref_file: str = "BrainTumorDeconRef.parquet",
                         n_features=4000,
                         feature_counts: Sequence[int] | None = None,
                         logit_transform: bool = True,
                         feature_selection: str = "marker",
                         n_rounds: int = 1,
                         sparsity: float = 0.0,
                         noise_scale: float = 0.0,
                         random_seed: int | None = None,
                         method="huber",
                         include_other: bool = False,
                         verbose: bool = False) -> pd.DataFrame:
    """
    Decomposes methylation beta values into contributions from normal tissues.
    Uses Huber regression for robustness and parallel processing for speedup.
    Sample beta values can optionally be logit-transformed before regression.
    If include_other is True, an 'Other' category is added as the probe-wise
    median of all reference columns not in normal_fluids, normal_tissues, or
    the per-sample tumor type.
    """
    reference = _load_decomposition_reference(ref_file=ref_file, custom_classes=custom_classes)

    # Combine normal fluids and tissues
    normals = normal_fluids + normal_tissues
    if tumor_types is not None:
        normals = np.array(list(normals + list(set(pd.unique(tumor_types)) - set(["NonmalignantBackground"]))))
        tumor_types = {sample: tumor_types[i] for i, sample in enumerate(betas.index)}

    # Build the Other pseudo-reference before subsetting
    other_ref = None
    if include_other:
        used_set = set(normals.tolist() if isinstance(normals, np.ndarray) else normals)
        unused_cols = [c for c in reference.columns if c not in used_set]
        if len(unused_cols) > 0:
            other_ref = reference.loc[:, unused_cols].median(axis=1)

    reference = reference.loc[:, normals]

    # Parallelize processing of each sample
    if tumor_types is None:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_normal_decomposition)(
                sample,
                betas,
                reference,
                normal_fluids,
                normal_tissues,
                None,
                n_features,
                method,
                logit_transform,
                feature_selection,
                n_rounds,
                sparsity,
                noise_scale,
                feature_counts,
                None if random_seed is None else random_seed + idx,
                other_ref,
            )
            for idx, sample in enumerate(betas.index)
        )
    else:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_normal_decomposition)(
                sample,
                betas,
                reference,
                normal_fluids,
                normal_tissues,
                tumor_types[sample],
                n_features,
                method,
                logit_transform,
                feature_selection,
                n_rounds,
                sparsity,
                noise_scale,
                feature_counts,
                None if random_seed is None else random_seed + idx,
                other_ref,
            )
            for idx, sample in enumerate(betas.index)
        )

    # Construct results DataFrame
    other_cols = ["Other"] if include_other and other_ref is not None else []
    if tumor_types is not None:
        decom = pd.DataFrame(index=betas.index, columns=normal_fluids + ["Neuron"] + ["Tumor"] + other_cols, dtype=float)
    else:
        decom = pd.DataFrame(index=betas.index, columns=normal_fluids + ["Neuron"] + other_cols, dtype=float)
    for sample, values in results:
        if values is not None:
            decom.loc[sample] = values

    row_sums = decom.clip(lower=0).sum(axis=1)
    valid_rows = row_sums > 0
    decom.loc[valid_rows] = decom.loc[valid_rows].clip(lower=0).div(row_sums.loc[valid_rows], axis=0)

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
                        feature_counts: Sequence[int] | None = None,
                        logit_transform: bool = True,
                        feature_selection: str = "marker",
                        n_rounds: int = 1,
                        sparsity: float = 0.0,
                        noise_scale: float = 0.0,
                        random_seed: int | None = None,
                        n_jobs: int = -1,
                        min_purity: float = 0.05,
                        ref_file: str = "BrainTumorDeconRef.parquet",
                        custom_classes: str = None,
                        method: str = "huber",
                        include_other: bool = False,
                      verbose: bool = False):
    """
    """

    reference = _load_decomposition_reference(ref_file=ref_file, custom_classes=custom_classes)
    tumor_types = np.asarray(tumor_types)
    if len(tumor_types) != len(betas.index):
        raise ValueError("tumor_types must have the same length as betas.")

    other_ref = None
    if include_other:
        valid_tumor_types = {
            tumor_type for tumor_type in pd.unique(tumor_types)
            if tumor_type not in normals and tumor_type not in {"Control", "NonmalignantBackground"}
        }
        used_set = set(normals).union(valid_tumor_types)
        unused_cols = [column for column in reference.columns if column not in used_set]
        if len(unused_cols) > 0:
            other_ref = reference.loc[:, unused_cols].median(axis=1)

    required_columns = set(normals).union({
        tumor_type for tumor_type in pd.unique(tumor_types)
        if tumor_type not in {"Control", "NonmalignantBackground"}
    })
    available_columns = [column for column in reference.columns if column in required_columns]
    reference = reference.loc[:, available_columns]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_tumor_decomposition)(
            sample,
            betas,
            reference,
            normals,
            tumor_types[idx],
            n_features,
            min_purity,
            method,
            logit_transform,
            feature_selection,
            n_rounds,
            sparsity,
            noise_scale,
            feature_counts,
            None if random_seed is None else random_seed + idx,
            other_ref,
        )
        for idx, sample in enumerate(betas.index)
    )

    tumor_purity = pd.Series(0.0, index=betas.index, dtype=float)
    for sample, value in results:
        tumor_purity.loc[sample] = value

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




# =========================================================================== #
# Hierarchical (recursive / nested) deconvolution
#
# The hierarchy is an arbitrarily deep tree. Each internal node deconvolves its
# immediate children -- where a child is either a terminal reference column or a
# deeper subtree -- using an aggregated signature per child (median of the leaf
# columns beneath it). Fractions multiply down each root->leaf path, so:
#
#     fine[leaf] = prod_over_path( within_parent_proportion )
#
# This lets a real cell type sit at an intermediate level: e.g. circulating
# Monocytes are resolved at the same step as Macrophages and the DC subtree,
# and the DC fraction is then split into mDC / tolDC one level deeper.
#
# Tree grammar (write it however reads cleanly):
#   - str                : a leaf = a reference column name, e.g. "NK"
#   - list[str]          : an internal node whose children are leaves,
#                          e.g. ["B", "B-Mem"]
#   - dict{name: node}   : an internal node; keys are display names, values are
#                          sub-nodes. A value that is a str is a leaf whose
#                          OUTPUT column is that string, while the key is just
#                          the node's display name -- this is how "Monocyte"
#                          (node) maps to the "Monocytes" reference column.
#
# Both sample and every aggregated child signature pass through _apply_transform
# together, so there is no logit/beta mismatch at any level.
# =========================================================================== #

# Brain / CSF-oriented default. Edit freely; depth is unlimited.
# Microglia is a yolk-sac-derived CNS macrophage (not monocyte-derived), so it
# sits as its own Myeloid child rather than under the monocyte lineage; move it
# under "Glia" if you prefer a CNS-context grouping.
DEFAULT_HIERARCHY: Dict[str, object] = {
    "Lymphoid": {
        "T-CD4": ["T-CD4-Th", "T-CenMem-CD4", "T-EffMem-CD4", "Treg"],
        "T-CD8": ["T-Eff-CD8", "T-EffMem-CD8"],
        "B":     ["B", "B-Mem"],
        "NK":    "NK",
    },
    "Myeloid": {
        "Mononuclear-Phagocyte": {
            "Monocyte":   "Monocytes",
            "Macrophage": "Macrophages",
            "DC":         {"mDC": "mDC", "tolDC": "tolDC"},
        },
        "Granulocyte": ["Neutrophils", "Granulocytes"],
        "Microglia":   "Microglia",
    },
    "Glia": {
        "Oligodendrocyte": "Oligodendrocytes",
        "Astrocyte":       "Astrocyte",
    },
    "Stroma": {
        "Fibroblast":     "Fibroblasts",
        "Vein-Endothel":  "Vein-Endothel",
    },
}


def _children_of(node) -> Optional[Dict[str, object]]:
    """Return {child_name: child_node} for an internal node, or None for a leaf."""
    if isinstance(node, str):
        return None
    if isinstance(node, dict):
        return dict(node)
    if isinstance(node, (list, tuple)):
        for item in node:
            if not isinstance(item, str):
                raise TypeError(
                    "List nodes may contain only cell-type name strings; "
                    "use a dict for deeper nesting."
                )
        return {item: item for item in node}
    raise TypeError(f"Unsupported hierarchy node type: {type(node)!r}")


def _iter_leaves(node) -> List[str]:
    """All reference column names beneath a node (preorder)."""
    children = _children_of(node)
    if children is None:
        return [node]
    leaves: List[str] = []
    for child in children.values():
        leaves.extend(_iter_leaves(child))
    return leaves


def _unique(seq: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_broad_signature(reference: pd.DataFrame,
                          hierarchy: Dict[str, object],
                          agg: str = "median") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Aggregate the reference into one signature per TOP-LEVEL node (the coarsest
    level of the tree). Returns (broad_ref [probes x top-level node], leaf->node).
    Mostly a convenience/inspection helper; the recursive solver does its own
    per-node aggregation internally.
    """
    leaf_to_node: Dict[str, str] = {}
    cols: Dict[str, pd.Series] = {}
    for name, node in hierarchy.items():
        leaves = [l for l in _iter_leaves(node) if l in reference.columns]
        if len(leaves) == 0:
            continue
        for l in leaves:
            leaf_to_node[l] = name
        block = reference[leaves]
        cols[name] = block.median(axis=1) if agg == "median" else block.mean(axis=1)
    if len(cols) == 0:
        raise ValueError("No reference columns matched the hierarchy.")
    broad_ref = pd.DataFrame(cols)
    return broad_ref, leaf_to_node


def _decompose_node(node,
                    parent_fraction: float,
                    sample_betas: pd.Series,
                    ref_block: pd.DataFrame,
                    feature_count: int,
                    method: str,
                    feature_selection: str,
                    logit_transform: bool,
                    min_node_fraction: float,
                    agg: str,
                    fine_out: pd.Series) -> None:
    """
    Recursively distribute ``parent_fraction`` across the leaves under ``node``.

    ``sample_betas`` is already in regression space; ``ref_block`` is the
    probe-subset reference in BETA space (each child signature is aggregated
    then transformed here, so sample and reference always match).
    """
    children = _children_of(node)
    if children is None:                       # leaf: ``node`` is a column name
        if node in fine_out.index:
            fine_out[node] += parent_fraction
        return

    # Keep only children that have at least one present reference column.
    usable = {}
    for cname, cnode in children.items():
        present = [l for l in _iter_leaves(cnode) if l in ref_block.columns]
        if present:
            usable[cname] = (cnode, present)
    if len(usable) == 0:
        return

    names = list(usable.keys())
    if len(names) == 1:                        # nothing to split here
        cnode = usable[names[0]][0]
        _decompose_node(cnode, parent_fraction, sample_betas, ref_block, feature_count,
                        method, feature_selection, logit_transform, min_node_fraction,
                        agg, fine_out)
        return

    # Build one aggregated signature column per child (beta space), then transform.
    sig = pd.DataFrame(index=ref_block.index, columns=names, dtype=float)
    for cname, (_, present) in usable.items():
        if len(present) > 1:
            sig[cname] = (ref_block[present].median(axis=1) if agg == "median"
                          else ref_block[present].mean(axis=1)).values
        else:
            sig[cname] = ref_block[present[0]].values
    sig = _apply_transform(sig, logit_transform)

    p = _fit_compartment(sig, sample_betas, feature_count, method, feature_selection)
    p = _normalize_decomposition_result(p)
    if min_node_fraction > 0:
        p[p < min_node_fraction] = 0.0
        p = _normalize_decomposition_result(p)
    if p.sum() <= 0:                           # fit failed -> split evenly
        p = pd.Series(1.0 / len(names), index=names)

    for cname, (cnode, _) in usable.items():
        frac = parent_fraction * float(p.get(cname, 0.0))
        if frac <= 0:
            continue
        _decompose_node(cnode, frac, sample_betas, ref_block, feature_count, method,
                        feature_selection, logit_transform, min_node_fraction, agg, fine_out)


def _fit_compartment(ref_block: pd.DataFrame,
                     sample_betas: pd.Series,
                     feature_count: int,
                     method: str,
                     feature_selection: str,
                     min_probes: int = 400,
                     max_iter: int = 5,
                     prune_threshold: float = 0.001) -> pd.Series:
    """
    One iterative marker-select -> robust-regress -> prune fit (mirrors the loop
    in _normal_decomposition). ``ref_block`` and ``sample_betas`` must already be
    in the SAME space. Returns a non-negative coef Series over ref_block.columns.
    """
    sample_ref = ref_block
    coef_v = np.zeros(sample_ref.shape[1])
    iteration = 0
    while np.any(coef_v <= prune_threshold) and sample_ref.shape[0] > min_probes and iteration < max_iter:
        var_probes = _select_features(sample_ref, feature_count, feature_selection)

        X = sample_ref.loc[var_probes, :].values
        y = sample_betas.loc[var_probes].values
        if method == "huber":
            coef_v, _ = huber_regress(X, y)
        elif method == "nnls":
            coef_v, _ = nnls_regress(X, y)
            iteration = max_iter
        else:
            raise ValueError(f"Unknown method: {method}")

        if np.sum(coef_v > prune_threshold) == 0:
            break
        sample_ref = sample_ref.loc[:, coef_v > prune_threshold]
        coef_v = coef_v[coef_v > prune_threshold]
        iteration += 1

    coefs = pd.Series(coef_v, index=sample_ref.columns)
    return coefs.reindex(ref_block.columns).fillna(0.0)


def _hierarchical_decomposition(sample,
                                betas,
                                reference,
                                hierarchy,
                                n_features=4000,
                                method="huber",
                                logit_transform=True,
                                feature_selection="marker",
                                n_rounds=1,
                                sparsity=0.0,
                                noise_scale=0.0,
                                feature_counts: Optional[Sequence[int]] = None,
                                random_seed: Optional[int] = None,
                                min_node_fraction: float = 0.0,
                                agg: str = "median"):
    """Per-sample worker. Returns (sample, fine_fractions) summing to 1 over leaves."""
    base_sample_betas = betas.loc[sample, :].dropna()

    if feature_counts is None:
        feature_counts = [n_features]
    feature_counts = [int(fc) for fc in feature_counts if int(fc) > 0]
    if len(feature_counts) == 0:
        raise ValueError("feature_counts must contain at least one positive integer.")

    leaves = [l for l in _unique(_iter_leaves(hierarchy)) if l in reference.columns]
    common = reference.index.intersection(base_sample_betas.index)
    if len(common) == 0 or len(leaves) == 0:
        return sample, None

    rng = np.random.default_rng(random_seed)
    rounds = []
    for _ in range(max(1, int(n_rounds))):
        perturbed = _perturb_sample_betas(base_sample_betas.loc[common], rng,
                                          sparsity=sparsity, noise_scale=noise_scale)
        if len(perturbed) == 0:
            continue
        probes = perturbed.index
        y_sample = _apply_transform(perturbed, logit_transform)
        ref_block = reference.loc[probes]      # beta space; transformed per node

        for feature_count in feature_counts:
            fine = pd.Series(0.0, index=leaves, dtype=float)
            _decompose_node(hierarchy, 1.0, y_sample, ref_block, feature_count, method,
                            feature_selection, logit_transform, min_node_fraction, agg, fine)
            rounds.append(_normalize_decomposition_result(fine))

    if len(rounds) == 0:
        return sample, None
    return sample, _normalize_decomposition_result(pd.concat(rounds, axis=1).median(axis=1))


def node_fractions(fine_df: pd.DataFrame,
                   hierarchy: Dict[str, object] = None) -> pd.DataFrame:
    """
    Roll leaf fractions up to every internal node (samples x node). Each node's
    value is the summed fraction of its descendant leaves -- e.g. a single
    "Monocyte"/"Macrophage"/"DC" or "T-CD4" column. Node display names are
    assumed unique across the tree (true for DEFAULT_HIERARCHY).
    """
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY

    out: Dict[str, pd.Series] = {}

    def _walk(name, node):
        children = _children_of(node)
        if children is None:
            return
        leaves = [l for l in _iter_leaves(node) if l in fine_df.columns]
        if leaves:
            out[name] = fine_df[leaves].sum(axis=1)
        for cname, cnode in children.items():
            _walk(cname, cnode)

    for top_name, top_node in hierarchy.items():
        _walk(top_name, top_node)
    return pd.DataFrame(out, index=fine_df.index)


def hierarchical_decomposition(betas: pd.DataFrame,
                               reference: Optional[pd.DataFrame] = None,
                               hierarchy: Optional[Dict[str, object]] = None,
                               ref_file: str = "BrainTumorDeconRef.parquet",
                               custom_classes: Optional[str] = None,
                               n_features: int = 4000,
                               feature_counts: Optional[Sequence[int]] = None,
                               method: str = "huber",
                               logit_transform: bool = True,
                               feature_selection: str = "marker",
                               n_rounds: int = 1,
                               sparsity: float = 0.0,
                               noise_scale: float = 0.0,
                               random_seed: Optional[int] = None,
                               n_jobs: int = -1,
                               agg: str = "median",
                               min_node_fraction: float = 0.0,
                               return_nodes: bool = False,
                               verbose: bool = False):
    """
    Recursive hierarchical reference-based deconvolution over an arbitrarily deep
    cell-type tree. Sample and every aggregated child signature are transformed
    together, so there is no logit/beta mismatch at any level.

    Parameters
    ----------
    betas : samples x probes (beta values)
    reference : probes x cell-types (beta values). If None, loaded via
        ``ref_file`` / ``custom_classes`` like normal_decomposition.
    hierarchy : nested tree (see module grammar); defaults to DEFAULT_HIERARCHY.
    min_node_fraction : within each node, zero out children below this proportion
        before recursing (then renormalize).
    return_nodes : if True, also return a samples x internal-node DataFrame of
        rolled-up fractions (e.g. Monocyte, T-CD4, Myeloid totals).

    Returns
    -------
    fine_df  : samples x leaf cell-types (fractions, rows sum to 1)
    node_df  : samples x internal nodes (only if return_nodes=True)
    """
    if hierarchy is None:
        hierarchy = DEFAULT_HIERARCHY
    if reference is None:
        reference = _load_decomposition_reference(ref_file=ref_file, custom_classes=custom_classes)

    all_leaves = _unique(_iter_leaves(hierarchy))
    leaves = [l for l in all_leaves if l in reference.columns]
    missing = [l for l in all_leaves if l not in reference.columns]
    if verbose and missing:
        print("Hierarchy leaves absent from reference (ignored):", missing, flush=True)
    if len(leaves) == 0:
        raise ValueError("None of the hierarchy leaves are present in the reference.")

    reference = reference.loc[:, leaves]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_hierarchical_decomposition)(
            sample,
            betas,
            reference,
            hierarchy,
            n_features,
            method,
            logit_transform,
            feature_selection,
            n_rounds,
            sparsity,
            noise_scale,
            feature_counts,
            None if random_seed is None else random_seed + idx,
            min_node_fraction,
            agg,
        )
        for idx, sample in enumerate(betas.index)
    )

    fine_df = pd.DataFrame(0.0, index=betas.index, columns=leaves, dtype=float)
    for sample, fine in results:
        if fine is not None:
            fine_df.loc[sample, fine.index] = fine.values

    if return_nodes:
        return fine_df, node_fractions(fine_df, hierarchy)
    return fine_df
