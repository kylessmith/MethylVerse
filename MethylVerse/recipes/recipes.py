import numpy as np
import pandas as pd
from typing import List
import os
import ngsfragments as ngs

# Local imports
from ..data.import_data import get_data_file
from ..data.download_data import download_MPACT
from ..tools.classifiers.MPACT.MPACT_classifier_torch import MPACT_classifier
from ..tools.decomposition.reference_decom import remove_normal_csf, tumor_decomposition
from ..tools.imputation.imputeEPIC import impute_epic, EPICimputer
from ..tools.cnvs.seq_cnv_calling import sequencing_call_cnvs
from ..tools.cnvs.idat_cnv_calling import microarray_call_cnvs
from ..core.methyl_core import read_sequencing_methylation, RawArray
from ..core.utilities import position_to_cpg


def MPACT_process_raw(input_data: List[str],
                        impute: bool = False,
                        regress: bool = False,
                        probability_threshold: float = 0.7,
                        max_contamination_fraction: float = 0.5,
                        call_cnvs: bool = False,
                        output_cnv: str = None,
                        verbose: bool = False) -> pd.DataFrame:
    """
    """
    # Check data downloaded
    download_MPACT()

    # Initialize
    is_sequencing = True
    if len(input_data) > 1:
        is_sequencing = False

    # Read methylation
    if verbose: print("Reading methylation data", flush=True)
    if is_sequencing:
        sample_name = os.path.basename(input_data[0]).split(".")[0]
        data = read_sequencing_methylation(input_data[0])
        cpgs = position_to_cpg(data, genome_version="hg38")
        data = data.df.loc[cpgs!='', :]
        data.index = cpgs[cpgs!='']
        data = data.T
        data.index = [sample_name]
        data = data.loc[:,~data.columns.duplicated()]
    else:
        if "Red" in input_data[0]:
            sample_name = os.path.basename(input_data[0]).split("_Red")[0]
        else:
            sample_name = os.path.basename(input_data[0]).split("_Grn")[0]
        raw_array = RawArray(input_data, genome_version="hg38", n_jobs=1, verbose=False)
        data = raw_array.get_betas()
        data.index = [sample_name]

    # Impute
    if verbose: print("Imputing data", flush=True)
    if impute:
        data = impute_epic(data, method="regress")
    else:
        imputer = EPICimputer()
        # Find missing features
        missing = imputer.features[~pd.Series(imputer.features).isin(data.columns)]
        # Set missing features to nan
        null_df = pd.DataFrame(np.nan, index=data.index, columns=missing)
        data = pd.concat([data, null_df], axis=1)
        data = data.loc[:,imputer.features]

    # Regress
    if regress:
        if verbose: print("Regressing data", flush=True)
        transformed_data = remove_normal_csf(data, max_fraction=max_contamination_fraction)

    # Classify
    # Get classifier files
    if verbose: print("Classifying data", flush=True)
    onnx_directory = os.path.dirname(get_data_file("MPACT_D1_v1.pth"))
    classifier = MPACT_classifier(onnx_directory+"/")
    if regress:
        transformed_data = (transformed_data * 2) - 1
    else:
        transformed_data = (data * 2) - 1
    transformed_data.fillna(0, inplace=True)
    predictions = classifier.predict(transformed_data, transform=False)
    prediction = predictions[0][0]
    probability = predictions[1].loc[:,prediction].values[0]

    # Deconvolve
    if verbose: print("Deconvolving data", flush=True)
    if probability > probability_threshold and prediction != "NonmalignantBackground":
        decon = tumor_decomposition(data, [prediction], verbose=False)
        purity = decon.values[0]
        if purity == 0 and prediction != "NonmalignantBackground":
            prediction = "NonmalignantBackground"
            purity = 1.0
            probability = 1.0
    elif probability < probability_threshold and prediction == "NonmalignantBackground":
        secondary_prediction = predictions[1].columns.values[predictions[1].values.argsort()[0][-2]]
        decon = tumor_decomposition(data, [secondary_prediction], verbose=False)
        # GCT corresponds to over regressed samples meaning no tumor
        if decon.values[0] == 0 or secondary_prediction == "GCT":
            purity = 0.0
            probability = 1.0
    else:
        purity = 0.0
    
    # Call CNVs
    if call_cnvs:
        if verbose: print("Calling CNVs", flush=True)
        if is_sequencing:
            data = read_sequencing_methylation(input_data[0], read_cov=True)
            data.df.columns = [sample_name]
            cnvs = sequencing_call_cnvs(data)

            # Plot CNVs
            if output_cnv is not None:
                ngs.plot.plot_cnv(cnvs.pf,
                                obs = sample_name,
                                show = False,
                                save = output_cnv,
                                plot_max = 5,
                                plot_min = -3)
            else:
                ngs.plot.plot_cnv(cnvs.pf,
                                    obs = sample_name,
                                    show = False,
                                    save = sample_name+"_cnvs.pdf",
                                    plot_max = 5,
                                    plot_min = -3)
        else:
            cnvs = microarray_call_cnvs(raw_array)

            # Plot CNVs
            if output_cnv is not None:
                ngs.plot.plot_cnv(cnvs.pf,
                                obs = sample_name,
                                show = False,
                                save = output_cnv,
                                plot_max = 2,
                                plot_min = -2)
            else:
                ngs.plot.plot_cnv(cnvs.pf,
                                    obs = sample_name,
                                    show = False,
                                    save = sample_name+"_cnvs.pdf",
                                    plot_max = 2,
                                    plot_min = -2)
        if verbose: print("Output plots saved:", sample_name+"_cnvs.pdf", flush=True)

        # Construct results
        if verbose: print("Constructing results", flush=True)
        cnv_purity = cnvs.pf.anno.loc[sample_name, "purity"]
        cnv_ploidy = cnvs.pf.anno.loc[sample_name, "ploidy"]
        results = pd.DataFrame([prediction, probability, purity, cnv_purity, cnv_ploidy], columns = [sample_name],
                            index=["MPACT_Prediction", "MPACT_Probability", "MPACT_Purity", "CNV_Purity", "CNV_Ploidy"]).T
    else:
        results = pd.DataFrame([prediction, probability, purity], columns = [sample_name],
                            index=["MPACT_Prediction", "MPACT_Probability", "MPACT_Purity"]).T
    
    return results