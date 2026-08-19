import h5py
from projectframe import ProjectFrame
from intervalframe import IntervalFrame
import ngsfragments as ngs
from ngsfragments.plot.plot_plt import plot_cnv
import os
import glob
import numpy as np
import pandas as pd

# Local imports
from ..data.import_data import get_data_file
from ..core.utilities import predict_data_type, position_to_cpg
from ..core.microarray.RawArray import RawArray
from ..core.sequencing import read_methyldackel
from ..recipes.recipes import MPACT_process_raw
from ..core.methyl_core import read_sequencing_methylation, read_methylation
from ..tools.regions.methyl_segment import region_classification
from ..tools.cnvs.seq_cnv_calling import sequencing_call_cnvs
from ..tools.cnvs.idat_cnv_calling import microarray_call_cnvs
from ..tools.decomposition.reference_decom import normal_decomposition
from ..tools.imputation.imputeEPIC import intervals_to_probes


def NonmalignantDecomposition(args):
    """
    """

    # Read input data
    betas = read_sequencing_methylation(args.input_data)
    
    # Format values
    betas = intervals_to_probes(betas, impute=args.impute)

    # Decompose samples
    results = normal_decomposition(betas.T, tumor_types=[args.tumor_type], n_jobs=1, verbose=args.verbose)

    # Remove and correct for tumor
    if "Tumor" in results.columns.values:
        results = results.drop(columns=["Tumor"])
        # Renormalize columns to sum to 1
        results = results.div(results.sum(axis=1), axis=0)

    # Write output
    results.to_csv(args.prefix + "_nonmalignant_fractions.txt", header=True, index=True, sep="\t")


def MPACT_process_single(args):
    """
    """

    # Determine data type
    check_idats = glob.glob(args.input_data + "*.idat")
    if len(check_idats) > 0:
        input_data = check_idats
    else:
        input_data = [args.input_data]
    classifications = MPACT_process_raw(input_data = input_data,
                                        impute = args.impute,
                                        regress = args.regress,
                                        probability_threshold = args.probability_threshold,
                                        max_contamination_fraction = args.max_contamination_fraction,
                                        call_cnvs = args.call_cnvs,
                                        verbose = args.verbose)
    classifications.to_csv(args.out, header=True, index=True, sep="\t")


def MethylSegment_process_single(args):
    """
    Process a single MethylSegment job.
    """

    # Read input data
    betas = read_sequencing_methylation(args.input_data)

    # Call methylation segments
    sample_name = betas.df.columns.values[0]
    pmd_results, umr_results, lmr_results, hmr_results = region_classification(betas, sample_name=sample_name, verbose=args.verbose)

    # Write results
    results = {
        "PMDs": pmd_results,
        "UMRs": umr_results,
        "LMRs": lmr_results,
        "HMRs": hmr_results
    }
    for key in results:
        # Process each interval frame
        iframe = results[key]
        if iframe is None or iframe.shape[0] == 0:
            continue
        if args.verbose:
            print("Found", iframe.shape[0], "regions classified as", key)
        df = iframe.df
        df["chrom"] = iframe.index.labels
        df["start"] = iframe.index.starts
        df["end"] = iframe.index.ends
        df = df.loc[:,["chrom", "start", "end", "mean"]]
        df.to_csv(sample_name + "_" + key + ".bedGraph", header=False, index=False, sep="\t")

    return None


def CNV_process_single(args):
    """
    Process a single CNV job.
    """

    # Determine data type
    check_idats = glob.glob(args.input_data + "*.idat")
    if len(check_idats) > 0:
        input_data = check_idats
    else:
        input_data = [args.input_data]

    # Initialize
    is_sequencing = True
    if len(input_data) > 1:
        is_sequencing = False

    if is_sequencing:
        if args.rpca:
            print("RPCA normalization is not available for sequencing data. Proceeding without RPCA.", flush=True)
        # Set parameters
        if args.n_probes == 0:
            n_probes = None
        else:
            n_probes = args.n_probes
        if args.n_probes_hmm == 0:
            n_probes_hmm = None
        else:
            n_probes_hmm = args.n_probes_hmm
        if args.n_mads == 0.0:
            n_mads = 1.48
        else:
            n_mads = args.n_mads
        if args.bcp_cutoff == 0.0:
            bcp_cutoff = 0.3
        else:
            bcp_cutoff = args.bcp_cutoff

        sample_name = os.path.basename(input_data[0]).split(".")[0]
        data = read_sequencing_methylation(args.input_data, read_cov=True)
        data.df.columns = [sample_name]
        cnvs = sequencing_call_cnvs(data,
                                    genome_version = args.genome_version,
                                    n_probes = n_probes,
                                    n_probes_hmm = n_probes_hmm,
                                    zscore = args.zscore,
                                    n_mads = n_mads,
                                    bcp_cutoff = bcp_cutoff)

        # Plot CNVs
        ngs.plot.plot_cnv(cnvs.pf,
                            obs = sample_name,
                            show = False,
                            save = sample_name+"_cnvs.pdf",
                            plot_max = 5,
                            plot_min = -3)
    else:
        # Set parameters
        if args.n_probes == 0:
            n_probes = 10
        else:
            n_probes = args.n_probes
        if args.n_probes_hmm == 0:
            n_probes_hmm = 15
        else:
            n_probes_hmm = args.n_probes_hmm
        if args.n_mads == 0.0:
            n_mads = 0.75
        else:
            n_mads = args.n_mads
        if args.bcp_cutoff == 0.0:
            bcp_cutoff = 0.2
        else:
            bcp_cutoff = args.bcp_cutoff

        if "Red" in input_data[0]:
            sample_name = os.path.basename(input_data[0]).split("_Red")[0]
        else:
            sample_name = os.path.basename(input_data[0]).split("_Grn")[0]
        raw_array = RawArray(input_data, genome_version="hg38", n_jobs=1, verbose=False)
        cnvs = microarray_call_cnvs(raw_array,
                                    genome_version = args.genome_version,
                                    n_probes = n_probes,
                                    n_probes_hmm = n_probes_hmm,
                                    zscore = args.zscore,
                                    n_mads = n_mads,
                                    bcp_cutoff = bcp_cutoff,
                                    rpca_normal = args.rpca)

        # Plot CNVs
        ngs.plot.plot_cnv(cnvs.pf,
                            obs = sample_name,
                            show = False,
                            save = sample_name+"_cnvs.pdf",
                            plot_max = 2,
                            plot_min = -2)
    if args.verbose: print("Output plots saved:", sample_name+"_cnvs.pdf", flush=True)

    # Write seg file
    if args.segs:
        # Write seg file
        if args.segs:
            seg_fn = sample_name + ".seg"
            ngs.segment.cnv_utilities.write_seg_file(cnvs.pf,
                                                        seg_fn,
                                                        sample_name)
            
    # Write annotations
    if args.anno_file:
        cnvs.pf.anno.engine.df.to_csv(sample_name+"_metrics.txt", header=True, index=True, sep="\t")

    # Write seg annotations
    if args.anno_segs:
        df = cnvs.pf.obs_intervals[sample_name]["cnv_segments"].df
        df.loc[:,"chrom"] = cnvs.pf.obs_intervals[sample_name]["cnv_segments"].index.labels
        df.loc[:,"start"] = cnvs.pf.obs_intervals[sample_name]["cnv_segments"].index.starts
        df.loc[:,"end"] = cnvs.pf.obs_intervals[sample_name]["cnv_segments"].index.ends
        df.loc[:,"sample"] = sample_name
        df = df.loc[:,['sample', 'chrom', 'start', 'end', 'copy_number', 'event', 'subclone_status',
                    'logR_Copy_Number', 'Corrected_Copy_Number', 'Corrected_Call', 'median']]
        df.to_csv(sample_name+"_seg_annotations.seg", header=True, index=False, sep="\t")

        # Drop columns
        df.drop(columns=["chrom", "start", "end"], inplace=True)

    # Write bins
    if args.bins_file:
        df = cnvs.pf.intervals["cnv_bins"].df
        df.loc[:,"chrom"] = cnvs.pf.intervals["cnv_bins"].index.labels
        df.loc[:,"start"] = cnvs.pf.intervals["cnv_bins"].index.starts
        df.loc[:,"end"] = cnvs.pf.intervals["cnv_bins"].index.ends
        df.loc[:,"ratio"] = df.loc[:,sample_name].values
        if args.zscore:
            dfz = cnvs.pf.intervals["cnv_zscore_bins"].df
            df.loc[:,"zscore"] = dfz.loc[:,sample_name].values
            df = df.loc[:,['chrom', 'start', 'end', 'ratio', 'zscore']]
        else:
            df = df.loc[:,['chrom', 'start', 'end', 'ratio']]
        df.to_csv(sample_name+"_bins.txt", header=True, index=False, sep="\t")

        # Drop columns
        if args.zscore:
            df.drop(columns=["chrom", "start", "end", "ratio", "zscore"], inplace=True)
        else:
            df.drop(columns=["chrom", "start", "end", "ratio"], inplace=True)

    return None


def read_dir_methylation(args):
    """
    Read methylation data from a directory.
    """

    # Read methylation data
    methylation = read_methylation(args.input_data,
                                    genome_version=args.genome_version,
                                    n_jobs=args.n_jobs,
                                    verbose=args.verbose)
    # Convert to probes if specified
    if args.to_probes:
        cpgs = position_to_cpg(methylation, genome_version=args.genome_version)
        methylation = methylation.df
        methylation.index = cpgs
        methylation = methylation.loc[cpgs!='', :]
        methylation = methylation.loc[methylation.index.duplicated()==False, :]
    
    # Write output
    methylation.to_parquet(args.prefix + ".parquet", index=True)
    
    return None