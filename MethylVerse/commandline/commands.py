import h5py
from projectframe import ProjectFrame
from intervalframe import IntervalFrame
import ngsfragments as ngs
from ngsfragments.plot.plot_plt import plot_cnv
import os
import glob
import numpy as np

# Local imports
from ..data.import_data import get_data_file
from ..core.utilities import predict_data_type
from ..core.microarray.RawArray import RawArray
from ..core.sequencing import read_methyldackel
from ..recipes.recipes import MPACT_process_raw


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


def microarray_call_cnvs(array_info, normal_450k_references, normal_EPIC_references, args):
    """
    """

    # Read copy number values
    cn = array_info.get_cn()

    # Filter normal
    if "450k" in array_info.array_type[0]:
        cn = cn.exact_match(normal_450k_references)
        normal_references = normal_450k_references.exact_match(cn)
    elif "EPIC" in array_info.array_type[0]:
        cn = cn.exact_match(normal_EPIC_references)
        normal_references = normal_EPIC_references.exact_match(cn)
    
    # Call CNVs
    cnvs = ngs.segment.CNVcaller(genome_version = args.genome,
                                        scStates = None,
                                        n_per_bin = args.n_probes,
                                        n_per_bin_hmm = args.n_probes_hmm)
    cnvs.predict_cnvs(data = cn,
                        normal_data = normal_references)
    if args.zscore:
        cnvs.calculate_zscore()

    # Plot CNVs
    if args.plot:
        sample = cn.df.columns.values[0]
        ngs.plot.plot_cnv(cnvs.pf,
                            obs = sample,
                            show = False,
                            save = sample+"_cnvs.pdf",
                            plot_max = 2,
                            plot_min = -2)
    # Write seg file
    if args.segs:
        sample = cn.df.columns.values[0]
        # Write seg file
        if args.segs:
            seg_fn = sample + ".seg"
            ngs.segment.cnv_utilities.write_seg_file(cnvs.pf,
                                                        seg_fn,
                                                        sample)
            
    # Write annotations
    if args.anno_file:
        cnvs.pf.anno.engine.df.to_csv(sample+"_metrics.txt", header=True, index=True, sep="\t")

    # Write seg annotations
    if args.anno_segs:
        sample = cn.df.columns.values[0]
        df = cnvs.pf.obs_intervals[sample]["cnv_segments"].df
        df.loc[:,"chrom"] = cnvs.pf.obs_intervals[sample]["cnv_segments"].index.labels
        df.loc[:,"start"] = cnvs.pf.obs_intervals[sample]["cnv_segments"].index.starts
        df.loc[:,"end"] = cnvs.pf.obs_intervals[sample]["cnv_segments"].index.ends
        df.loc[:,"sample"] = sample
        df = df.loc[:,['sample', 'chrom', 'start', 'end', 'copy_number', 'event', 'subclone_status',
                    'logR_Copy_Number', 'Corrected_Copy_Number', 'Corrected_Call', 'median']]
        df.to_csv(sample+"_seg_annotations.seg", header=True, index=False, sep="\t")

        # Drop columns
        df.drop(columns=["chrom", "start", "end"], inplace=True)

    # Write bins
    if args.bins_file:
        sample = cn.df.columns.values[0]
        df = cnvs.pf.intervals["cnv_bins"].df
        df.loc[:,"chrom"] = cnvs.pf.intervals["cnv_bins"].index.labels
        df.loc[:,"start"] = cnvs.pf.intervals["cnv_bins"].index.starts
        df.loc[:,"end"] = cnvs.pf.intervals["cnv_bins"].index.ends
        df.loc[:,"ratio"] = df.loc[:,sample].values
        if args.zscore:
            dfz = cnvs.pf.intervals["cnv_zscore_bins"].df
            df.loc[:,"zscore"] = dfz.loc[:,sample].values
            df = df.loc[:,['chrom', 'start', 'end', 'ratio', 'zscore']]
        else:
            df = df.loc[:,['chrom', 'start', 'end', 'ratio']]
        df.to_csv(sample+"_bins.txt", header=True, index=False, sep="\t")

        # Drop columns
        if args.zscore:
            df.drop(columns=["chrom", "start", "end", "ratio", "zscore"], inplace=True)
        else:
            df.drop(columns=["chrom", "start", "end", "ratio"], inplace=True)

    return None


def sequencing_call_cnvs(bedgraph, args):
    """
    """

    # Read coverage data
    cov = read_methyldackel(np.array([bedgraph]), read_coverage=True)

    # Call CNVs
    cnvs = ngs.segment.CNVcaller(genome_version = args.genome,
                                        scStates = None,
                                        n_per_bin = args.n_per_bin,
                                        n_per_bin_hmm = args.n_per_bin_hmm)
    cnvs.predict_cnvs(data = cov)
    if args.zscore:
        cnvs.calculate_zscore()

    # Plot CNVs
    if args.plot:
        sample = cov.df.columns.values[0]
        ngs.plot.plot_cnv(cnvs.pf,
                        obs = sample,
                        show = False,
                        save = sample+"_cnvs.pdf",
                        plot_max = 5,
                        plot_min = -3)
    # Write seg file
    if args.segs:
        sample = cov.df.columns.values[0]
        # Write seg file
        if args.segs:
            seg_fn = sample + ".seg"
            ngs.segment.cnv_utilities.write_seg_file(cnvs.pf,
                                                        seg_fn,
                                                        sample)
            
    # Write annotations
    if args.anno_file:
        cnvs.pf.anno.engine.df.to_csv(sample+"_metrics.txt", header=True, index=True, sep="\t")

    # Write seg annotations
    if args.anno_segs:
        sample = cov.df.columns.values[0]
        df = cnvs.pf.obs_intervals[sample]["cnv_segments"].df
        df.loc[:,"chrom"] = cnvs.pf.obs_intervals[sample]["cnv_segments"].index.labels
        df.loc[:,"start"] = cnvs.pf.obs_intervals[sample]["cnv_segments"].index.starts
        df.loc[:,"end"] = cnvs.pf.obs_intervals[sample]["cnv_segments"].index.ends
        df.loc[:,"sample"] = sample
        df.loc[:,"n_bins"] = ((df.loc[:,"end"].values - df.loc[:,"start"].values) / args.bin_size).astype(int)
        df = df.loc[:,['sample', 'chrom', 'start', 'end', 'copy_number', 'event', 'subclone_status',
                    'logR_Copy_Number', 'Corrected_Copy_Number', 'Corrected_Call', 'var', 'n_bins', 'median']]
        df.to_csv(sample+"_seg_annotations.seg", header=True, index=False, sep="\t")

        # Drop columns
        df.drop(columns=["chrom", "start", "end"], inplace=True)

    # Write bins
    if args.bins_file:
        sample = cov.df.columns.values[0]
        df = cnvs.pf.intervals["cnv_bins"].df
        df.loc[:,"chrom"] = cnvs.pf.intervals["cnv_bins"].index.labels
        df.loc[:,"start"] = cnvs.pf.intervals["cnv_bins"].index.starts
        df.loc[:,"end"] = cnvs.pf.intervals["cnv_bins"].index.ends
        df.loc[:,"ratio"] = df.loc[:,sample].values
        if args.zscore:
            dfz = cnvs.pf.intervals["cnv_zscore_bins"].df
            df.loc[:,"zscore"] = dfz.loc[:,sample].values
            df = df.loc[:,['chrom', 'start', 'end', 'ratio', 'zscore']]
        else:
            df = df.loc[:,['chrom', 'start', 'end', 'ratio']]
        df.to_csv(sample+"_bins.txt", header=True, index=False, sep="\t")

        # Drop columns
        if args.zscore:
            df.drop(columns=["chrom", "start", "end", "ratio", "zscore"], inplace=True)
        else:
            df.drop(columns=["chrom", "start", "end", "ratio"], inplace=True)

    return None



    
