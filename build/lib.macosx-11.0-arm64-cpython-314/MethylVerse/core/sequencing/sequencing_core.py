from projectframe import ProjectFrame
import ngsfragments as ngs
import glob
import pandas as pd
import numpy as np
import os

# Local imports
from .seq_utilities import read_methyldackel_bed
from .read_methyldackel import read_methyldackel                       


def read_betas(meth_files: str,
               use_sparse: bool = False):
    """
    Read methyldackel files

    Parameters
    ----------
        meth_files : str
            Methyldackel files
        use_sparse : bool = False
            Whether to use sparse optimization

    Returns
    -------
        all_cpgs : IntervalFrame
            IntervalFrame containing all CpGs
    """

    # Find sample name
    meth_names = np.array([os.path.split(file)[-1].split(".bedGraph")[0] for file in meth_files])
    # Remove CpG string
    meth_names = np.array([name.split("_CpG")[0] for name in meth_names])

    # Iterate through files
    for i, file in enumerate(meth_files):
        meth_iframe = read_methyldackel_bed(file)
        if i == 0:
            all_cpgs = meth_iframe
        else:
            all_cpgs = all_cpgs.combine([meth_iframe.index], sparse_optimize=use_sparse)
    
    return all_cpgs


def process_sequencing(methyldackel_directory: str,
                       bam_directory: str = None,
                       genome_version: str = "hg19",
                       nthreads: int = 1,
                       call_cnvs: bool = False,
                       use_sparse: bool = False,
                       verbose: bool = False):
    """
    Process sequencing data

    Parameters
    ----------
        methyldackel_directory : str
            Directory containing methyldackel files
        bam_directory : str = None
            Directory containing bam files
        genome_version : str = "hg19"
            Which genome version to use
        nthreads : int = 1
            Number of threads to use
        call_cnvs : bool = False
            Whether to call copy number variations
        verbose : bool = False
            Whether to print progress

    Returns
    -------
        pf : ProjectFrame
            Object containing results
    """

    # Import genome
    if genome_version == "hg19":
        from hg19genome import Hg19Genome
        Genome = Hg19Genome()
    elif genome_version == "hg38":
        from hg38genome import Hg38Genome
        Genome = Hg38Genome()
    else:
        raise NotImplementedError("Other genome versions not implemented...yet.")

    # Initialize projectframe
    pf = ProjectFrame()
    pf.params["ref_genome"] = genome_version

    # Find files
    meth_files = glob.glob(os.path.join(methyldackel_directory, "*.bedGraph"))
    # Find sample name
    meth_names = np.array([os.path.split(file)[-1].split(".bedGraph")[0] for file in meth_files])
    # Remove CpG string
    meth_names = np.array([name.split("_CpG")[0] for name in meth_names])

    # Read methyldackel
    if verbose: print("Reading methyldackel files", flush=True)
    betas = read_methyldackel(np.array(meth_files).astype(str), use_sparse=use_sparse)
    pf.add_intervals("betas", betas)
    pf.add_anno("methyl_type", value="Sequencing")

    # Read bams
    if call_cnvs and bam_directory is not None:
        bam_files = glob.glob(os.path.join(bam_directory, "*.bam"))
        # Find sample name
        bam_names = np.array([os.path.split(file)[-1].split(".bam")[0] for file in bam_files])
        if verbose: print("Reading bams.")
        for i, bam in enumerate(bam_files):
            if verbose: print(bam)
            pf = ngs.segment.call_cnv_pipeline(pf, bam, genome_version=genome_version, verbose=verbose, nthreads=nthreads, wgbs=True)
            del pf.uns.dict[bam_names[i]]
            pf.obs_intervals[bam_names[i]]["cnv_segments"].drop_columns(["copy_number","event","subclone_status","logR_Copy_Number"])

    return pf