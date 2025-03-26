#from ailist import LabeledIntervalArray
#from cfdna import cfDNA
from statistics import median
from ..data.import_data import get_data_file
from intervalframe.read.read_h5 import read_h5_intervalframe
from intervalframe.read.read_text import read_bed
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray
from projectframe import ProjectFrame
import h5py
import pandas as pd
import numpy as np
import glob
import os


def predict_data_type(directory: str) -> str:
    """
    Predict data type from directory

    Parameters
    ----------
        directory : str
            Directory containing data

    Returns
    -------
        data_type : str
            Data type
    """

    # Check if bedgraph files in directory
    if os.path.isdir(directory):
        bed_files = glob.glob(directory + "**/*.bedGraph", recursive=True)
        if len(bed_files) > 0:
            return "sequencing_directory"
        else:
            return "microarray_directory"
    
    # Check if directory is idat file
    if os.path.isfile(directory + "_Grn.idat"):
        return "microarray_single"
    
    # Check if directory is bed file
    if os.path.isfile(directory + ".bedGraph") or directory.endswith(".bedGraph"):
        return "sequencing_single"

    return "unknown"    


def position_to_cpg(iframe: IntervalFrame,
                    genome_version: str = "hg38"):
    
    # Read annotation file
    if genome_version == "hg38":
        anno_file = get_data_file("hg38_anno.parquet")
    else:
        anno_file = get_data_file("hg19_anno.parquet")
    iframe_cn = IntervalFrame.read_parquet(anno_file)
    probes = iframe_cn.df.loc[:,"cpg_name"].values

    # Overlap
    cpg_name = np.empty(iframe.shape[0], dtype="U25")
    for i, index in enumerate(iframe_cn.index.iter_intersect(iframe.index, return_intervals=False, return_index=True)):
        if len(index > 0):
            cpg_name[i] = probes[index[0]]

    return cpg_name


def get_probe_names():
    # Read annotation file
    iframe_cn = IntervalFrame.read_parquet(get_data_file("hg19_anno.parquet"))
    probes = iframe_cn.df.loc[:,"cpg_name"].values

    return probes


def convert_v2_to_v1(cpgs: pd.DataFrame) -> pd.DataFrame:
    """
    """

    # Read conversion file
    v2_to_v1 = pd.read_parquet(get_data_file("EPIC2_to_EPIC.parquet"))

    # Convert
    #new_probes = cpgs.columns.intersection(v2_to_v1.index.values)
    #cpgs = cpgs.loc[:,common_probes]
    #cpgs.columns = v2_to_v1.loc[common_probes,"EPICv1_Loci"].values
    cpgs.rename(columns=v2_to_v1.to_dict()["EPICv1_Loci"], inplace=True)

    return cpgs


def cpg_to_position(cpgs,
                    genome_version: str = "hg38"):
    """
    """

    # Read annotation file
    if genome_version == "hg38":
        iframe_cn = IntervalFrame.read_parquet(get_data_file("hg38_anno.parquet"))
    else:
        iframe_cn = IntervalFrame.read_parquet(get_data_file("hg19_anno.parquet"))
    probes = iframe_cn.df.loc[:,"cpg_name"].values

    # Build cn probes
    cn_probes = pd.Series(np.arange(iframe_cn.shape[0]),
                          index=iframe_cn.df.loc[:,"cpg_name"].values)

    # Find common
    common_probes = pd.Index(cpgs).intersection(probes)
    iframe_cn = iframe_cn.iloc[cn_probes.loc[common_probes].values,:]

    return iframe_cn


def bin_cpgs(cpg_iframe,
             bins = None,
             bin_size=100000,
             method="mean"):
    """
    """

    # Chosen function
    if method == "mean":
        func = np.mean
    elif method == "sum":
        func = np.sum
    elif method == "median":
        func = np.median
    elif method == "std":
        func = np.std
    elif method == "nanmean":
        func = np.nanmean
    elif method == "nansum":
        func = np.nansum
    elif method == "nanmedian":
        func = np.nanmedian

    if bins is None:
        # Get ranges
        ranges = cpg_iframe.index.label_ranges
        range_dict = {chrom:ranges[chrom][1] for chrom in ranges}

        # Create bins
        bin_iframe = IntervalFrame.from_dict_range(range_dict, bin_size)
    else:
        bin_iframe = bins

    values = pd.DataFrame(np.zeros((bin_iframe.df.shape[0],cpg_iframe.shape[1])),
                          columns = cpg_iframe.columns.values)
    values[:] = np.nan
    for i, index in enumerate(cpg_iframe.index.iter_intersect(bin_iframe.index, return_intervals=False, return_index=True)):
        value = cpg_iframe.df.iloc[index,:].values
        values.values[i,:] = func(value, axis=0)

    bin_iframe.df = values
    
    return bin_iframe


def mean_methyl(intervals, cpg_iframe):
    """
    """

    bin_iframe = IntervalFrame(intervals=intervals.index.copy())

    values = pd.DataFrame(np.zeros((bin_iframe.df.shape[0],cpg_iframe.shape[1])),
                          columns = cpg_iframe.columns.values)
    for i, index in enumerate(cpg_iframe.index.iter_intersect(bin_iframe.index, return_intervals=False, return_index=True)):
        value = cpg_iframe.df.values[index,:]
        values.values[i,:] = np.mean(value, axis=0)

    bin_iframe.df = values
    
    return bin_iframe


def methyl_windows(intervals, cpg_iframe, obs):
    """
    """

    size = intervals.index[0].end - intervals.index[0].start
    obs_cpgs = cpg_iframe.loc[:,obs]
    
    total_values = np.zeros((intervals.shape[0], size))
    for i, interval in enumerate(intervals.index):
        cpgs = obs_cpgs.intersect(interval.start, interval.end, interval.label)
        values = np.zeros(interval.end - interval.start)
        values[:] = np.nan
        values[cpgs.index.starts - interval.start] = cpgs.values

        total_values[i,:] = values

    return total_values


def mean_bigwig(intervals, bigwig_fn):
    """
    """
    import pyBigWig
    bw = pyBigWig.open(bigwig_fn)

    means = np.zeros(len(intervals))
    for i, interval in enumerate(intervals):
        values = bw.values(interval.label, interval.start, interval.end, numpy=True)
        means[i] = np.nanmean(values)

    iframe = IntervalFrame(intervals=intervals, df=pd.DataFrame(means))
    iframe.df.columns = ["mean"]

    return iframe


def to_microarray(pf: ProjectFrame,
                    genome_version: str = "hg38") -> ProjectFrame:
    """
    """
    betas = pf.intervals["betas"]
    cpgs = position_to_cpg(betas, genome_version=genome_version)
    present = cpgs != ""
    filtered_betas = betas.df.loc[present,:]
    filtered_betas.index = cpgs[present]
    filtered_betas = filtered_betas.T
    filtered_betas = filtered_betas.loc[:,~filtered_betas.columns.duplicated()]

    pf.add_values("microarray_betas", filtered_betas)

    return pf