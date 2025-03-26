import glob
import os
from typing import List
from projectframe import ProjectFrame
import gc
import numpy as np
import pandas as pd
from ailist import LabeledIntervalArray
from intervalframe import IntervalFrame

# Local imports
from ..data.import_data import get_data_file
from .sequencing import read_methyldackel
from .microarray import RawArray


def bams_in_directory(directory: str) -> bool:
    """
    Check if bams are present in directory
    """

    files = glob.glob(os.path.join(directory, "*.bam"))
    if len(files) > 0:
        return True
    else:
        return False


def idats_in_directory(directory: str) -> bool:
    """
    Check if idats are present in directory
    """

    files = glob.glob(os.path.join(directory, "*.idat"))
    if len(files) > 0:
        return True
    else:
        return False


def read_methylbed(filenames: str,
                   read_cov: bool = False,
                   verbose: bool = False) -> pd.DataFrame:
    """
    """

    if verbose:
        print("Reading", filenames, flush=True)

    # Initialize
    filename = filenames[0]
    # Read
    x = pd.read_csv(filename, header=None, index_col=None, sep='\s+')

    # Filter
    if read_cov:
        x.loc[:,filename] = x.iloc[:,9].values
    else:
        x.loc[:,filename] = x.iloc[:,10].values / 100.0
    x = x.loc[x.iloc[:,3].values == "m",:]
    ail = LabeledIntervalArray()
    ail.add(x.iloc[:,1].values, x.iloc[:,2].values + 1, x.iloc[:,0].values)
    x = IntervalFrame(df=x.drop(columns=x.columns.values[x.columns.values!=filename]), intervals=ail)

    # Create list
    iframes = []
    iframes.append(x)

    for filename in filenames[1:]:
        # Read
        smp = pd.read_csv(filename, header=None, index_col=None, sep="\s+")

        # Filter
        if read_cov:
            smp.loc[:,filename] = smp.iloc[:,9].values
        else:
            smp.loc[:,filename] = smp.iloc[:,10].values / 100.0
        smp = smp.loc[smp.iloc[:,3].values == "m",:]
        ail = LabeledIntervalArray()
        ail.add(smp.iloc[:,1].values, smp.iloc[:,2].values + 1, smp.iloc[:,0].values)
        smp = IntervalFrame(df=smp.drop(columns=smp.columns.values[smp.columns.values!=filename]), intervals=ail)
        iframes.append(smp)

    # Merge
    iframe = IntervalFrame.combine(iframes)
    del iframes
    gc.collect()

    return iframe


def read_bed(filenames: str,
             read_cov: bool = False,
             verbose: bool = False) -> pd.DataFrame:
    """
    """

    if verbose:
        print("Reading", filenames, flush=True)

    # Initialize
    x = IntervalFrame()

    for filename in filenames:
        # Read
        x = IntervalFrame.read_bed(filename, header=None, skipfirst=True)

        # Filter
        if read_cov:
            x.df.loc[:,filename] = x.df.iloc[:,1].values + x.df.iloc[:,2].values
        else:
            x.df.loc[:,filename] = x.df.iloc[:,0].values / 100.0
        x.df = x.df.drop(columns=x.df.columns.values[x.df.columns.values!=filename])

    return x


def read_sequencing_methylation(file_path: str,
                                read_cov: bool = False,
                                verbose: bool = False) -> IntervalFrame:
    """
    """

    # Initialize
    methyl_data = []
    methylation = None
    
    # Read uncompressed bedgraph files
    ## Check if file_path is a file name
    if os.path.isfile(file_path) and ".bedGraph" in file_path:
        seq_files = np.array([file_path])
    elif os.path.isfile(file_path) and ".bedgraph" in file_path:
        seq_files = np.array([file_path])
    else:
        seq_files = glob.glob(file_path + "/*.bedGraph*")
        seq_files = np.array(glob.glob(file_path + "/*.bedgraph*") + seq_files)
    if len(seq_files) > 0:
        if verbose:
            print("Reading", seq_files, flush=True)
        seq_methylation = read_methyldackel(seq_files, use_sparse=False, read_coverage=read_cov)
        methyl_data.append(seq_methylation)
        del seq_methylation
        gc.collect()

    # Read bed files from methylbed
    if os.path.isfile(file_path) and file_path.endswith(".bed"):
        seq_files = [file_path]
    elif os.path.isfile(file_path) and file_path.endswith(".bed.gz"):
        seq_files = [file_path]
    elif os.path.isfile(file_path) and ".bedmethyl" in file_path:
        seq_files = [file_path]
    else:
        seq_files = glob.glob(file_path + "/*.bed")
        seq_files = glob.glob(file_path + "/*.bed.gz") + seq_files
        seq_files = glob.glob(file_path + "/*.bedmethyl") + seq_files
        seq_files = np.array(glob.glob(file_path + "/*.bedmethyl.gz") + seq_files)
    if len(seq_files) > 0:
        if verbose:
            print("Reading", seq_files, flush=True)
        seq_methylation = read_methylbed(seq_files, read_cov=read_cov)
        methyl_data.append(seq_methylation)
        del seq_methylation
        gc.collect()

    # Merge
    if len(methyl_data) > 0:
        methylation = IntervalFrame.combine(methyl_data)
        del methyl_data
        gc.collect()

    return methylation


def read_methylation(file_path: str | List[str],
                     genome_version: str = "hg38",
                     verbose: bool = False) -> IntervalFrame:
    """
    """

    # Read parquet
    if file_path.endswith(".parquet"):
        methylation = IntervalFrame.read_parquet(file_path)

    else:
        # Read sequencing data
        seq_methylation = read_sequencing_methylation(file_path, verbose=verbose)

        # Read idat data
        idat_files = glob.glob(file_path + "/*.idat")
        if len(idat_files) > 0:
            if verbose:
                print("Reading", idat_files, flush=True)
            # Read
            data = RawArray(file_path, genome_version=genome_version, n_jobs=1, verbose=False)
            idat_methylation = data.get_betas(return_intervals=True)

        # Merge
        if seq_methylation is not None and len(idat_files) > 0:
            methylation = IntervalFrame.combine([seq_methylation, idat_methylation])
        elif seq_methylation is not None > 0:
            methylation = seq_methylation
        elif len(idat_files) > 0:
            methylation = idat_methylation
        else:
            raise ValueError("No methylation files found in directory")
    
    return methylation




