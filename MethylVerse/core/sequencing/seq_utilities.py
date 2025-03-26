#from ailist import LabeledIntervalArray
#from cfdna import cfDNA
from statistics import median
from ...data.import_data import get_data_file
from intervalframe.read.read_h5 import read_h5_intervalframe
from intervalframe.read.read_text import read_bed
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray
import h5py
import pandas as pd
import numpy as np


def read_methyldackel_bed(filename: str,
                          use_str: bool = False):
    """
    """

    # Read
    data = read_bed(filename, skipfirst=True)

    if data.shape[0] > 0:
        # Calculate beta
        data.df.loc[:,"betas"] = data.df.loc[:,4].values / (data.df.loc[:,4].values + data.df.loc[:,5].values)
        data.df = data.df.drop([3, 4, 5], axis=1)

        if use_str:
            # Convert
            starts = np.array(["_"+s for s in data.index.starts.astype(str)]).astype(str)
            chroms = data.index.labels.astype(str)
            cpg_names = np.core.defchararray.add(chroms, starts)

            # Make DataFrame
            data = data.df
            data.index = cpg_names
    else:
        data.df["betas"] = []

    return data