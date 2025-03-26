import h5py
import numpy as np
import pandas as pd
from intervalframe import IntervalFrame
import os
import glob
from typing import List, Dict, Sequence, Any
from ngsfragments.correct.correction import correct_counts

# Local imports
from .read_IDAT import read_idat, get_idat_type
from ...data.import_data import get_data_file
from ...data.download_data import download_methyl_anno
from .illumina_normalize import control_illumina_single, bgcorrect_illumina
from ..utilities import cpg_to_position, convert_v2_to_v1


def sort_chrom_names(chrom_names: Sequence[str]):
    """
    """

    # Sort chromosomes
    has_chr = chrom_names[0].startswith("chr")
    chroms = [l.split("chr")[-1] if l.startswith("chr") else l for l in chrom_names]
    int_chroms = []
    str_chroms = []
    for chrom in chroms:
        try:
            int_chroms.append(int(chrom))
        except ValueError:
            str_chroms.append(chrom)
    int_chroms.sort()
    str_chroms.sort()
    if has_chr:
        sorted_chroms = np.array(["chr"+str(c) for c in int_chroms + str_chroms])
    else:
        sorted_chroms = np.array([str(c) for c in int_chroms + str_chroms])

    return sorted_chroms


def parse_csv(filename: str):
    """
    Parse CSV formatted file
    """
    
    record = {}
    
    # iterate over lines in file
    for i, line in enumerate(open(filename,"r")):
        fields = line.strip().split(",")
        
        # if header
        if i == 0:
            n_fields = len(fields)
            header = fields
            for field in fields:
                record[field] = []  
              
        # if not header
        else:
            for j in range(n_fields):
                record[header[j]].append(fields[j])
                
    return record


def find_files(filenames: List, 
               match: Sequence[str] = None):
    # Pair up files
    file_pairs = {}
    idat_ids = set(["_".join(os.path.basename(f).split("_")[:2]) for f in filenames])
    for id in idat_ids:
        file_pairs[id] = ["tmp","tmp"]
        for f in filenames:
            if id in f:
                if "Red" in f:
                    file_pairs[id][0] = f
                else:
                    file_pairs[id][1] = f
                    
    if match is not None:
        filtered_pairs = {}
        for name in match:
            filtered_pairs[name] = file_pairs[name]

        return filtered_pairs

    else:
        return file_pairs
    

def find_files_v2(filenames: List, 
               match: Sequence[str] = None):
    

    # Pair up files
    array_types = ["27k","450k","EPIC_v1","EPIC_v2"]
    file_pairs = {t:{} for t in np.unique(array_types)}

    # Iterate over files
    #idat_ids = set(["_".join(os.path.basename(f).split("_")[:2]) for f in filenames])
    idat_ids = set(os.path.basename(f).split("_Grn")[0].split("_Red")[0] for f in filenames)
    for id in idat_ids:
        #file_pairs[id] = ["tmp","tmp"]
        for f in filenames:
            if id in f:
                array_type = get_idat_type(f)

                try:
                    file_pairs[array_type][id]
                except KeyError:
                    file_pairs[array_type][id] = ["tmp","tmp"]

                if "Red" in f:
                    file_pairs[array_type][id][0] = f
                else:
                    file_pairs[array_type][id][1] = f
                    
    if match is not None:
        filtered_pairs = {t:{} for t in np.unique(array_types)}
        for t in np.unique(array_types):
            for name in match:
                filtered_pairs[t][name] = file_pairs[t][name]

        return filtered_pairs

    else:
        return file_pairs


def find_files_v3(filenames: List, 
               match: Sequence[str] = None):
    

    # Define supported array types
    array_types = ["27k", "450k", "EPIC_v1", "EPIC_v2"]
    file_pairs = {t: {} for t in array_types}

    # Iterate through filenames to extract sample IDs and organize by array type
    for f in filenames:
        # Extract sample ID
        sample_id = os.path.basename(f).split("_Grn")[0].split("_Red")[0]
        array_type = get_idat_type(f)

        # Initialize if not present
        file_pairs[array_type].setdefault(sample_id, ["tmp", "tmp"])

        # Identify and store the correct channel
        if "Red" in f:
            file_pairs[array_type][sample_id][0] = f
        elif "Grn" in f:
            file_pairs[array_type][sample_id][1] = f

    # Filter results if 'match' is provided
    if match is not None:
        filtered_pairs = {t: {} for t in array_types}
        for t in array_types:
            for name in match:
                if name in file_pairs[t]:
                    filtered_pairs[t][name] = file_pairs[t][name]
        return filtered_pairs

    return file_pairs
        

def validate_idat_pairs(file_pairs: Dict[str, Dict]):
    """
    Remove pairs with only one idat
    """
    
    for array_type in file_pairs:
        names = list(file_pairs[array_type].keys())
        for idat_name in names:
            if file_pairs[array_type][idat_name][0] == "tmp":
                del file_pairs[array_type][idat_name]
            elif file_pairs[array_type][idat_name][1] == "tmp":
                del file_pairs[array_type][idat_name]
                
    return file_pairs


def validate_idat_pairs_v2(file_pairs: Dict[str, Dict]):
    """
    Remove pairs with only one idat.
    """

    filtered_pairs =  {array_type: {
                                idat_name: pair
                                for idat_name, pair in file_pairs[array_type].items()
                                if pair[0] != "tmp" and pair[1] != "tmp"
                                    } for array_type in file_pairs}

    return filtered_pairs


def read_all_idats(filenames: List[str],
                   verbose: bool = False):
    """
    Iteratively read idats and merge info
    """

    # Pair up files
    file_pairs = find_files(filenames)
    array_type = get_idat_type(filenames[0])
    
    # record all required idat info
    names = np.array(list(file_pairs.keys()))

    # Read all possible probes
    #man = get_data_file("hg19_manifest.h5")
    #f = h5py.File(man, "r")
    probe_files = {"27k": get_data_file("HM450_probes.parquet"),
                    "450k": get_data_file("HM450_probes.parquet"),
                    "EPIC_v1": get_data_file("EPIC_probes.parquet"),
                    "EPIC_v2": get_data_file("EPIC2_probes.parquet")}
    control_files = {"27k": get_data_file("controls.parquet"),
                        "450k": get_data_file("controls.parquet"),
                        "EPIC_v1": get_data_file("controls.parquet"),
                        "EPIC_v2": get_data_file("EPIC2_controls.parquet")}
    snp_files = {"27k": get_data_file("snps.parquet"),
                    "450k": get_data_file("snps.parquet"),
                    "EPIC_v1": get_data_file("snps.parquet"),
                    "EPIC_v2": get_data_file("EPIC2_snps.parquet")}
    probes_file = probe_files[array_type]
    controls_file = control_files[array_type]
    snps_file = snp_files[array_type]

    # Read control and snp probes
    #total_ids = pd.Series(np.array(f["probes"]["names"]).astype(str),
    #                        index = np.array(f["probes"]["illumina_id"]))
    #controls = pd.Series(np.array(f["controls"]["illumina_id"]),
    #                        index=np.array(f["controls"]["Type"]).astype(str))
    #snp_probes = pd.Series(np.array(f["snps"]["names"]).astype(str),
    #                        index = np.array(f["snps"]["illumina_id"]))
    #snp_probe_type = pd.Series(np.array(f["snps"]["type"]).astype(str),
    #                        index = np.array(f["snps"]["illumina_id"]))
    #probe_type = pd.Series(np.array(f["probes"]["type"]).astype(str),
    #                        index = np.array(f["probes"]["illumina_id"]))
    total_ids = pd.read_parquet(probes_file, columns=["names"]).names
    controls = pd.read_parquet(controls_file).illumina_id
    snp_probes = pd.read_parquet(snps_file, columns=["names"]).names
    snp_probe_type = pd.read_parquet(snps_file, columns=["type"]).type
    probe_type = pd.read_parquet(probes_file, columns=["type"]).type

    probe_names = pd.unique(total_ids.values)
    snp_names = pd.unique(snp_probes.values)

    #f.close()

    # Initialize DataFrames for green and red intensity values
    meth = pd.DataFrame(np.zeros((len(names), len(probe_names))),
                            index = names,
                            columns = probe_names)
    meth.values[:] = np.nan

    unmeth = pd.DataFrame(np.zeros((len(names), len(probe_names))),
                            index = names,
                            columns = probe_names)
    unmeth.values[:] = np.nan

    snps_unmeth = pd.DataFrame(np.zeros((len(names), len(snp_names))),
                        index = names,
                        columns = snp_names)
    snps_unmeth[:] = np.nan
    snps_meth = pd.DataFrame(np.zeros((len(names), len(snp_names))),
                        index = names,
                        columns = snp_names)
    snps_meth[:] = np.nan

    # Iterate over idats and normalize
    for k, name in enumerate(names):
        if verbose: print("Reading IDATs for sample %s" % name, flush=True)
        Grn = file_pairs[name][1]
        Red = file_pairs[name][0]

        idat_grn = read_idat(Grn)
        idat_red = read_idat(Red)

        control_illumina_single(idat_grn, idat_red, controls, ref=10000)
        bgcorrect_illumina(idat_grn, idat_red, controls)

        sample_probes = total_ids.index.intersection(idat_grn.index.values).values
        sample_snps = snp_probes.index.intersection(idat_grn.index.values).values
        #sample_probes = sample_probes.difference(sample_snps).values

        # Set meth
        selected = probe_type.loc[sample_probes].values == "B_I_Red"
        values = idat_red.loc[sample_probes,"means"][selected]
        #values = values.groupby(by=meth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).mean()
        values = values[~pd.Index(meth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).duplicated()]
        #meth.loc[name,values.index.values] = values.values
        meth.loc[name, total_ids.loc[values.index.values]] = values.values

        selected = probe_type.loc[sample_probes].values == "B_I_Grn"
        values = idat_grn.loc[sample_probes,"means"][selected]
        #values = values.groupby(by=meth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).mean()
        values = values[~pd.Index(meth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).duplicated()]
        #meth.loc[name,values.index.values] = values.values
        meth.loc[name, total_ids.loc[values.index.values]] = values.values
        
        selected = probe_type.loc[sample_probes].values == "A_II_Both"
        values = idat_grn.loc[sample_probes,"means"][selected]
        #values = values.groupby(by=meth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).mean()
        values = values[~pd.Index(meth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).duplicated()]
        #meth.loc[name,values.index.values] = values.values
        meth.loc[name, total_ids.loc[values.index.values]] = values.values

        # Set unmeth
        selected = probe_type.loc[sample_probes].values == "A_I_Red"
        values = idat_red.loc[sample_probes,"means"][selected]
        #values = values.groupby(by=unmeth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).mean()
        values = values[~pd.Index(unmeth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).duplicated()]
        #unmeth.loc[name,values.index.values] = values.values
        unmeth.loc[name, total_ids.loc[values.index.values]] = values.values
        
        selected = probe_type.loc[sample_probes].values == "A_I_Grn"
        values = idat_grn.loc[sample_probes,"means"][selected]
        #values = values.groupby(by=unmeth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).mean()
        values = values[~pd.Index(unmeth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).duplicated()]
        #unmeth.loc[name,values.index.values] = values.values
        unmeth.loc[name, total_ids.loc[values.index.values]] = values.values
        
        selected = probe_type.loc[sample_probes].values == "A_II_Both"
        values = idat_red.loc[sample_probes,"means"][selected]
        #values = values.groupby(by=unmeth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).mean()
        values = values[~pd.Index(unmeth.loc[name,total_ids.loc[sample_probes].values[selected]].index.values).duplicated()]
        #unmeth.loc[name,values.index.values] = values.values
        unmeth.loc[name, total_ids.loc[values.index.values]] = values.values

        # Set snps
        selected = snp_probe_type.loc[sample_snps].values == "A_I_Red"
        values = idat_red.loc[sample_snps,"means"][selected]
        #values = values.groupby(by=snps_unmeth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).mean()
        values = values[~pd.Index(snps_unmeth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).duplicated()]
        #snps_unmeth.loc[name,values.index.values] = values.values
        #snps_unmeth.loc[name, total_ids.loc[values.index.values]] = values.values
        snps_unmeth.loc[name, snp_probes.loc[values.index.values]] = values.values
        
        selected = snp_probe_type.loc[sample_snps].values == "A_I_Grn"
        values = idat_grn.loc[sample_snps,"means"][selected]
        #values = values.groupby(by=snps_unmeth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).mean()
        values = values[~pd.Index(snps_unmeth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).duplicated()]
        #snps_unmeth.loc[name,values.index.values] = values.values
        #snps_unmeth.loc[name, total_ids.loc[values.index.values]] = values.values
        snps_unmeth.loc[name, snp_probes.loc[values.index.values]] = values.values
        
        selected = snp_probe_type.loc[sample_snps].values == "A_II_Both"
        values = idat_red.loc[sample_snps,"means"][selected]
        #values = values.groupby(by=snps_unmeth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).mean()
        values = values[~pd.Index(snps_unmeth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).duplicated()]
        #snps_unmeth.loc[name,values.index.values] = values.values
        #snps_unmeth.loc[name, total_ids.loc[values.index.values]] = values.values
        snps_unmeth.loc[name, snp_probes.loc[values.index.values]] = values.values

        selected = snp_probe_type.loc[sample_snps].values == "B_I_Red"
        values = idat_red.loc[sample_snps,"means"][selected]
        #values = values.groupby(by=snps_meth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).mean()
        values = values[~pd.Index(snps_meth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).duplicated()]
        #snps_meth.loc[name,values.index.values] = values.values
        #snps_meth.loc[name, total_ids.loc[values.index.values]] = values.values
        snps_meth.loc[name, snp_probes.loc[values.index.values]] = values.values
        
        selected = snp_probe_type.loc[sample_snps].values == "B_I_Grn"
        values = idat_grn.loc[sample_snps,"means"][selected]
        #values = values.groupby(by=snps_meth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).mean()
        values = values[~pd.Index(snps_meth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).duplicated()]
        #snps_meth.loc[name,values.index.values] = values.values
        #snps_meth.loc[name, total_ids.loc[values.index.values]] = values.values
        snps_meth.loc[name, snp_probes.loc[values.index.values]] = values.values
        
        selected = snp_probe_type.loc[sample_snps].values == "A_II_Both"
        values = idat_grn.loc[sample_snps,"means"][selected]
        #values = values.groupby(by=snps_meth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).mean()
        values = values[~pd.Index(snps_meth.loc[name,snp_probes.loc[sample_snps].values[selected]].index.values).duplicated()]
        #snps_meth.loc[name,values.index.values] = values.values
        #snps_meth.loc[name, total_ids.loc[values.index.values]] = values.values
        snps_meth.loc[name, snp_probes.loc[values.index.values]] = values.values

        #snps.loc[name,sample_snps] = idat_grn.loc[sample_snps,"means"].values  / (idat_grn.loc[sample_snps,"means"].values + idat_red.loc[sample_snps,"means"].values)

    # Filter out NaNs
    selected = pd.isnull(meth.values).sum(axis=0) < meth.shape[0]
    meth = meth.loc[:,selected]
    unmeth = unmeth.loc[:,selected]

    # Drop snps
    meth.drop(columns=snps_meth.columns.values, inplace=True, errors="ignore")
    unmeth.drop(columns=snps_unmeth.columns.values, inplace=True, errors="ignore")
                
    return meth, unmeth, snps_meth, snps_unmeth


def combine_methylation_values(df1: pd.DataFrame,
                               df2: pd.DataFrame,
                               df1_type: str,
                               df2_type: str):
    """
    """

    # Combine
    if df1.shape[0] == 0:
        df1 = df2.copy()
        return df1
    
    # Check if EPIC version conversion needed
    if df1_type == "EPIC_v1" and df2_type == "EPIC_v2":
        df2 = convert_v2_to_v1(df2)
        df2 = df2.loc[:,~df2.columns.duplicated()]
    elif df1_type == "450k" and df2_type == "EPIC_v2":
        df2 = convert_v2_to_v1(df2)
        df2 = df2.loc[:,~df2.columns.duplicated()]
    
    # Add new columns
    new_columns = df2.columns.union(df1.columns).values
    df1 = df1.reindex(columns=new_columns)
    df2 = df2.reindex(columns=new_columns)

    # Concatenate
    values = np.zeros((df1.shape[0] + df2.shape[0], df1.shape[1]))
    values[:,:] = np.nan
    values[:df1.shape[0],:] = df1.values
    values[df1.shape[0]:,:] = df2.values
    df1 = pd.DataFrame(values,
                       index=np.concatenate([df1.index.values, df2.index.values]),
                       columns=new_columns)

    return df1


def multiprocess_read_all_idats(files: Sequence,
                                n_jobs: int = 1):
    """
    Read all idats in files using multiprocessing.

    Parameters
    ----------
        files : Sequence
            List of idat files.
        n_jobs : int
            Number of jobs to use.

    Returns
    -------
        meth : pd.DataFrame
            Methylation values.
        unmeth : pd.DataFrame
            Unmethylation values.
        snps_meth : pd.DataFrame
            Methylation values for SNPs.
        snps_unmeth : pd.DataFrame
            Unmethylation values for SNPs.
        array_types : np.ndarray
            Array types.
    """

    # Initialize
    total_meth = pd.DataFrame([])
    total_unmeth = pd.DataFrame([])
    total_snp_meth = pd.DataFrame([])
    total_snp_unmeth = pd.DataFrame([])
    arraytype_file_pairs = find_files_v3(files)
    arraytype_file_pairs = validate_idat_pairs_v2(arraytype_file_pairs)

    # Determine array types
    array_types = []
    for arraytype in arraytype_file_pairs.keys():
        if len(arraytype_file_pairs[arraytype]) > 0:
            array_types += [arraytype] * len(arraytype_file_pairs[arraytype])
    array_types = np.array(array_types)
    previous_array_type = None

    if n_jobs == 1:
        for arraytype in arraytype_file_pairs.keys():
            file_pairs = arraytype_file_pairs[arraytype]
            if len(file_pairs) == 0:
                continue
            # Read files
            #filenames = [reduce(lambda a,b:a+b, list(file_pairs.values())) for i in range(0, len(file_pairs))]
            filenames = np.array(list(file_pairs.values())).flatten()
            results = read_all_idats(filenames, verbose=True)

            # Combine results
            total_meth = combine_methylation_values(total_meth, results[0], previous_array_type, arraytype)
            total_unmeth = combine_methylation_values(total_unmeth, results[1], previous_array_type, arraytype)
            total_snp_meth = combine_methylation_values(total_snp_meth, results[2], previous_array_type, arraytype)
            total_snp_unmeth = combine_methylation_values(total_snp_unmeth, results[3], previous_array_type, arraytype)

            previous_array_type = arraytype

    else:
        from functools import reduce
        import multiprocessing as mp

        # Split files into n_jobs
        for arraytype in arraytype_file_pairs.keys():
            file_pairs = arraytype_file_pairs[arraytype]
            if len(file_pairs) == 0:
                continue
            n = int(np.ceil(len(file_pairs) / n_jobs))
            job_files = [reduce(lambda a,b:a+b, list(file_pairs.values())[i:i+n]) for i in range(0, len(file_pairs), n)]
            # Read files
            with mp.Pool(n_jobs) as pool:
                results = pool.map(read_all_idats, job_files)

            # Combine results
            meth = []
            unmeth = []
            snps_meth = []
            snps_unmeth = []
            for i in range(len(results)):
                meth.append(results[i][0])
                unmeth.append(results[i][1])
                snps_meth.append(results[i][2])
                snps_unmeth.append(results[i][3])
            meth_values = pd.concat(meth, axis=0)
            unmeth_values = pd.concat(unmeth, axis=0)
            snps_meth_values = pd.concat(snps_meth, axis=0)
            snps_unmeth_values = pd.concat(snps_unmeth, axis=0)

            # Combine results
            total_meth = combine_methylation_values(total_meth, meth_values, previous_array_type, arraytype)
            total_unmeth = combine_methylation_values(total_unmeth, unmeth_values, previous_array_type, arraytype)
            total_snp_meth = combine_methylation_values(total_snp_meth, snps_meth_values, previous_array_type, arraytype)
            total_snp_unmeth = combine_methylation_values(total_snp_unmeth, snps_unmeth_values, previous_array_type, arraytype)

            previous_array_type = arraytype
        
    return total_meth, total_unmeth, total_snp_meth, total_snp_unmeth, array_types


class RawArray(object):
    """
    """

    def __init__(self, 
                 data: Any = None,
                 genome_version: str = "hg38",
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        """

        self.genome_version = genome_version

        # Make sure genome is downloaded
        download_methyl_anno()

        if data is None:
            pass
        
        # load information from h5 file
        elif isinstance(data, str):
            if os.path.isfile(data) and data.endswith(".h5"):
                #load_RGset(filename)
                pass

            elif os.path.isdir(data):
                data = glob.glob(data+"**/*.idat", recursive=True)
                if verbose:
                    print("Found %i idat files" % len(data), flush=True)
                self.meth, self.unmeth, self.snps_meth, self.snps_unmeth, self.array_type = multiprocess_read_all_idats(data, n_jobs=n_jobs)
        
        elif isinstance(data, list):
            if verbose:
                print("Found %i idat files" % len(data), flush=True)
            self.meth, self.unmeth, self.snps_meth, self.snps_unmeth, self.array_type = multiprocess_read_all_idats(data, n_jobs=n_jobs)

    
    def filter_common(self):
        """
        """

        selected = pd.isnull(self.meth.values).sum(axis=0) == 0
        self.meth = self.meth.loc[:,selected]
        self.unmeth = self.unmeth.loc[:,selected]

        #selected = pd.isnull(self.snps.values).sum(axis=0) == self.snps.shape[0]
        #self.snps = self.snps.loc[:,selected]

        return
    

    def get_betas(self,
                  offset: int = 100,
                  return_intervals: bool = False):
        """
        Calculate beta values from methylation and unmethylation counts

        Parameters
        ----------
            offset : int
                Offset to add to methylation and unmethylation counts to avoid
                division by zero
            return_intervals : bool
                If True, return a IntervalFrame with beta values for each
                CpG site. If False, return a DataFrame with beta values for each
                sample
            
        Returns
        -------
            beta : pd.DataFrame or IntervalFrame
        """

        # Calculate beta
        beta = self.meth / (self.meth + self.unmeth + offset)

        # Calculate intervals
        if return_intervals:
            pos_cn = cpg_to_position(beta.columns.values,
                                     genome_version=self.genome_version)

            sorted_chroms = sort_chrom_names(pos_cn.index.unique_labels)
            pos_cn = pos_cn.loc[sorted_chroms,:]

            pos_cn.df = beta.T.loc[pos_cn.df.loc[:,"cpg_name"].values,:]

            return pos_cn

        else:
            return beta

    
    def get_cn(self,
               bias_correct: bool = False):
        """
        """

        cn = self.meth + self.unmeth

        pos_cn = cpg_to_position(cn.columns.values,
                                 genome_version=self.genome_version)

        sorted_chroms = sort_chrom_names(pos_cn.index.unique_labels)
        pos_cn = pos_cn.loc[sorted_chroms,:]

        pos_cn.df = cn.T.loc[pos_cn.df.loc[:,"cpg_name"].values,:]

        if bias_correct:
            # Assign genome
            if self.genome_version == "hg19":
                from hg19genome import calculate_bias
            elif self.genome_version == "hg38":
                from hg38genome import calculate_bias
            else:
                raise ValueError("Genome version not supported")

            # Initialize bias records
            bias_record = calculate_bias(pos_cn.index)
            
            # Remove blacklist
            chosen = bias_record.df.loc[:,"blacklist"].values < 0.1
            bias_record = bias_record.iloc[chosen,:]
            pos_cn = pos_cn.iloc[chosen,:]
            bias_record.drop_columns(["blacklist"])

            # Correct
            for sample in pos_cn.df.columns:
                pos_cn.df.loc[:,sample] = correct_counts(pos_cn.df.loc[:,sample].values,
                                                                        bias_record.df)

        return pos_cn


    def get_snps(self,
                 offset: int = 100):
        """
        """
        
        # Read SNP intervals
        if self.genome_version == "hg19":
            intervals = IntervalFrame.read_parquet(get_data_file("hg19_snp_anno.parquet"))
        else:
            intervals = IntervalFrame.read_parquet(get_data_file("hg38_snp_anno.parquet"))

        # Re-order
        snps = self.snps_meth / (self.snps_meth + self.snps_unmeth + offset)
        snps = snps.T
        found = np.in1d(intervals.df.loc[:,"cpg_name"].values, snps.index.values)
        intervals = intervals.iloc[found,:]
        snps = snps.loc[intervals.df.loc[:,"cpg_name"].values,:]
        snp_intervals = IntervalFrame(intervals=intervals.index, df=snps)

        return snp_intervals