import numpy as np
import pandas as pd
from projectframe import ProjectFrame
from intervalframe import IntervalFrame

# Local imports
from ...core.utilities import bin_cpgs


def tile_methylation(pf: ProjectFrame,
                     bin_size: int = 100000,
                     genome_version: str = "hg19",
                     remove_nans: bool = True,
                     method: str = "mean") -> ProjectFrame:
    """
    Tile methylation values into bins.

    Parameters
    ----------
        pf : ProjectFrame
            ProjectFrame of CpG methylation values.
        bin_size : int
            Number of bins to tile into.
        genome_version : str
            Genome version to use.
        remove_nans : bool
            If True, remove bins with NaNs.
        method : str
            Method to use for tiling.

    Returns
    -------
        pf : ProjectFrame
    
    """

    # Assign genome
    import genome_info
    genome = genome_info.GenomeInfo(genome_version)

    # Get bins
    bins = IntervalFrame.from_dict_range(genome["chrom_sizes"], bin_size = bin_size)

    # Tile betas
    bins = bin_cpgs(pf.intervals["betas"], bins=bins, method=method)

     # Remove NaNs
    if remove_nans:
        bins = bins.iloc[np.sum(pd.isnull(bins.values), axis=1) == 0,:]

    # Assign to projectframe
    pf.add_intervals("tile_methylation", bins)

    return pf


def region_methylation(pf: ProjectFrame,
                        feature: str = "gene",
                        upstream: int = 1000,
                        downstream: int = 1000,
                        gene_type: str = "protein_coding",
                        genome_version: str = "hg19",
                        remove_nans: bool = True,
                        method: str = "mean") -> ProjectFrame:
    """
    Region methylation values into bins.

    Parameters
    ----------
        pf : ProjectFrame
            ProjectFrame of CpG methylation values.
        bin_size : int
            Number of bins to tile into.
        genome_version : str
            Genome version to use.
        remove_nans : bool
            Remove NaNs from the resulting ProjectFrame.
        method : str
            Method to use for tiling.

    Returns
    -------
        pf : ProjectFrame
    
    """

    # Assign genome
    import genome_info
    genome = genome_info.GenomeInfo(genome_version)

    # Get regions
    if feature == "gene":
        genes = genome.get_intervals("gene", upstream = 5000, filter_column = "gene_type", filter_selection="protein_coding")
        names = genes.df.loc[:,"gene_name"].values
    elif feature == "promoter":
        genes = genome.get_intervals("tss", upstream = 5000, downstream=1000, filter_column = "gene_type", filter_selection="protein_coding")
        names = genes.df.loc[:,"gene_name"].values
    else:
        raise NotImplementedError("Only gene and promoter are supported currently")

    # Remove unnecessary columns
    genes.drop_columns(["Strand","gene_type","level","Source"])

    # Tile betas
    bins = bin_cpgs(pf.intervals["betas"], bins=genes, method=method)

     # Remove NaNs
    if remove_nans:
        selected = np.sum(pd.isnull(bins.values), axis=1) == 0
        names = names[selected]
        bins = bins.iloc[selected,:]
    bins.df.index = names
    bins = bins.df.T

    # Append to projectframe
    pf.add_values(feature+"_methylation", bins)

    return pf


def binarize_methylation(pf: ProjectFrame,
                         key: str = "betas",
                         new_key: str = "binary_methylation",
                         threshold: float = 0.6,
                         use_sparse: bool = False) -> ProjectFrame:
    """
    """

    # Copy betas
    betas = pf.intervals["betas"].copy()
    
    # Binarize
    betas.df[betas.df < threshold] = -1
    betas.df[betas.df >= threshold] = 1
    betas.df[pd.isnull(betas.df)] = 0
    if use_sparse:
        betas.df = betas.df.astype(pd.SparseDtype(int, 0))
    else:
        betas.df = betas.df.astype(int)

    # Assign to projectframe
    pf.add_intervals("binary_betas", betas)

    return pf



