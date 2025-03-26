from intervalframe import IntervalFrame
import ngsfragments as ngs

# Local imports
from ...data.import_data import get_data_file
from ...core.microarray.RawArray import RawArray


def microarray_call_cnvs(array_info: RawArray,
                         genome_version: str = "hg38",
                         n_probes: int = 10,
                         n_probes_hmm: int = 15,
                         zscore: bool = False,
                         n_mads: float = 1.0) -> ngs.segment.CNVcaller:
    """
    """

    # Read copy number values
    cn = array_info.get_cn()

    # Get normal references
    if genome_version  == "hg19":
        normal_450k_references = IntervalFrame.read_parquet(get_data_file("normal450k_hg19_cn.parquet"))
        normal_EPIC_references = IntervalFrame.read_parquet(get_data_file("normalEPIC_hg19_cn.parquet"))
    elif genome_version == "hg38":
        normal_450k_references = IntervalFrame.read_parquet(get_data_file("normal450k_hg38_cn.parquet"))
        normal_EPIC_references = IntervalFrame.read_parquet(get_data_file("normalEPIC_hg38_cn.parquet"))

    # Filter normal
    if "450k" in array_info.array_type[0]:
        cn = cn.exact_match(normal_450k_references)
        normal_references = normal_450k_references.exact_match(cn)
    elif "EPIC" in array_info.array_type[0]:
        cn = cn.exact_match(normal_EPIC_references)
        normal_references = normal_EPIC_references.exact_match(cn)
    
    # Call CNVs
    cnvs = ngs.segment.CNVcaller(genome_version = genome_version,
                                        scStates = None,
                                        n_per_bin = n_probes,
                                        n_per_bin_hmm = n_probes_hmm,
                                        n_mads = n_mads)
    cnvs.predict_cnvs(data = cn,
                        normal_data = normal_references)
    if zscore:
        cnvs.calculate_zscore()

    return cnvs