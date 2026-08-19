from intervalframe import IntervalFrame
import ngsfragments as ngs


def sequencing_call_cnvs(data: IntervalFrame,
                         genome_version: str = "hg38",
                         n_probes: int = None,
                         n_probes_hmm: int = None,
                         zscore: bool = False,
                         n_mads: float = 1.48,
                         bcp_cutoff: float = 0.3) -> ngs.segment.CNVcaller:
    """
    """

    if n_probes is None:
        n_probes = data.shape[0] // 40000
        n_probes_hmm = data.shape[0] // 20000

        # Minimum number of probes
        if n_probes < 10:
            n_probes = 10
        if n_probes_hmm < 15:
            n_probes_hmm = 15

    # Call CNVs
    cnvs = ngs.segment.CNVcaller(genome_version = genome_version,
                                        scStates = None,
                                        n_per_bin = n_probes,
                                        n_per_bin_hmm = n_probes_hmm,
                                        n_mads = n_mads,
                                        bcp_cutoff = bcp_cutoff)
    
    cnvs.predict_cnvs(data = data)

    # Calculate zscore
    if zscore:
        cnvs.calculate_zscore()

    return cnvs