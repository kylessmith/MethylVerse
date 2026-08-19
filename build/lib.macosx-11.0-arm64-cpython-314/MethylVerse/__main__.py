import argparse
from .commandline.commands import MPACT_process_single, MethylSegment_process_single, CNV_process_single, read_dir_methylation, NonmalignantDecomposition


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Nonmalignant decomposition
    nonmalignant_parser = subparsers.add_parser("nonmalignant_decomposition")
    nonmalignant_parser.add_argument("input_data", type=str, help="Input data")
    nonmalignant_parser.add_argument("--prefix", type=str, help="Output prefix", default="MethylVerse_nonmalignant_decomposition_")
    nonmalignant_parser.add_argument("--tumor_type", type=str, help="Tumor type for nonmalignant decomposition (default = NonmalignantBackground)", default="NonmalignantBackground")
    nonmalignant_parser.add_argument("--impute", action="store_true", help="Impute data before nonmalignant decomposition")
    nonmalignant_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    nonmalignant_parser.set_defaults(func=NonmalignantDecomposition)

    # Run MPACT
    mpact_parser = subparsers.add_parser("MPACT")
    mpact_parser.add_argument("input_data", type=str, help="Input data")
    mpact_parser.add_argument("--impute", action="store_true", help="Impute data")
    mpact_parser.add_argument("--regress", action="store_true", help="Regress data")
    mpact_parser.add_argument("--probability_threshold", type=float, help="Probability threshold for M-PACT classification", default=0.7)
    mpact_parser.add_argument("--max_contamination_fraction", type=float, help="Max contamination fraction for M-PACT classification", default=0.3)
    mpact_parser.add_argument("--call_cnvs", action="store_true", help="Call CNVs")
    mpact_parser.add_argument("--out", type=str, help="Output file", default="MPACT_classifications.tsv")
    mpact_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    mpact_parser.set_defaults(func=MPACT_process_single)

    # Methylation segments
    segment_parser = subparsers.add_parser("MethylSegment")
    segment_parser.add_argument("input_data", type=str, help="Input data")
    segment_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    segment_parser.set_defaults(func=MethylSegment_process_single)

    # Call CNV detection
    cnv_parser = subparsers.add_parser("CNV")
    cnv_parser.add_argument("input_data", type=str, help="Path to directory")
    cnv_parser.add_argument("--genome_version", type=str, help="Genome version (default = hg38)", default="hg38")
    cnv_parser.add_argument("--n_probes", type=int, help="Number of probes (default = 10 or None)", default=0)
    cnv_parser.add_argument("--n_probes_hmm", type=int, help="Number of probes for HMM (default = 15 or None)", default=0)
    cnv_parser.add_argument("--zscore", action="store_true", help="Calculate z-score")
    cnv_parser.add_argument("--n_mads", type=float, help="Number of MADS (default = 1.0 or 1.48)", default=0.0)
    cnv_parser.add_argument("--bcp_cutoff", type=float, help="BCP cutoff (default = 0.2 or 0.3)", default=0.0)
    cnv_parser.add_argument("--rpca", action="store_true", help="Use RPCA normalization")
    cnv_parser.add_argument("--bins_file", action="store_true", help="Output CNV bins file")
    cnv_parser.add_argument("--seg_file", action="store_true", help="Output CNV segments file")
    cnv_parser.add_argument("--segs", action="store_true", help="Output seg file")
    cnv_parser.add_argument("--anno_file", action="store_true", help="Output annotation file")
    cnv_parser.add_argument("--anno_segs", action="store_true", help="Output annotation segs file")
    cnv_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    cnv_parser.set_defaults(func=CNV_process_single)

    # Read methylation
    methyl_parser = subparsers.add_parser("read_methylation")
    methyl_parser.add_argument("input_data", type=str, help="Input data")
    methyl_parser.add_argument("--genome_version", type=str, help="Genome version (default = hg38)", default="hg38")
    methyl_parser.add_argument("--n_jobs", type=int, help="Number of jobs (default = 1)", default=1)
    methyl_parser.add_argument("--to_probes", action="store_true", help="Return data as probes")
    methyl_parser.add_argument("--prefix", type=str, help="Output prefix", default="MethylVerse_methylation")
    methyl_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    methyl_parser.set_defaults(func=read_dir_methylation)

    args = parser.parse_args()

    args.func(args)