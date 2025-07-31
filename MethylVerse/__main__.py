import argparse
from .commandline.commands import MPACT_process_single


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

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

    args = parser.parse_args()

    args.func(args)