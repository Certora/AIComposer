import argparse
from typing import cast
import sys

from analyzer.analysis import analyze
from analyzer.types import AnalysisArgs

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze Certora Prover counterexamples and generate natural language explanations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    %(prog)s /path/to/report myRule
    %(prog)s /path/to/report myRule --method myMethod
    %(prog)s /path/to/report myRule --method MyContract.myMethod
    """
    )

    parser.add_argument(
        'folder',
        type=str,
        help='Path to the Certora report directory containing the counterexample data'
    )

    parser.add_argument(
        'rule',
        type=str,
        help='Name of the rule to analyze'
    )

    parser.add_argument(
        '--method',
        type=str,
        default=None,
        help='Optional method identifier. Can be either "method" or "contract.method" format'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress intermediate output during analysis (only show final result)'
    )

    args = parser.parse_args()
    sys.exit(analyze(cast(AnalysisArgs, args)))
