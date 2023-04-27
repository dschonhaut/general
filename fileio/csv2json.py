#!/usr/bin/env python3


import sys
import os
import os.path as op
from argparse import ArgumentParser, FileType
from collections import defaultdict
from contextlib import suppress
from io import StringIO
import csv
import json
import sys


def convert_csv(args):
    """Convert a CSV file into JSON."""
    # Set up the converter function.
    if args.convert_numbers:
        converter = _convert_numbers
    else:
        converter = lambda x: x

    # Read the CSV file.
    with args.infile:
        csv_input = args.infile.read()

    try:
        dialect = csv.Sniffer().sniff(csv_input)
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(StringIO(csv_input), dialect=dialect)

    # Convert the CSV file into the desired format.
    if args.format == "aos":
        result = [
            {field: converter(value) for field, value in row.items()} for row in reader
        ]
    elif args.format == "soa":
        result = defaultdict(list)
        for row in reader:
            for field, value in row.items():
                result[field].append(converter(value))

    # Get the output file name.
    if args.outfile is None:
        args.outfile = args.infile.name.replace(".csv", ".json")

    # Write the output file.
    if op.isfile(args.outfile) and not args.overwrite:
        print(f"{args.outfile} already exists. Use --overwrite to overwrite.")
    else:
        with open(args.outfile, "w") as outfile:
            json.dump(result, outfile, indent=4)

    # Print the output to the console.
    if args.verbose:
        print(json.dumps(result))


def _convert_numbers(x):
    """Attempt to convert a string into a number, otherwise
    return the original value.

    x -- str. E.g. "3", "hello".
    """
    with suppress(ValueError):
        return int(x)
    with suppress(ValueError):
        return float(x)
    return x


def _parse_args():
    """Parse the command line arguments."""
    parser = ArgumentParser(
        description="""
    Convert a CSV file into JSON, either into "Structure of
    Arrays" (soa) format -- {x:[...],y:[...]} -- or
    "Array of Structures" (aos) format -- [{x,y},{x,y},...].
    """
    )
    parser.add_argument(
        "infile",
        type=FileType(),
        help="CSV file to convert.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=FileType("w"),
        help="Output filepath. Default replaces .csv with .json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the output.",
    )
    parser.add_argument(
        "-c",
        "--convert_numbers",
        action="store_true",
        help="Convert strings into numbers when possible.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["soa", "aos"],
        default="soa",
        help="JSON format. Default=aos.",
    )

    # Parse the command line arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    convert_csv(args)
