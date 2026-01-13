"""CLI entry point for PDF to GP5 converter."""

import argparse
from pathlib import Path

from omnitab.converter import OmniTabConverter


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF sheet music to Guitar Pro 5 format"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert PDF to GP5")
    convert_parser.add_argument("input", type=str, help="Input PDF file path")
    convert_parser.add_argument(
        "-o", "--output", type=str, help="Output GP5 file path"
    )
    convert_parser.add_argument(
        "--tuning",
        type=str,
        default="standard",
        help="Guitar tuning (standard, drop-d, etc.)",
    )

    args = parser.parse_args()

    if args.command == "convert":
        input_path = Path(args.input)
        output_path = (
            Path(args.output) if args.output else input_path.with_suffix(".gp5")
        )

        converter = OmniTabConverter()
        result = converter.convert(input_path, output_path)

        if result.success:
            print(f"Successfully converted to: {output_path}")
        else:
            print(f"Error: {result.error}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
