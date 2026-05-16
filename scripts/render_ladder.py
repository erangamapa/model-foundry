"""Write the ladder table into the README between the ladder markers."""

import argparse
import sys
from pathlib import Path

from foundry.ladder import render_table

START = '<!-- ladder:start -->'
END = '<!-- ladder:end -->'
README = Path(__file__).resolve().parent.parent / 'README.md'


def replace_between_markers(text: str, table: str) -> str:
    start, end = text.find(START), text.find(END)
    if start == -1 or end < start:
        raise ValueError('ladder markers not found in README')
    return f'{text[: start + len(START)]}\n{table}\n{text[end:]}'


def main() -> int:
    parser = argparse.ArgumentParser(description='Render the ladder table into the README')
    parser.add_argument('--check', action='store_true', help='fail if the README is stale')
    args = parser.parse_args()

    current = README.read_text(encoding='utf-8')
    updated = replace_between_markers(current, render_table())
    if args.check:
        return 0 if updated == current else 1
    README.write_text(updated, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
