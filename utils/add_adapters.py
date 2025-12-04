#!/usr/bin/env python3

# Imports -----
import sys
import argparse
import gzip
from typing import Iterator, Tuple, TextIO, cast
# --------------

def smart_open(path: str, mode: str) -> TextIO:
    if path == '-':
        return sys.stdin if 'r' in mode else sys.stdout
    if path.endswith('.gz'):
        return cast(TextIO, gzip.open(path, mode + 't'))
    return open(path, mode, encoding = 'utf-8', newline = '')

def wrap_seq(seq: str, width: int) -> str:
    if width <= 0:
        return seq
    return '\n'.join(seq[i:i + width] for i in range(0, len(seq), width))

def read_fasta(handle: TextIO) -> Iterator[Tuple[str, str]]:
    header: str = ''
    chunks: list[str] = []
    for raw in handle:
        line = raw.rstrip('\n')
        if not line:
            continue
        if line.startswith('>'):
            if header:
                yield header, ''.join(chunks)
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header:
        yield header, ''.join(chunks)

def process(in_fa: str, out_fa: str, adapter5: str, adapter3: str, upper: bool, line_width: int, first_only: bool, quiet: bool) -> None:
    n_in: int = 0
    n_out: int = 0
    with smart_open(in_fa, 'r') as fin, smart_open(out_fa, 'w') as fout:
        for header, seq in read_fasta(fin):
            n_in += 1
            s = seq
            a5 = adapter5
            a3 = adapter3
            if upper:
                s = s.upper()
                a5 = a5.upper()
                a3 = a3.upper()
            out_seq = a5 + s + a3
            fout.write(f'>{header}\n')
            fout.write(wrap_seq(out_seq, line_width) + '\n')
            n_out += 1
            if first_only:
                break
    if not quiet:
        print(f'[append_adapters] wrote {n_out} of {n_in} input record(s)', file = sys.stderr)

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description = 'Append adapters to FASTA sequences.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('-i', '--input', default = '-', help = 'Input FASTA (or \'-\' for stdin). .gz ok.')
    ap.add_argument('-o', '--output', default = '-', help = 'Output FASTA (or \'-\' for stdout). .gz ok.')
    ap.add_argument('--adapter5', default = 'AGGACCGGATCAACT', help = '5\' adapter to prepend.')
    ap.add_argument('--adapter3', default = 'CATTGCGTGAACCGA', help = '3\' adapter to append.')
    ap.add_argument('--upper', action = 'store_true', help = 'Force uppercase for sequence and adapters.')
    ap.add_argument('--line-width', type = int, default = 60, help = 'FASTA line wrap width (0 for no wrap).')
    ap.add_argument('--first-only', action = 'store_true', help = 'Only process the first FASTA entry.')
    ap.add_argument('--quiet', action = 'store_true', help = 'Suppress progress to stderr.')
    return ap

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    process(
        in_fa = args.input,
        out_fa = args.output,
        adapter5 = args.adapter5,
        adapter3 = args.adapter3,
        upper = args.upper,
        line_width = args.line_width,
        first_only = args.first_only,
        quiet = args.quiet,
    )

if __name__ == '__main__':
    main()

