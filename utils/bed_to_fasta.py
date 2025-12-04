#!/usr/bin/env python3

# Imports -----
import sys
import argparse
import gzip
from typing import Dict, Optional, Tuple, TextIO, cast
from pyfaidx import Fasta
# --------------

def smart_open(path: str, mode: str) -> TextIO:
    if path == '-':
        return sys.stdin if 'r' in mode else sys.stdout
    if path.endswith('.gz'):
        return cast(TextIO, gzip.open(path, mode + 't'))
    return open(path, mode, encoding = 'utf-8', newline = '')

def rc(seq: str) -> str:
    table: Dict[str, str] = {'A':'T','C':'G','G':'C','T':'A','a':'t','c':'g','g':'c','t':'a','N':'N','n':'n'}
    return ''.join(table.get(b, b) for b in reversed(seq))

def wrap_seq(seq: str, width: int) -> str:
    if width <= 0:
        return seq
    return '\n'.join(seq[i:i + width] for i in range(0, len(seq), width))

def clamp_interval(chrom: str, start: int, end: int, chrom_len: int) -> Tuple[int, int]:
    if start < 0:
        start = 0
    if end > chrom_len:
        end = chrom_len
    if end < start:
        end = start
    return start, end

def process(in_bed: str, out_fa: str, fasta_path: str, use_name: bool, upper: bool, line_width: int, strict: bool, quiet: bool) -> None:
    fa = Fasta(fasta_path, as_raw = True, sequence_always_upper = False)
    skipped: int = 0
    written: int = 0

    with smart_open(in_bed, 'r') as fin, smart_open(out_fa, 'w') as fout:
        for raw in fin:
            line = raw.rstrip('\n')
            if not line or line.startswith(('#', 'track', 'browser')):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                skipped += 1
                continue

            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                skipped += 1
                continue

            name: Optional[str] = parts[3] if (use_name and len(parts) >= 4 and parts[3]) else None
            strand: Optional[str] = parts[5] if len(parts) >= 6 else None

            if chrom not in fa:
                if strict:
                    skipped += 1
                    continue
                else:
                    skipped += 1
                    continue

            clen = len(fa[chrom])
            s, e = clamp_interval(chrom, start, end, clen)

            if strict and (s != start or e != end or s == e):
                skipped += 1
                continue
            if s == e:
                skipped += 1
                continue

            seq = fa[chrom][s:e]
            seq_s = str(seq)

            if strand == '-':
                seq_s = rc(seq_s)

            if upper:
                seq_s = seq_s.upper()

            header_id = name if name else f'{chrom}:{start}-{end}{("(" + strand + ")") if strand in ("+","-") else ""}'
            fout.write(f'>{header_id}\n')
            fout.write(wrap_seq(seq_s, line_width) + '\n')
            written += 1

    if not quiet:
        print(f'[bed_to_fasta] wrote {written} sequences (skipped {skipped})', file = sys.stderr)

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description = 'Extract FASTA sequences for BED intervals.')
    ap.add_argument('-i', '--input', default = '-', help = 'Input BED (or \'-\' for stdin). .gz ok.')
    ap.add_argument('-o', '--output', default = '-', help = 'Output FASTA (or \'-\' for stdout). .gz ok.')
    ap.add_argument('-f', '--fasta', required = True, help = 'Reference FASTA (indexed with .fai).')
    ap.add_argument('--use-name', action = 'store_true', help = 'Use BED name (col 4) as FASTA header if present.')
    ap.add_argument('--upper', action = 'store_true', help = 'Force uppercase sequences.')
    ap.add_argument('--line-width', type = int, default = 60, help = 'FASTA line wrap width (0 for no wrap).')
    ap.add_argument('--strict', action = 'store_true', help = 'Skip intervals that exceed chromosome bounds or are empty.')
    ap.add_argument('--quiet', action = 'store_true', help = 'Suppress progress to stderr.')
    return ap

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    process(in_bed = args.input, out_fa = args.output, fasta_path = args.fasta, use_name = args.use_name, upper = args.upper, line_width = args.line_width, strict = args.strict, quiet = args.quiet)

if __name__ == '__main__':
    main()

