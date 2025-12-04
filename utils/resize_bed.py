#!/usr/bin/env python3

# Imports -----
import sys
import argparse
import gzip
from typing import Dict, Optional, Tuple, TextIO, cast
# -------------

def smart_open(path: str, mode: str) -> TextIO:
    if path == '-':
        return sys.stdin if 'r' in mode else sys.stdout
    if path.endswith('.gz'):
        return cast(TextIO, gzip.open(path, mode + 't'))
    return open(path, mode, encoding = 'utf-8', newline = '')

def load_chrom_sizes(path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with smart_open(path, 'r') as fh:
        for line in fh:
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            chrom, size_str = parts[0], parts[1]
            try:
                size = int(size_str)
            except ValueError:
                continue
            sizes[chrom] = size
    return sizes

def clamp_interval(chrom: str, start: int, end: int, length: int, chrom_sizes: Optional[Dict[str, int]],) -> Tuple[int, int]:
    if chrom_sizes is None:
        if start < 0:
            start = 0
            end = start + length
        return start, end

    size = chrom_sizes.get(chrom)
    if size is None:
        if start < 0:
            start = 0
            end = start + length
        return start, end

    if start < 0:
        start = 0
        end = min(length, size)
    if end > size:
        end = size
        start = max(0, end - length)
    return start, end

def process(in_path: str, out_path: str, length: int, chrom_sizes_path: Optional[str] = None, quiet: bool = False) -> None:
    chrom_sizes: Optional[Dict[str, int]] = (load_chrom_sizes(chrom_sizes_path) if chrom_sizes_path else None)

    n_lines: int = 0
    bad_lines: int = 0

    with smart_open(in_path, 'r') as fin, smart_open(out_path, 'w') as fout:
        for raw in fin:
            line = raw.rstrip('\n')
            if not line or line.startswith(('track', 'browser', '#')):
                print(line, file = fout)
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                bad_lines += 1
                continue

            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                bad_lines += 1
                continue

            center: int = (start + end) // 2
            new_start: int = center - (length // 2)
            new_end: int = new_start + length

            new_start, new_end = clamp_interval(chrom, new_start, new_end, length, chrom_sizes)

            parts[1] = str(new_start)
            parts[2] = str(new_end)
            fout.write('\t'.join(parts) + '\n')
            n_lines += 1

    if not quiet:
        msg = f'[resize_bed] processed {n_lines} intervals'
        if bad_lines:
            msg += f' (skipped {bad_lines} malformed)'
        print(msg, file = sys.stderr)

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description = 'Resize BED intervals around their centers to fixed length.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('-i', '--input', default = '-', help = 'Input BED (or \'-\' for stdin). .gz ok.')
    ap.add_argument('-o', '--output', default = '-', help = 'Output BED (or \'-\' for stdout). .gz ok.')
    ap.add_argument('-n', '--length', required = True, type = int, help = 'Target length (bp).')
    ap.add_argument('--chrom-sizes', dest = 'chrom_sizes', help = 'Two-column chrom sizes file to clamp intervals (chrom\\tsize).')
    ap.add_argument('--quiet', action = 'store_true', help = 'Suppress progress to stderr.')
    return ap

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.length <= 0:
        ap.error('length must be > 0')

    process(
        in_path = args.input,
        out_path = args.output,
        length = args.length,
        chrom_sizes_path = args.chrom_sizes,
        quiet = args.quiet,
    )

if __name__ == '__main__':
    main()
