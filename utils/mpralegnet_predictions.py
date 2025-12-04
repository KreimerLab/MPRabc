#!/usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from fasta import FastaDataset
from trainer import LitModel, TrainingConfig


def check_seqs(seqs: list[str], batch_size: int) -> None:
    lengths = {len(seq) for seq in seqs}
    if len(lengths) != 1 and batch_size != 1:
        raise Exception(
            'All sequences in the file must be of the same size or batch size should be set to 1'
        )
    if len(lengths) != 1 and batch_size == 1:
        print(
            'Warning: sequences in the file are not of the same size',
            file = sys.stderr,
        )

    if len(lengths) == 1:
        (seq_len,) = tuple(lengths)
        if seq_len != 230:
            print(
                'Warning: sequence length differs from 230. This can affect predictions quality',
                file = sys.stderr,
            )
    elif 230 not in lengths:
        print(
            'Warning: none of the sequences are 230bp long. This can affect predictions quality',
            file = sys.stderr,
        )


def run_pred(
    trainer: pl.Trainer,
    model: LitModel,
    dataset: FastaDataset,
    batch_size: int,
) -> np.ndarray:
    dataloader = DataLoader(dataset, batch_size = batch_size)
    y_preds = trainer.predict(model, dataloaders = dataloader)
    y_preds = torch.concat(y_preds).cpu().numpy()
    return y_preds


@dataclass
class FoldArtifacts:
    name: str
    config_path: Path
    checkpoint_paths: list[Path]


def _select_checkpoint(paths: Iterable[Path]) -> Path:
    candidates = sorted(paths)
    if not candidates:
        raise FileNotFoundError('No checkpoint (.ckpt) files were found')

    for preferred in candidates:
        if 'last' in preferred.name:
            return preferred
    return candidates[-1]


_TEST_VAL_RE = re.compile(r'.*?_test(\d+)_val(\d+)\.ckpt$')


def _parse_test_val(p: Path) -> tuple[int, int] | None:
    m = _TEST_VAL_RE.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def discover_folds(models_root: Path, tests_filter: set[int] | None = None) -> list[FoldArtifacts]:
    root_configs = list(models_root.glob('config.json'))
    root_ckpts = sorted(models_root.glob('*.ckpt'))

    if root_configs and root_ckpts:
        config_path = root_configs[0]

        grouped: dict[int, list[Path]] = {}
        for ckpt in root_ckpts:
            parsed = _parse_test_val(ckpt)
            if parsed is None:
                continue
            test_id, _ = parsed
            if tests_filter is not None and test_id not in tests_filter:
                continue
            grouped.setdefault(test_id, []).append(ckpt)

        folds: list[FoldArtifacts] = []
        for test_id, ckpts in sorted(grouped.items()):
            ckpts_sorted = sorted(
                ckpts,
                key = lambda p: (_parse_test_val(p)[1] if _parse_test_val(p) else 0),
            )
            folds.append(
                FoldArtifacts(
                    name = f'test{test_id}',
                    config_path = config_path,
                    checkpoint_paths = ckpts_sorted,
                )
            )

        if not folds:
            raise FileNotFoundError(
                f'No matching test groups found under {models_root}. '
                'Ensure filenames look like best_model_test{N}_val{M}.ckpt and/or adjust --tests.'
            )
        return folds

    folds: list[FoldArtifacts] = []
    for path in sorted(models_root.iterdir()):
        if not path.is_dir():
            continue

        config_candidates = list(path.glob('config.json'))
        if not config_candidates:
            config_candidates = list(path.rglob('config.json'))
        if not config_candidates:
            continue

        ckpt_candidates = list(path.rglob('*.ckpt'))
        if not ckpt_candidates:
            continue

        config_path = config_candidates[0]
        checkpoint_path = _select_checkpoint(ckpt_candidates)
        folds.append(
            FoldArtifacts(
                name = path.name,
                config_path = config_path,
                checkpoint_paths = [checkpoint_path],
            )
        )

    if not folds:
        raise FileNotFoundError(
            f'No folds with config.json and checkpoint found under {models_root}'
        )
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fasta',
        type = str,
        required = True,
        help = 'Path to fasta file with sequences to score',
    )
    parser.add_argument(
        '--models_root',
        type = str,
        required = True,
        help = 'Directory containing either: '
               '(a) a flat set of *.ckpt files plus config.json, grouped by test ID; or '
               '(b) fold subdirectories with config.json and checkpoints',
    )
    parser.add_argument(
        '--out_path',
        type = str,
        required = True,
        help = 'Path to write averaged predictions (TSV)',
    )
    parser.add_argument(
        '--device',
        type = str,
        default = '0',
        help = 'GPU device id (e.g. 0) or "cpu" to run predictions on the CPU',
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 512,
        help = 'Batch size for prediction dataloader',
    )
    parser.add_argument(
        '--precision',
        type = str,
        default = '16-mixed',
        help = 'Precision argument passed to the Lightning Trainer',
    )
    parser.add_argument(
        '--tests',
        type = str,
        default = 'all',
        help = 'Comma-separated test IDs to run (e.g. "10" or "1,3,7") or "all" for all discovered tests (flat layout).',
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    models_root = Path(args.models_root)

    if not fasta_path.exists():
        raise FileNotFoundError(f'FASTA file not found: {fasta_path}')
    if not models_root.exists():
        raise FileNotFoundError(f'Models directory not found: {models_root}')

    tests_filter: set[int] | None = None
    if args.tests.strip().lower() != 'all':
        try:
            tests_filter = {int(x.strip()) for x in args.tests.split(',') if x.strip()}
            if not tests_filter:
                tests_filter = None
        except ValueError:
            raise ValueError('Invalid --tests value. Use "all" or a comma-separated list of integers, e.g. "10" or "1,2,3".')

    folds = discover_folds(models_root, tests_filter = tests_filter)

    forward_dataset = FastaDataset(str(fasta_path), reverse = False)
    seqs = forward_dataset.raw_seqs()
    check_seqs(seqs, args.batch_size)

    reverse_dataset: FastaDataset | None = None

    fold_predictions: list[np.ndarray] = []
    reverse_flags: set[bool] = set()

    accelerator = 'cpu'
    devices: int | list[int] = 1
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested but no GPU is available')
        accelerator = 'gpu'
        devices = [int(args.device)]

    trainer = pl.Trainer(
        accelerator = accelerator,
        devices = devices,
        precision = args.precision,
    )

    map_location = torch.device('cpu') if accelerator == 'cpu' else None

    for fold in folds:
        train_cfg = TrainingConfig.from_json(fold.config_path)
        reverse_flags.add(train_cfg.reverse_augment)

        ckpt_preds: list[np.ndarray] = []
        for ckpt_path in fold.checkpoint_paths:
            model = LitModel.load_from_checkpoint(
                ckpt_path,
                tr_cfg = train_cfg,
                map_location = map_location,
            )

            preds = run_pred(
                trainer = trainer,
                model = model,
                dataset = forward_dataset,
                batch_size = args.batch_size,
            )

            if train_cfg.reverse_augment:
                if reverse_dataset is None:
                    reverse_dataset = FastaDataset(str(fasta_path), reverse = True)
                reverse_preds = run_pred(
                    trainer = trainer,
                    model = model,
                    dataset = reverse_dataset,
                    batch_size = args.batch_size,
                )
                preds = (preds + reverse_preds) / 2.0

            ckpt_preds.append(preds)

        fold_avg = np.mean(np.stack(ckpt_preds, axis = 0), axis = 0)
        fold_predictions.append(fold_avg)

    if len(reverse_flags) > 1:
        raise ValueError(
            'Fold configs disagree on reverse_augment. Ensure all folds were trained with the same setting.'
        )

    stacked = np.stack(fold_predictions, axis = 0)
    averaged = stacked.mean(axis = 0)

    names = forward_dataset.seq_names()

    with open(args.out_path, 'w') as out:
        for name, score in zip(names, averaged):
            print(name, score, sep = '\t', file = out)


if __name__ == '__main__':
    main()

