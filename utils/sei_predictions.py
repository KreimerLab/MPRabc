import argparse
import concurrent.futures
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from selene_sdk.utils import load_path, parse_configs_and_run

from utils import get_data, get_filename_prefix, get_targets

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime
    torch = None


def _replace_config_paths(configs: dict, base_dir: str) -> None:
    '''Recursively replace `<PATH>` placeholders with `base_dir`.'''
    for key, value in configs.items():
        if hasattr(value, 'keywords'):
            _replace_config_paths(value.keywords, base_dir)
        elif isinstance(value, dict):
            _replace_config_paths(value, base_dir)
        elif isinstance(value, str) and '<PATH>' in value:
            configs[key] = value.replace('<PATH>', base_dir)


def _run_sequence_prediction(
    fasta_path: str,
    output_dir: str,
    use_cuda: bool,
    batch_size: Optional[int],
) -> List[str]:
    '''Execute Selene's sequence prediction workflow.'''

    if not fasta_path.endswith(('.fa', '.fasta', '.fa.gz', '.fasta.gz')):
        raise ValueError('Input file must be a FASTA (*.fa or *.fasta).')

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    chromatin_dir = os.path.join(output_dir, 'chromatin-profiles-hdf5')
    os.makedirs(chromatin_dir, exist_ok = True)

    existing_h5 = set(glob.glob(os.path.join(chromatin_dir, '*.h5')))

    configs = load_path(os.path.join(repo_dir, 'model', 'sei_seq_prediction.yml'),
                        instantiate = False)
    _replace_config_paths(configs, repo_dir)

    configs['prediction'].update(input_path = fasta_path, output_dir = chromatin_dir)

    analyze_sequences = configs['analyze_sequences']
    if batch_size is not None:
        analyze_sequences.bind(batch_size = batch_size)
    analyze_sequences.bind(use_cuda = use_cuda)

    parse_configs_and_run(configs)

    updated_h5 = set(glob.glob(os.path.join(chromatin_dir, '*.h5')))
    new_files = sorted(updated_h5 - existing_h5)
    if not new_files:
        raise RuntimeError('No prediction files were created. Please check the logs.')
    return new_files


def _write_chromatin_profile_outputs(
    prediction_file: str,
    output_dir: str,
    top_k: int,
) -> Dict[str, str]:
    '''Format epigenomic feature predictions for a single HDF5 output.'''

    chromatin_data = get_data(prediction_file)
    pred_dir, pred_filename = os.path.split(prediction_file)
    prefix = get_filename_prefix(pred_filename)

    row_labels_path = os.path.join(pred_dir, f'{prefix}_row_labels.txt')
    if not os.path.exists(row_labels_path):
        raise FileNotFoundError(
            f"Row labels file '{row_labels_path}' was not found for '{prediction_file}'."
        )

    row_labels = pd.read_csv(row_labels_path, sep = '\t')
    if len(row_labels) != len(chromatin_data):
        raise ValueError(
            'Row labels and prediction matrix have mismatched lengths: '
            f'{len(row_labels)} vs {len(chromatin_data)}.'
        )

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    feature_names = np.array(get_targets(os.path.join(model_dir, 'target.names')))
    if chromatin_data.shape[1] != len(feature_names):
        raise ValueError(
            'Prediction matrix and feature name list have mismatched widths: '
            f'{chromatin_data.shape[1]} vs {len(feature_names)}.'
        )

    outputs: Dict[str, str] = {}

    npy_path = os.path.join(output_dir, f'{prefix}.chromatin_profile_predictions.npy')
    np.save(npy_path, chromatin_data)
    outputs['npy'] = npy_path

    predictions_df = pd.DataFrame(chromatin_data, columns = feature_names)
    combined_df = pd.concat([row_labels.reset_index(drop = True), predictions_df], axis = 1)

    if len(combined_df) > 10000:
        tsv_path = os.path.join(
            output_dir,
            f'{prefix}.chromatin_profile_predictions.tsv.gz',
        )
        combined_df.to_csv(tsv_path, sep = '\t', index = False, compression = 'gzip')
    else:
        tsv_path = os.path.join(
            output_dir,
            f'{prefix}.chromatin_profile_predictions.tsv',
        )
        combined_df.to_csv(tsv_path, sep = '\t', index = False)
    outputs['chromatin_profile_tsv'] = tsv_path

    if top_k:
        if top_k < 0:
            raise ValueError('--top-k must be a non-negative integer')
        if top_k > chromatin_data.shape[1]:
            top_k = chromatin_data.shape[1]

        partition_indices = np.argpartition(-chromatin_data, kth = top_k - 1, axis = 1)[:, :top_k]
        partition_scores = np.take_along_axis(chromatin_data, partition_indices, axis = 1)

        order = np.argsort(-partition_scores, axis = 1)
        top_indices = np.take_along_axis(partition_indices, order, axis = 1)
        top_scores = np.take_along_axis(partition_scores, order, axis = 1)
        top_labels = feature_names[top_indices]

        summary_df = row_labels.copy()
        for rank in range(top_k):
            summary_df[f'top_{rank + 1}_feature'] = top_labels[:, rank]
            summary_df[f'top_{rank + 1}_value'] = top_scores[:, rank]

        top_path = os.path.join(
            output_dir,
            f'sorted.{prefix}.top{top_k}_features.tsv',
        )
        summary_df.to_csv(top_path, sep = '\t', index = False)
        outputs['top_features_tsv'] = top_path

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description = 'Run Sei on a FASTA file and format chromatin profile predictions.'
    )
    parser.add_argument('fasta', nargs = '?', help = 'Input FASTA file containing sequences to score.')
    parser.add_argument('--fasta', dest = 'fasta_flag', help = 'Input FASTA file containing sequences to score.')
    parser.add_argument('output_dir', nargs = '?', help = 'Directory where outputs (HDF5, NPY, TSV) will be saved.')
    parser.add_argument('--output-dir', dest = 'output_dir_flag', help = 'Directory where outputs (HDF5, NPY, TSV) will be saved.')
    parser.add_argument('--cuda', action = 'store_true', help = 'Enable CUDA for inference.')
    parser.add_argument('--top-k', type = int, default = 5, help = 'Number of top epigenomic features to report per sequence (default: 5).')
    parser.add_argument('--batch-size', type = int, default = None, help = 'Override Sei batch size (default: YAML config value).')
    parser.add_argument('--torch-threads', type = int, default = None, help = 'Set torch.set_num_threads and related environment variables.')
    parser.add_argument('--torch-interop-threads', type = int, default = None, help = 'Set torch.set_num_interop_threads (default heuristic).')
    parser.add_argument('--format-workers', type = int, default = None, help = 'Threads for formatting HDF5 outputs (default: min(#files, CPU cores)).')

    args = parser.parse_args()

    fasta_path = args.fasta_flag or args.fasta
    if not fasta_path:
        parser.error('a FASTA path must be supplied via positional argument or --fasta')

    output_dir = args.output_dir_flag or args.output_dir
    if not output_dir:
        parser.error('an output directory must be supplied via positional argument or --output-dir')

    os.makedirs(output_dir, exist_ok = True)

    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError('--torch-threads must be a positive integer')
        if torch is None:
            raise ImportError('PyTorch is required to set thread counts but is not installed.')
        for env_var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[env_var] = str(args.torch_threads)
        torch.set_num_threads(args.torch_threads)

    if args.torch_interop_threads is not None:
        if args.torch_interop_threads < 1:
            raise ValueError('--torch-interop-threads must be a positive integer')
        if torch is None:
            raise ImportError('PyTorch is required to set interop threads but is not installed.')
        torch.set_num_interop_threads(args.torch_interop_threads)

    prediction_files = _run_sequence_prediction(fasta_path, output_dir, args.cuda, args.batch_size)

    print(f'Generated {len(prediction_files)} chromatin profile prediction file(s).')

    if len(prediction_files) == 1:
        format_workers = 1
    else:
        max_workers_default = min(len(prediction_files), os.cpu_count() or 1)
        if args.format_workers is None:
            format_workers = max_workers_default
        else:
            if args.format_workers < 1:
                raise ValueError('--format-workers must be a positive integer')
            format_workers = args.format_workers

    results: Dict[str, Dict[str, str]] = {}

    if format_workers == 1:
        for prediction_file in prediction_files:
            results[prediction_file] = _write_chromatin_profile_outputs(prediction_file, output_dir, args.top_k)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers = format_workers) as executor:
            future_to_file = {
                executor.submit(
                    _write_chromatin_profile_outputs,
                    prediction_file,
                    output_dir,
                    args.top_k,
                ): prediction_file
                for prediction_file in prediction_files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                prediction_file = future_to_file[future]
                results[prediction_file] = future.result()

    for prediction_file in prediction_files:
        paths = results[prediction_file]
        print('Chromatin profile outputs:')
        for label, path in paths.items():
            print(f'  {label}: {path}')


if __name__ == '__main__':
    main()
