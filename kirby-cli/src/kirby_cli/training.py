from typing import Optional, Any
import argparse
import os
import sys
import numpy as np
from dotenv import load_dotenv
import torch
import wandb
import atexit
import signal
import warnings
from rich.columns import Columns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
import kirby_core
from ._util import      \
    _read_mono_f32    , \
    _write_mono_f32   , \
    _power_hash       , \
    _best_device_name , \
    _read_dict        , \
    _set_default_value, \
    _rmap

TABLE_WIDTH = 30

GENERAL_DEFAULTS = {
    'seed'         : 42                 ,
    'device'       : _best_device_name(),
    'deterministic': False              ,
    'compile'      : True
}

files_to_delete = set()

def cleanup():
    for file in files_to_delete:
        try:
            os.remove(file)
        except:
            pass

def signal_handler(sig, frame):
    cleanup()
    sys.exit(1)

def main() -> int:
    # Register cleanup function
    atexit.register(cleanup)

    # Register signal handler
    signal.signal (signal.SIGINT , signal_handler)
    signal.signal (signal.SIGTERM, signal_handler)

    # Load environment variables
    load_dotenv()

    # Make libraries silent
    os.environ[ 'WANDB_SILENT' ] = 'true'
    warnings.filterwarnings('ignore')

    # Rich console
    console = Console()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type   = str         , help = 'Path to config file (.json / .toml / .yaml / .yml)')
    parser.add_argument('-f', '--force' , action = 'store_true', help = 'Force run even if training already exists'         )
    args       = parser.parse_args()
    args.force = 'allow' if args.force else 'never'

    # W&B run for audio files
    entity  = os.getenv('ENTITY' )
    project = os.getenv('PROJECT')

    # audio path -> audio id
    tables   = []
    replaced = {}

    def _id_maker(key: str, value: Any) -> Optional[str]:
        if isinstance(key  , str) and key  .endswith('_audio') and \
           isinstance(value, str) and value.endswith('.wav'):
            samples = _read_mono_f32(value, sample_rate)
            id      = _power_hash(samples)
            _write_mono_f32(f'{id}.wav', samples, sample_rate)
            au_art.add_file(f'{id}.wav')
            files_to_delete.add(f'{id}.wav')
            replaced[ value ] = id
            return id

        return value

    # audio id -> np.ndarray
    def _np_maker(key: str, value: Any) -> Optional[np.ndarray]:
        if isinstance(key  , str) and key.endswith('_audio') and \
           isinstance(value, str):
            entry   = ad_art.get_entry(f'{value}.wav')
            path    = entry.download()
            samples = _read_mono_f32(path, sample_rate)
            files_to_delete.add(path)
            return samples

        return value

    # Read config
    config = _read_dict(args.config)

    if config is None:
        console.print('Error: Cannot read config file (JSON / TOML / YAML / YML)')
        return 1

    # Set default values
    config[ 'general' ], inserted = _set_default_value(config[ 'general' ], GENERAL_DEFAULTS)

    if inserted:
        inserted_table = Table('Key', 'Value', title = 'Inserted', width = TABLE_WIDTH)

        for key, value in inserted.items():
            inserted_table.add_row(key, repr(value))

        tables.append(inserted_table)

    # Upload audio files
    with console.status('Uploading audio files...'):
        au_run = wandb.init(
            entity  = entity        ,
            project = project       ,
            id      = 'audio-upload',
            resume  = 'allow')
        au_art      = wandb.Artifact('audio', type = 'dataset')
        sample_rate = config[ 'general' ][ 'sample_rate' ]

        id_config = _rmap(config, _id_maker)

        if id_config is None:
            console.print('Error: Cannot upload audio files')
            return 1

        au_run.log_artifact(au_art)
        au_run.finish()

    if replaced:
        replaced_table = Table('File path', 'Audio ID', title = 'Replaced', width = TABLE_WIDTH)

        for key, value in replaced.items():
            replaced_table.add_row(key, value)

        tables.append(replaced_table)

    if tables:
        console.print(Columns(tables, title = 'Preprocessing Info', expand = False))

    # Download audio files
    with console.status('Downloading audio files...'):
        ad_run = wandb.init(
            entity  = entity        ,
            project = project       ,
            id      = 'audio-download',
            resume  = 'allow')
        ad_art    = ad_run.use_artifact('audio:latest')
        np_config = _rmap(id_config, _np_maker)

        if np_config is None:
            console.print('Error: Cannot download audio files')
            return 1

        ad_run.finish()

    # Train!
    run_id     = _power_hash(id_config)
    max_epochs = config[ 'train' ][ 'max_epochs' ]

    run_info_table = Table('Info'       , 'URL', title = 'Run Info')
    run_info_table.add_row('Project'    , f'https://wandb.ai/{entity}/{project}')
    run_info_table.add_row('Audio files', f'https://wandb.ai/{entity}/{project}/artifacts/dataset/audio/latest/files')
    run_info_table.add_row('State dicts', f'https://wandb.ai/{entity}/{project}/artifacts/dataset/state_dict/latest/files')
    run_info_table.add_row('Run'        , f'https://wandb.ai/{entity}/{project}/runs/{run_id}')
    console.print(run_info_table)

    try:
        run = wandb.init(
            entity  = entity   ,
            project = project  ,
            config  = id_config,
            id      = run_id   ,
            resume  = args.force)
    except wandb.errors.errors.UsageError:
        console.print('Error: Run ID already exists. If you want to force training, use the --force option.')
        return 1

    best_val_loss = float('inf')
    best_column   = TextColumn(f'Best: {best_val_loss:.4f}')

    with Progress(
        TextColumn('{task.description}'),
        BarColumn (),
        TextColumn('{task.completed} / {task.total}'),
        best_column) as progress:
        training_task = progress.add_task('Training...', total = max_epochs)

        def _callback(epoch: int, result: kirby_core.Result):
            nonlocal best_val_loss, best_column

            run.log({ result.type: result.value })

            if result.type == 'training loss':
                progress.update(training_task, advance = 1)
            elif result.type == 'validation loss':
                if result.value < best_val_loss:
                    best_val_loss = result.value
                    progress.update(training_task)
                    best_column.text_format = f'Best: {best_val_loss:.4f}'

        best_dict, result = kirby_core.training(np_config, _callback)
        run.finish()

    # Upload state dict
    with console.status('Uploading state dict...'):
        sd_run = wandb.init(
            entity  = entity             ,
            project = project            ,
            id      = 'state-dict-upload',
            resume  = 'allow')
        sd_art = wandb.Artifact('state_dict', type = 'model')

        torch.save(best_dict, f'{run_id}.pt')
        sd_art.add_file(f'{run_id}.pt')
        files_to_delete.add(f'{run_id}.pt')
        sd_run.log_artifact(sd_art)
        sd_run.finish()

    return 0

if __name__ == '__main__':
    sys.exit(main())
