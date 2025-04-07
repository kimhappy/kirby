from typing import Optional, Any
import argparse
import os
import sys
import numpy as np
from dotenv import load_dotenv
import torch
import wandb
import kirby

from ._util import      \
    _read_mono_f32    , \
    _write_mono_f32   , \
    _power_hash       , \
    _best_device_name , \
    _read_dict        , \
    _set_default_value, \
    _rmap

GENERAL_DEFAULTS = {
    'seed'         : 42                 ,
    'device'       : _best_device_name(),
    'deterministic': False              ,
    'compile'      : True
}

def main() -> int:
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type   = str         , help = 'Path to config file (.json or .toml)'     )
    parser.add_argument('-f', '--force' , action = 'store_true', help = 'Force run even if training already exists')
    args       = parser.parse_args()
    args.force = 'allow' if args.force else None

    # W&B run for audio files
    entity          = os.getenv('ENTITY' )
    project         = os.getenv('PROJECT')
    files_to_delete = []

    # audio path -> audio id
    def _id_maker(key: str, value: Any) -> Optional[str]:
        if isinstance(key  , str) and key  .endswith('_audio') and \
           isinstance(value, str) and value.endswith('.wav'):
            samples = _read_mono_f32(value, sample_rate)
            id      = _power_hash(samples)
            _write_mono_f32(f'{ id }.wav', samples, sample_rate)
            au_art.add_file(f'{ id }.wav')
            files_to_delete.append(f'{ id }.wav')
            return id

        return value

    # audio id -> np.ndarray
    def _np_maker(key: str, value: Any) -> Optional[np.ndarray]:
        if isinstance(key  , str) and key.endswith('_audio') and \
           isinstance(value, str):
            entry   = ad_art.get_entry(f'{ value }.wav')
            path    = entry.download()
            samples = _read_mono_f32(path, sample_rate)
            files_to_delete.append(path)
            return samples

        return value

    # Read config
    config = _read_dict(args.config)

    if config is None:
        print('ERROR: CANNOT READ FILE (JSON / TOML)')
        return 1

    # Set default values
    config[ 'general' ] = _set_default_value(config[ 'general' ], GENERAL_DEFAULTS)
    sample_rate         = config[ 'general' ][ 'sample_rate' ]

    # Upload audio files
    au_run = wandb.init(
        entity  = entity        ,
        project = project       ,
        id      = 'audio-upload',
        resume  = 'allow')
    au_art = wandb.Artifact('audio', type = 'dataset')

    id_config = _rmap(config, _id_maker)

    if id_config is None:
        print('ERROR: CANNOT UPLOAD AUDIO FILES')
        return 1

    au_run.log_artifact(au_art)
    au_run.finish()

    # Download audio files
    ad_run = wandb.init(
        entity  = entity        ,
        project = project       ,
        id      = 'audio-download',
        resume  = 'allow')
    ad_art = ad_run.use_artifact('audio:latest')

    np_config = _rmap(id_config, _np_maker)

    if np_config is None:
        print('ERROR: CANNOT DOWNLOAD AUDIO FILES')
        return 1

    ad_run.finish()

    # Train!
    run_id = _power_hash(id_config)
    run    = wandb.init(
        entity  = entity   ,
        project = project  ,
        config  = id_config,
        id      = run_id   ,
        resume  = args.force)

    def _callback(epoch: int, result: kirby.Result):
        print(f'EPOCH { epoch }: { result.type } = { result.value }')
        run.log({ result.type: result.value })

    best_dict, result = kirby.train(np_config, _callback)
    print(f'BEST VALIDATION LOSS: { result.value }')
    run.finish()

    # Upload state dict
    sd_run = wandb.init(
        entity  = entity             ,
        project = project            ,
        id      = 'state-dict-upload',
        resume  = 'allow')
    sd_art = wandb.Artifact('state_dict', type = 'model')

    torch.save(best_dict, f'{ run_id }.pth')
    sd_art.add_file(f'{ run_id }.pth')
    sd_run.log_artifact(sd_art)
    sd_run.finish()

    # Remove temporary files
    for file in files_to_delete:
        os.remove(file)

    return 0

if __name__ == '__main__':
    sys.exit(main())
