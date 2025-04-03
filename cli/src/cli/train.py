import argparse
import os
import torch
import toml
from copy   import deepcopy
from dotenv import load_dotenv
import wandb
import kirby

from .util import _best_device_name, _read_mono_f32, _write_mono_f32, _power_hash

def _set_default(config: dict) -> dict:
    new_config = deepcopy(config)

    GENERAL_DEFAULTS = {
        'seed'         : 42                 ,
        'device'       : _best_device_name(),
        'deterministic': False              ,
        'compile'      : True
    }

    for key, value in GENERAL_DEFAULTS.items():
        if key not in new_config[ 'general' ]:
            new_config[ 'general' ][ key ] = value

    return new_config

def _load_data(config: dict) -> dict:
    new_config = deepcopy(config)

    for data_field in ['train_data', 'vali_data']:
        for data in new_config[ data_field ]:
            for field in ['input', 'output']:
                data[ field ] = _read_mono_f32(data[ field ])

    return new_config

def _save_audio_cache(config: dict) -> dict:
    new_config = deepcopy(config)

    for data_field in ['train_data', 'vali_data']:
        for data in new_config[ data_field ]:
            for field in ['input', 'output']:
                hash = _power_hash(data[ field ])
                path = os.path.join(CACHE_PATH, f'{ hash }.wav')

                if not os.path.exists(path):
                    _write_mono_f32(path, data[ field ])

                data[ field ] = path

    return new_config

def main():
    load_dotenv()

    global ENTITY, PROJECT, CACHE_PATH
    ENTITY     = os.getenv('ENTITY'    )
    PROJECT    = os.getenv('PROJECT'   )
    CACHE_PATH = os.getenv('CACHE_PATH')

    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type   = str, required = True, help = 'Path to config TOML file'                 )
    parser.add_argument('-f', '--force',  action = 'store_true'        , help = 'Force run even if training already exists')
    args   = parser.parse_args()

    with open(args.config, 'r') as f:
        config = toml.load(f)

    resume  = 'allow' if args.force else 'never'
    config  = _set_default     (config )
    config  = _load_data       (config )
    pconfig = _save_audio_cache(config )
    id      = _power_hash  (pconfig)

    config_path = os.path.join(CACHE_PATH, f'{ id }_config.toml')
    dict_path   = os.path.join(CACHE_PATH, f'{ id }_dict.pth'   )

    wandb_run = wandb.init(
        entity  = ENTITY ,
        project = PROJECT,
        config  = pconfig,
        id      = id     ,
        resume  = resume)

    if os.path.exists(dict_path):
        print(f'Training already exists for <{ id }>')
        print(f'https://wandb.ai/{ ENTITY }/{ PROJECT }/{ id }')
        pass
    else:
        print(f'Training with <{ id }>')

        def _callback(epoch: int, result: kirby.Result):
            print(f'Epoch { epoch }: { result.type } = { result.value }')
            wandb_run.log({ result.type: result.value })

        best_dict, result = kirby.Train(config).run(_callback)

        with open(config_path, 'w') as f:
            toml.dump(pconfig, f)

        torch.save(best_dict, dict_path)

        print(f'Best validation loss: { result.value }')

    wandb_run.finish()

if __name__ == '__main__':
    main()
