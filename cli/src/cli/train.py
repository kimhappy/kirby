import argparse
import os
import torch
import toml
from copy import deepcopy
import kirby

from .util import _best_device_name, _read_mono_f32, _write_mono_f32, _recursive_hash

_CACHE_PATH = os.path.join(os.getcwd(), 'kirby-cache')

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
                hash = _recursive_hash(data[ field ])
                path = os.path.join(_CACHE_PATH, f'{ hash }.wav')

                if not os.path.exists(path):
                    _write_mono_f32(path, data[ field ])

                data[ field ] = path

    return new_config

def main():
    if not os.path.exists(_CACHE_PATH):
        os.makedirs(_CACHE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type   = str, required = True, help = 'Path to config TOML file'                    )
    parser.add_argument('-f', '--force' , action = 'store_true'        , help = 'Force training even if results already exist')
    args   = parser.parse_args()

    with open(args.config, 'r') as f:
        config = toml.load(f)

    config  = _set_default     (config )
    config  = _load_data       (config )
    pconfig = _save_audio_cache(config )
    hash    = _recursive_hash  (pconfig)

    info_path   = os.path.join(_CACHE_PATH, f'{ hash }_info.toml'  )
    dict_path   = os.path.join(_CACHE_PATH, f'{ hash }_dict.pth'   )
    result_path = os.path.join(_CACHE_PATH, f'{ hash }_result.toml')

    if not args.force and os.path.exists(result_path) and os.path.exists(dict_path):
        print(f'Loading train id <{ hash }>')

        with open(result_path, 'r') as f:
            result = kirby.Result.from_dict(toml.load(f))
    else:
        print(f'Training with train id <{ hash }>')

        callback          = lambda epoch, result: print(f'[{ epoch }] { result }')
        best_dict, result = kirby.Train(config).run(callback)

        with open(info_path, 'w') as f:
            toml.dump(pconfig, f)

        with open(result_path, 'w') as f:
            toml.dump(result.to_dict(), f)

        torch.save(best_dict, dict_path)
        print('Done!')

    print(result)

if __name__ == '__main__':
    main()
