import os
import re
import requests
import librosa
import numpy as np
from scipy.io import wavfile

def _read_mono_f32_from_file(
    path: str,
    sr  : int = 48000) -> np.ndarray:
    data_sr, data = wavfile.read(path)

    if data.ndim != 1:
        raise ValueError('Only mono audio is supported')

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.float32:
        pass
    elif data.dtype == np.float64:
        data = data.astype(np.float32)
        pass
    else:
        raise ValueError(f'{ data.dtype } not supported')

    if data_sr != sr:
        print(f'Resampling from { data_sr } to { sr }')
        data = librosa.resample(data, orig_sr = data_sr, target_sr = sr)

    return data

def _read_mono_f32(
    where: str,
    sr   : int = 48000) -> np.ndarray:
    _TEMP_WAV_PATH = 'temp.wav'

    is_url = re.compile(r'^(?:http|https|ftp)://\S+$')

    if is_url.match(where):
        response = requests.get(where, stream = True)

        with open(_TEMP_WAV_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)

        data = _read_mono_f32_from_file(_TEMP_WAV_PATH, sr)

        if os.path.exists(_TEMP_WAV_PATH):
            os.remove(_TEMP_WAV_PATH)
    else:
        data = _read_mono_f32_from_file(where, sr)

    return data

def _write_mono_f32(
    path   : str       ,
    samples: np.ndarray,
    sr     : int = 48000) -> None:
    wavfile.write(path, sr, samples)
