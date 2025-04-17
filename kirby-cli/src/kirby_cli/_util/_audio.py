from typing import Optional
import librosa
import numpy as np
from scipy.io import wavfile

def _read_mono_f32(
    path: str,
    sr  : int = 48000) -> Optional[np.ndarray]:
    data_sr, data = wavfile.read(path)

    if data.ndim != 1:
        return None

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
        return None

    if data_sr != sr:
        data = librosa.resample(data, orig_sr = data_sr, target_sr = sr)

    return data

def _write_mono_f32(
    path   : str       ,
    samples: np.ndarray,
    sr     : int = 48000) -> None:
    wavfile.write(path, sr, samples)
