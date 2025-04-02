from typing import Any
import hashlib
import json
import numpy as np

def _recursive_hash(obj: Any) -> str:
    def _hash_dict(d: dict) -> str:
        sorted_dict = json.dumps(d, sort_keys = True)
        dict_hash   = hashlib.sha256(sorted_dict.encode()).hexdigest()

        for key, value in d.items():
            if isinstance(value, (dict, list)):
                nested_hash = _recursive_hash(value)
                dict_hash   = hashlib.sha256((dict_hash + nested_hash).encode()).hexdigest()

        return dict_hash

    if isinstance(obj, dict):
        return _hash_dict(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return hashlib.sha256(''.join(str(_recursive_hash(item)) for item in obj).encode()).hexdigest()
    else:
        if isinstance(obj, str):
            obj = obj.encode()
        elif isinstance(obj, np.ndarray):
            obj = obj.tobytes()
        else:
            raise ValueError(f'{ type(obj) } not supported')

        return hashlib.sha256(obj).hexdigest()
