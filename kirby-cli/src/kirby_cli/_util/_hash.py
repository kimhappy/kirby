from typing import Any, Optional
import hashlib
import struct
import numpy as np

def _power_hash(obj: Any) -> Optional[str]:
    m = hashlib.sha256()

    if isinstance(obj, dict):
        def _valid_item(item: Any) -> bool:
            return (
                item[ 0 ] is not None and
                item[ 1 ] is not None and (
                    not isinstance(item[ 0 ], str) or
                    not item[ 0 ].startswith('_')))

        m.update('dict'.encode())
        obj = sorted(filter(_valid_item, obj.items()))
    elif isinstance(obj, int):
        m.update('int'.encode())
        obj = struct.pack('>q', obj)
    elif isinstance(obj, float):
        m.update('float'.encode())
        obj = struct.pack('>d', obj)
    elif isinstance(obj, bool):
        m.update('bool'.encode())
        obj = struct.pack('>B', int(obj))
    elif isinstance(obj, str):
        m.update('str'.encode())
        obj = obj.encode()
    elif isinstance(obj, np.ndarray):
        m.update('ndarray'.encode())
        obj = obj.tobytes()

    if isinstance(obj, (bytes, bytearray)):
        m.update(obj)
    elif isinstance(obj, (list, tuple)):
        m.update('list'.encode())

        for item in obj:
            m.update(_power_hash(item).encode())
    else:
        return None

    return m.hexdigest()[:7]
