from typing import Any, Callable, Optional
import toml
import json
import yaml
from copy import deepcopy

def _read_dict(path: str) -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            if path.endswith('.toml'):
                return toml.load(f)
            elif path.endswith('.json'):
                return json.load(f)
            elif path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            else:
                return None
    except:
        return None

def _write_dict(
    path: str,
    data: dict) -> bool:
    try:
        with open(path, 'w') as f:
            if path.endswith('.toml'):
                toml.dump(data, f)
            elif path.endswith('.json'):
                json.dump(data, f)
            elif path.endswith(('.yaml', '.yml')):
                yaml.safe_dump(data, f, sort_keys = False)
            else:
                return False
        return True
    except:
        return False

def _set_default_value(
    to     : dict,
    default: dict) -> dict:
    new_to = deepcopy(to)

    for key, value in default.items():
        if key not in new_to:
            new_to[ key ] = value
            print(f'\'{key}\': {repr(value)}')

    return new_to

def _rmap(
    to: Any,
    f : Callable[[str, Any], Optional[Any]]) -> Optional[Any]:
    if not isinstance(to, dict):
        return to

    new_to = deepcopy(to)

    for key, value in new_to.items():
        if isinstance(value, dict):
            new_to[ key ] = _rmap(value, f)
        elif isinstance(value, (list, tuple)):
            new_value = []

            for item in value:
                new_item = _rmap(item, f)
                new_value.append(new_item)

            new_to[ key ] = new_value
        else:
            new_to[ key ] = f(key, value)

    return new_to
