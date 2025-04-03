import toml
import json

def _read_dict(path: str) -> dict:
    with open(path, 'r') as f:
        if path.endswith('.toml'):
            return toml.load(f)
        elif path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError(f'Unsupported file extension: { path }')

def _write_dict(path: str, data: dict):
    with open(path, 'w') as f:
        if path.endswith('.toml'):
            toml.dump(data, f)
        elif path.endswith('.json'):
            json.dump(data, f)
