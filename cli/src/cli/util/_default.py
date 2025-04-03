def _set_default(default: dict, to: dict) -> dict:
    for key, value in default.items():
        if key not in to:
            to[ key ] = value

    return to
