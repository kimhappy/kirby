# Development Guide
## How to ...
### Build *kirby*
- Run `uv build` in the project root directory.

### Add or modify Model / Loss / Optimizer / Scheduler / Trainer
- Go

### After modifying *kirby* ...
- Run `../clear.sh && uv sync --refresh` in the `cli` directory.
- If you don't have enough permission, run `chmod +x ../clear.sh`.

### Add dependency
#### to *kirby*
- Run `uv add <package name>` in the project root directory.

#### to CLI
- Run `uv add <package name>` in the `cli` directory.

### Select Python interpreter (for vscode, ...)
- Select `cli/.venv/bin/python`

## Project Tree
- `src/kirby`: *kirby* library
  - `protocol`
    - `config`: Configurations for training
    - `impl`: Base classes for model / loss / optimizer / scheduler / trainer
  - `impl`: Implementations of protocols
    - `loss`: Loss function implementations
    - `model`: Model architecture implementations
    - `optimizer`: Optimizer implementations
    - `scheduler`: Learning rate scheduler implementations
    - `trainer`: Training loop implementations
  - `util`: Utility functions and helpers
  - `train.py`: Training implementation
  - `inference.py`: Inference implementation
- `cli`: Command-line interface for the *kirby* library
  - `util`: Utility functions and helpers
  - `train.py`: Training CLI entry point
  - `inference.py`: Inference CLI entry point
