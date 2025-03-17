# Kirby
Black-box modeling of complex non-linear audio system

## Prepare
### Clone this repository and install dependencies
```sh
git clone https://github.com/kimhappy/capstone
cd capstone
pyenv install 3.12 # install python 3.12
pyenv shell 3.12   # activate python 3.12
poetry install     # install dependencies
```

### Activate an environment
```sh
pyenv shell 3.12            # activate python 3.12
eval $(poetry env activate) # activate a virtual environment
```

## Train
### Make train.toml
Example:
```toml
[general]
seed      = 42
condition = 1

[model]
name   = 'lstm'
hidden = 64

[criterion]
name = 'mse'

[train]
max_epochs  = 2000
initial_lr  = 0.005
lr_decay    = 0.0001
lr_factor   = 0.5
lr_patience = 5
early_stop  = 20
data        = [
    [[0.0 ], 'data/train/input/0.wav', 'data/train/output/0.wav'],
    [[0.33], 'data/train/input/1.wav', 'data/train/output/1.wav'],
    [[0.66], 'data/train/input/2.wav', 'data/train/output/2.wav'],
    [[1.0 ], 'data/train/input/3.wav', 'data/train/output/3.wav'],
]

[validation]
cycle = 2
data  = [
    [[0.0 ], 'data/val/input/0.wav', 'data/val/output/0.wav'],
    [[0.33], 'data/val/input/1.wav', 'data/val/output/1.wav'],
    [[0.66], 'data/val/input/2.wav', 'data/val/output/2.wav'],
    [[1.0 ], 'data/val/input/3.wav', 'data/val/output/3.wav'],
]
```

### Run
```sh
poetry run python train.py train.toml
```
After the training, hash value would be printed.

## Inference
```sh
poetry run python main.py <hash from training>
```

## Util
### Remove unused .wav files
```sh
poetry run python util.py clean
```

### Reset
```sh
poetry run python util.py reset
```
