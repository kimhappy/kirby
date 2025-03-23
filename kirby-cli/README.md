# Kirby CLI
CLI for the kirby library.

## Train
### Make a configuration file
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
    [[0.0 ], 'data/train/input0.wav', 'data/train/output/0.wav'],
    [[0.33], 'data/train/input1.wav', 'data/train/output/1.wav'],
    [[0.66], 'data/train/input2.wav', 'data/train/output/2.wav'],
    [[1.0 ], 'data/train/input3.wav', 'data/train/output/3.wav'],
]

[validation]
cycle = 2
data  = [
    [[0.0 ], 'data/val/input0.wav', 'data/val/output0.wav'],
    [[0.33], 'data/val/input1.wav', 'data/val/output1.wav'],
    [[0.66], 'data/val/input2.wav', 'data/val/output2.wav'],
    [[1.0 ], 'data/val/input3.wav', 'data/val/output3.wav'],
]
```

### Run
```sh
uv run train.py train.toml
```
This will create a `.pt` file and copy the `train.toml` to the `kirby/<train_id>` directory, and print the `train_id`.

## Inference
```sh
uv run inference.py <train_id>
```
This uses the informations stored in the `kirby/<train_id>` directory.
