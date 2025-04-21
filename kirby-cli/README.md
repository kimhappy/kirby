# *kirby-cli*
*kirby-cli* 는 [*kirby-core*](../kirby-core/README.md) 의 CLI입니다.

## 요구사항
- [UV](https://docs.astral.sh/uv)
- [W&B](https://wandb.ai) 계정

## 사용법
### 시작하기
`kirby-cli` 디렉토리에 `.env` 파일을 만듭니다.
```
ENTITY=...
PROJECT=...
```
  - `ENTITY`: W&B 계정 이름
  - `PROJECT`: 프로젝트 이름

### 프로젝트 초기화하기
> 새 W&B 프로젝트를 만드는 경우에만 필요합니다.

`kirby-cli` 디렉토리에서 다음의 명령을 실행합니다.
```sh
uv run initialize
```

### *training config file* 만들기
Training을 위해서는 *training config file* 을 만들어야 합니다. *training config file* 의 예시는 다음과 같습니다:
```json
{
  "general": {
    "num_cond"   : 0,
    "sample_rate": 48000
  },
  "train": {
    "max_epochs": 1000,
    "early_stop": 10,
    "vali_cycle": 10
  },
  "model": {
    "name"  : "LSTM",
    "hidden": 40
  },
  "trainer": {
    "name"            : "RNN",
    "batch_size"      : 128,
    "train_init_frame": 200,
    "train_frame"     : 1000,
    "vali_frame"      : 100000,
    "chunk_sec"       : 0.5
  },
  "train_loss": {
    "name": "Mix",
    "elems": [
      {
        "name" : "ESR",
        "ratio": 0.75
      },
      {
        "name" : "DC",
        "ratio": 0.25
      }
    ]
  },
  "vali_loss": {
    "name": "Mix",
    "elems": [
      {
        "name" : "ESR",
        "ratio": 0.75
      },
      {
        "name" : "DC",
        "ratio": 0.25
      }
    ]
  },
  "optimizer": {
    "name"      : "Adam",
    "initial_lr": 0.005,
    "lr_decay"  : 0.0001
  },
  "scheduler": {
    "name"       : "Plateau",
    "lr_factor"  : 0.5,
    "lr_patience": 5
  },
  "train_data": [
    {
      "cond"        : [],
      "input_audio" : "data/input.wav",
      "output_audio": "data/output.wav"
    }
  ],
  "vali_data": [
    {
      "cond"        : [],
      "input_audio" : "data/input.wav",
      "output_audio": "data/output.wav"
    }
  ]
}
```
  - `general`: 일반적인 설정을 담고 있습니다.
    - `seed`: Python 내 각종 랜덤 함수들의 Seed를 설정합니다. 기본값은 `42` 입니다.
    - `device`: 학습에 사용할 Device (`'cuda'`, `'mps'`, `'cpu'` 중 선택) 를 설정합니다. 기본값은 해당 환경에서 가장 최적인 Device 입니다.
    - `deterministic`: 엄밀한 재현성을 위해 PyTorch가 결정론적 알고리즘을 사용하게 합니다. 학습 속도가 느려질 수 있습니다. 기본값은 `false` 입니다.
    - `compile`: 속도 향상을 위해 모델을 컴파일합니다. 기본값은 `true` 입니다.
    - `num_cond`: 모델의 조건부 파라미터 갯수 (모델에 따라 가능한 값이 다름) 를 설정합니다.
    - `sample_rate`: 오디오를 다루는 Sample Rate를 설정합니다. 기본값은 `48000` 입니다.
  - `train`: 학습 횟수에 대한 내용을 담고 있습니다.
    - `max_epoch`: 학습을 진행할 최대 Epoch를 설정합니다.
    - `vali_cycle`: Validation을 얼마나 자주 수행할지 설정합니다.
    - `early_stop`: 몇 회의 Validation 동안 Loss의 개선이 없을 때 조기 종료할지 설정합니다.
  - `model`: 사용할 모델에 대한 내용을 담고 있습니다.
    - `name`: 모델의 이름을 선택합니다. [`kirby-core/src/kirby_core/impl/model/__init__.py`](../kirby-core/src/kirby_core/impl/model/__init__.py)의 `_MODELS` 배열에서 선택할 수 있습니다.
    - 모델에 따라 요구하는 추가 필드는 각 구현체를 참고하세요.
  - `trainer`: 학습 방법에 대한 내용을 담고 있습니다.
    - `name`: 학습 방법의 이름을 선택합니다. [`kirby-core/src/kirby_core/impl/trainer/__init__.py`](../kirby-core/src/kirby_core/impl/trainer/__init__.py)의 `_TRAINERS` 배열에서 선택할 수 있습니다.
    - 학습 방법에 따라 요구하는 추가 필드는 각 구현체를 참고하세요.
  - `train_loss` 및 `vali_loss`: Training / Validation Loss에 대한 내용을 담고 있습니다.
    - `name`: Training / Validation Loss의 이름을 선택합니다. [`kirby-core/src/kirby_core/impl/loss/__init__.py`](../kirby-core/src/kirby_core/impl/loss/__init__.py)의 `_LOSSES` 배열에서 선택할 수 있습니다.
    - Loss에 따라 요구하는 추가 필드는 각 구현체를 참고하세요.
  - `optimizer`: Optimizer에 대한 내용을 담고 있습니다.
    - `name`: Optimizer의 이름을 선택합니다. [`kirby-core/src/kirby_core/impl/optimizer/__init__.py`](../kirby-core/src/kirby_core/impl/optimizer/__init__.py)의 `_OPTIMIZERS` 배열에서 선택할 수 있습니다.
    - Optimizer에 따라 요구하는 추가 필드는 각 구현체를 참고하세요.
  - `scheduler`: Scheduler에 대한 내용을 담고 있습니다.
    - `name`: Scheduler 이름을 선택합니다. [`kirby-core/src/kirby_core/impl/scheduler/__init__.py`](../kirby-core/src/kirby_core/impl/scheduler/__init__.py)의 `_SCHEDULERS` 배열에서 선택할 수 있습니다.
    - Scheduler에 따라 요구하는 추가 필드는 각 구현체를 참고하세요.
  - `train_data` 및 `vali_data`: Training / Validation Data에 대한 내용을 담고 있습니다.
    - `cond`: 해당 Data의 조건부 파라미터 값입니다.
    - `input_audio` / `output_audio`: Input / Output 오디오 파일 (*.wav* 형식) 의 경로 또는 *audio ID* (아래에서 설명) 입니다.

> 같은 내용을 담은 TOML / YAML 파일도 가능합니다.

### 실행하기
`kirby-cli` 디렉토리에서 다음의 명령을 실행합니다.
```sh
uv run training -c <training config file path>
```

정상적으로 실행되었다면, 다음의 내용들이 출력됩니다.
- 작성하지 않은 필드에 대한 기본값
- *training config file* 에 명시된 로컬 오디오 파일의 *audio ID*
  - 해당 파일을 *f32* 포맷으로 읽은 샘플 값에 기반한 해쉬 값입니다.
  - 처음 사용하는 파일이라면, 해당 파일이 W&B에 업로드됩니다.
  - 이후 학습에 해당 *audio ID* 를 사용할 수 있습니다.
- *training config file* 의 *training ID*
  - 해당 파일을 읽어 재귀적으로 구한 해쉬 값입니다.
  - 아래의 전처리 과정을 거치므로, 포맷 / 오디오 경로 / 필드 순서와 무관합니다.
    - 생략된 필드의 기본값 채우기
    - 각 key를 사전순으로 정렬
    - 로컬 오디오 파일의 경로를 *audio ID* 로 대체

> 같은 training ID로는 한 번만 학습할 수 있습니다. 다시 학습하려면 `-f` 옵션을 추가하거나, `seed` 값을 변경하세요.

## 개발 가이드
### 의존성 추가하기
`kirby-cli` 디렉토리에서 다음의 명령을 실행합니다.
```sh
uv add <dependency name>
```
