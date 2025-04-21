# *kirby-core*
*kirby-core* 는 Training / Inference를 위한 Python 라이브러리입니다.

## 요구사항
- [UV](https://docs.astral.sh/uv)

## 개발 가이드
### 빌드하기
`kirby-core` 디렉토리에서 다음의 명령을 실행합니다.
```sh
uv build
```

### 의존성 추가하기
`kirby-core` 디렉토리에서 다음의 명령을 실행합니다.
```sh
uv add <dependency name>
```

### Model / Trainer / Loss / Optimizer / Scheduler 추가하기
[`kirby-core/src/kirby_core/impl`](src/kirby_core/impl) 내의 다른 구현체들을 참고해 주세요.
  - Model: `ModelBase` 및 `torch.nn.Module` 을 상속받아 [필요 메소드](src/kirby_core/protocol/impl/model_base.py)를 구현하고, [`kirby-core/src/kirby_core/impl/model/__init__.py`](src/kirby_core/impl/model/__init__.py)의 `_MODELS` 에 등록해 주세요.
  - Trainer: `TrainerBase` 를 상속받아 [필요 메소드](src/kirby_core/protocol/impl/trainer_base.py)를 구현하고, [`kirby-core/src/kirby_core/impl/trainer/__init__.py`](src/kirby_core/impl/trainer/__init__.py)의 `_TRAINERS` 에 등록해 주세요.
  - Loss: `LossBase` 및 `torch.nn.Module` 을 상속받아 [필요 메소드](src/kirby_core/protocol/impl/loss_base.py)를 구현하고, [`kirby-core/src/kirby_core/impl/loss/__init__.py`](src/kirby_core/impl/loss/__init__.py)의 `_LOSSES` 에 등록해 주세요.
  - Optimier: `OptimizerBase` 및 `torch.optim.Optimizer` 를 상속받아 [필요 메소드](src/kirby_core/protocol/impl/optimizer_base.py)를 구현하고, [`kirby-core/src/kirby_core/impl/optimizer/__init__.py`](src/kirby_core/impl/optimizer/__init__.py)의 `_OPTIMIZERS` 에 등록해 주세요.
  - Scheduler: `SchedulerBase` 및 `torch.optim.lr_scheduler.LRScheduler` 를 상속받아 [필요 메소드](src/kirby_core/protocol/impl/scheduler_base.py)를 구현하고, [`kirby-core/src/kirby_core/impl/scheduler/__init__.py`](src/kirby_core/impl/scheduler/__init__.py)의 `_SCHEDULERS` 에 등록해 주세요.

### 캐시 삭제하기
> *kirby-core* 를 수정한 뒤 *kirby-cli* 에서 해당 변경 사항을 사용하기 위해 필요한 단계입니다.
> 실행 권한이 모자라다면 `chmod +x util.sh` 로 권한을 부여할 수 있습니다.

`kirby-core` 또는 `kirby-cli` 디렉토리에서 다음의 명령을 실행합니다.
```sh
../util.sh --uvclean
```

또는 프로젝트 루트에서 다음의 명령을 실행합니다.
```sh
./util.sh --uvclean
```
