import os
import random
import numpy as np
import torch

def _set_seed(seed: int) -> None:
    os.environ[ 'PYTHONHASHSEED' ] = str(seed)
    random   .seed       (seed)
    np.random.seed       (seed)
    torch    .manual_seed(seed)

def _set_deterministic(device: str) -> None:
    torch.use_deterministic_algorithms(True)

    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # See 'WARNING' in https://pytorch.org/docs/stable/generated/torch.nn.RNN
        cuda_version = float(torch.version.cuda)

        if cuda_version == 10.1:
            os.environ[ 'CUDA_LAUNCH_BLOCKING'    ] = '1'
        else:
            os.environ[ 'CUBLAS_WORKSPACE_CONFIG' ] = ':16:8'
