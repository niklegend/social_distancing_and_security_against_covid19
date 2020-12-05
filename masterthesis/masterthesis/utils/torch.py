import os
import random
from typing import *

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def deterministic_behavior(seed: Optional[int] = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def module_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False


def parallelize(x: Union[nn.Module, torch.Tensor]) -> Union[nn.Module, torch.Tensor]:
    ismodule = isinstance(x, nn.Module)
    if not ismodule and not isinstance(x, torch.Tensor):
        raise RuntimeError(f'Object of type \'{x.__class__.__name__}\' cannot be parallelized.')

    if ismodule and torch.cuda.device_count() > 0:
        return nn.DataParallel(cast(nn.Module, x))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return x.to(device)


class Checkpoint(object):

    @staticmethod
    def load(filepath: Union[str, os.PathLike]):
        return Checkpoint(torch.load(filepath))

    def __init__(self, checkpoint: Dict[str, Any] = None):
        if checkpoint is None:
            checkpoint = dict()
        self.checkpoint = checkpoint

    def get(self, key: str) -> Any:
        return self.checkpoint[key]

    def put(self, key: str, value: Any):
        if value is not None:
            has_state_dict = isinstance(value, nn.Module) or isinstance(value, Optimizer)
            if has_state_dict:
                self.checkpoint[key] = value.state_dict()
            else:
                self.checkpoint[key] = value

    def save(self, filepath: Union[str, os.PathLike]):
        torch.save(self.checkpoint, filepath)
