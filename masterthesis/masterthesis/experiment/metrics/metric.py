from typing import Optional

import torch

from ..utils import phase_key


class Metric(object):

    def __init__(self, name):
        self.name = name

    def reset(self) -> None:
        raise NotImplementedError()

    def update(self, targets: torch.Tensor, outputs: torch.Tensor) -> None:
        raise NotImplementedError()

    def result(self) -> float:
        raise NotImplementedError()

    def as_dict(self, tag: Optional[str] = None) -> dict:
        return {Metric.key(self.name, tag): self.result()}

    def to_str(self, tag: Optional[str] = None) -> str:
        return Metric.str(self.name, self.result(), tag)

    def __repr__(self):
        return self.to_str()

    @staticmethod
    def key(name: str, tag: Optional[str] = None):
        return name if tag is None else phase_key(tag, name)

    @staticmethod
    def str(name: str, value: float, tag: Optional[str] = None):
        return '%s: %.4f' % (Metric.key(name, tag), value)
