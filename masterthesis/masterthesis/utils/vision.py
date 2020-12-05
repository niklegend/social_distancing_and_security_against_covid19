from typing import *

import torch
from torch.utils.data import DataLoader


def compute_mean_std(loader: DataLoader, depth_channel_idx: Optional[int] = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    channels_sum: torch.Tensor = torch.empty(0)
    channels_squared_sum: torch.Tensor = torch.empty(0)
    num_batches = 0

    dims = None
    for data, _ in loader:
        if dims is None:
            dims = [i for i in range(data.ndimension()) if i != depth_channel_idx]
            size = data.size(depth_channel_idx)
            channels_sum = torch.zeros(size)
            channels_squared_sum = torch.zeros(size)

        channels_sum += torch.mean(data, dim=dims)
        channels_squared_sum += torch.mean(data ** 2, dim=dims)
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
