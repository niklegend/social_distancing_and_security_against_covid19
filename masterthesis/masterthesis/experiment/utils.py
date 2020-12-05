from typing import *

from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook


def looper(loader: DataLoader, progressbar: bool, notebook: bool) -> Union[Iterable, tqdm]:
    loop = loader
    if progressbar:
        total = len(loader)
        if notebook:
            loop = tqdm_notebook(loop, total=total, leave=True)
        else:
            loop = tqdm(loop, total=total, leave=True)
    return loop


def phase_key(phase: str, key: str) -> str:
    return '%s_%s' % (phase, key)
