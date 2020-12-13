import functools
import math
import time
from datetime import datetime
from itertools import islice
from typing import *


def foreach(items: Iterable[Any], f: Callable[[Any], None]) -> None:
    if items is not None:
        for it in items:
            f(it)


def timestampstr() -> str:
    today = datetime.today()
    diff = today.replace(hour=0, minute=0, second=0) - today

    return '_'.join([
        today.strftime("%Y%m%d"),
        str(diff.seconds)
    ])


def clip(val, minval, maxval):
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


def minmax(values):
    min_val = values[0]
    max_val = min_val

    for val in islice(values, 1, None):
        if val < min_val:
            min_val = val
        elif val > max_val:
            max_val = val

    return min_val, max_val


def find_largest_divisor(x):
    sqrtx = int(math.sqrt(x))
    for i in range(2, sqrtx):
        if x % i == 0:
            return x // i
    return 1


def memoize(func, maxsize=128, typed=False):
    @functools.lru_cache(maxsize=maxsize, typed=typed)
    def func_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return func_wrapper


class TimeIt:

    def __init__(self, label=None):
        self.start = None
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if self.label:
            print(self.label)
        print(f'Execution time: {TimeIt.format_elapsed(elapsed)}')

    @staticmethod
    def format_elapsed(elapsed, ndigits=3):
        def _round(x):
            x = round(x, ndigits=ndigits)
            return int(x) if ndigits == 0 else x

        fmt = []

        def append_value(value, label, forced=False):
            if forced or value > 0:
                fmt.append(str(value) + label)

        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(int(minutes), 60)

        append_value(hours, 'h')
        append_value(minutes, 'm')
        append_value(_round(seconds), 's', forced=True)

        return ''.join(fmt)


# Adapted from
# https://stackoverflow.com/questions/43761004/fps-how-to-divide-count-by-time-function-to-determine-fps
class FpsCounter(object):

    def __init__(self, num_seconds=1):
        self.num_seconds = num_seconds

        self.counter = 0
        self.start_time = None

    def __enter__(self):
        self._reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.counter = 0
        self.start_time = None

    def update(self) -> int:
        self.counter += 1
        elapsed_time = time.time() - self.start_time

        fps = None
        if elapsed_time > self.num_seconds:
            fps = self.counter / elapsed_time
            self._reset()

        return fps

    def _reset(self):
        self.counter = 0
        self.start_time = time.time()


inverse_marks = {
    '': '',
    '"': '"',
    '\'': '\'',
    '`': '`',
    '(': ')',
    '<': '>',
    '[': ']',
    '{': '}',
}


def create_reversed(items, func=None):
    ret = []
    for item in items:
        if func:
            item = func(item)
        ret.insert(0, item)
    return ret


def quote(s, prefix='\''):
    try:
        suffix = ''.join(create_reversed(prefix, lambda x: inverse_marks[x]))
    except KeyError as e:
        raise ValueError(f'Invalid quotation mark: \'{e.args[0]}\'')
    return prefix + s + suffix
