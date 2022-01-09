import wandb
import time
from contextlib import contextmanager
import numpy as np


@contextmanager
def timeit(key):
    start_time = time.time()
    yield
    wandb.summary[key] = time.time() - start_time


class SummaryItem:

    def __init__(self, data, verbosity=0, delete=False):
        self.data = data
        self.verbosity = verbosity
        self.delete = delete


class SummaryLog:
    properties = {'prefix', '_data', 'children'}

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.children = []
        self._data = {}

    def __setattr__(self, key, value):
        if key in self.properties:
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getattr__(self, key):
        return self._data[key]

    def clear(self):
        self._data.clear()

    def update(self, update_dict):
        self._data.update(update_dict)

    def _convert_value(self, value, verbosity):
        if isinstance(value, SummaryItem):
            if verbosity >= value.verbosity:
                return self._convert_value(value.data, verbosity)
            return None
        if callable(value):
            value = value()
        return value

    def create(self, verbosity):
        summary = {
            f'{self.prefix}.{k}': v_conv
            for k, v_conv in ((k, self._convert_value(v, verbosity)) for k, v in self._data.items())
            if v_conv is not None
        }
        for key in list(k for k, v in self._data.items() if isinstance(v, SummaryItem) and v.delete):
            del self._data[key]
        for s in self.children:
            summary.update({f'{self.prefix}.{k}': v for k, v in s(verbosity).items()})
        return summary

    def __call__(self, verbosity=0):
        return self.create(verbosity)

    def add(self, summary_log):
        self.children.append(summary_log)

    @contextmanager
    def timeit(self, key, increment=False):
        start_time = time.time()
        yield
        prev = self._data[key] if key in self._data and increment else 0
        self._data[key] = time.time() - start_time + prev

    def hist(self, key, tensor, verbosity=0):
        tensor = tensor.detach()
        self[key] = SummaryItem(lambda: wandb.Histogram(tensor.cpu()), verbosity)


def merge_summaries(*summary_logs: SummaryLog):
    def merged_summary_creator(verbosity=0):
        summary = {}
        for s in summary_logs:
            summary.update(s(verbosity))
        return summary
    return merged_summary_creator


def flatten_batched_array(array: np.ndarray) -> np.ndarray:
    assert array.ndim == 3
    b, r, c = array.shape
    return np.column_stack((np.repeat(np.arange(b), r),
                            array.reshape(b * r, c)))

