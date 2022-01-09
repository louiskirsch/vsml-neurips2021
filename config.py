import logging
import argparse
import ast
import functools

from contextlib import contextmanager
from typing import Any, Union, Sequence


GLOBAL_CONFIG = dict()


class DotDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def expand_dot_items(inp: dict):
    new_subdicts = []
    for k, v in list(inp.items()):
        if '.' in k:
            prefix, postfix = k.split('.', maxsplit=1)
            if prefix in inp:
                inp[prefix][postfix] = v
            else:
                inp[prefix] = DotDict(((postfix, v),))
                new_subdicts.append(inp[prefix])
            del inp[k]
    for sub in new_subdicts:
        expand_dot_items(sub)
    return inp


def flatten_dot_items(inp: dict):
    for k, v in list(inp.items()):
        if isinstance(v, dict):
            v = flatten_dot_items(v)
            inp.update((f'{k}.{k2}', v2) for k2, v2 in v.items())
            del inp[k]
    return inp


class WandbMockConfig(DotDict):

    def update(self, params, allow_val_change=False, **kwargs):
        super().update(params, **kwargs)


class WandbMockSummary:

    def update(self, *args):
        logging.debug(f'Non grouped run, can not write summary {args}')

    def __setattr__(self, key, value):
        logging.debug(f'Non grouped run, can not write summary {key} = {value}')

    def __setitem__(self, key, value):
        logging.debug(f'Non grouped run, can not write summary {key} = {value}')


_PARSE_PREFIXES = ('[', '(', '-', '+', 'False', 'True') + tuple(str(n) for n in range(10))


def _parse_value(value: str):
    if value.startswith(_PARSE_PREFIXES):
        return ast.literal_eval(value)
    return value


class StoreDictKeyPair(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = _parse_value(v)
            setattr(namespace, self.dest, my_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', action='store', nargs='+', type=str)
    parser.add_argument('--job_type', action='store', type=str)
    parser.add_argument('--array', action='store', type=str)
    parser.add_argument('--subset', action='store', nargs='+', type=int)
    parser.add_argument('--config_files', action='store', nargs='+', type=str)
    parser.add_argument('--config', action=StoreDictKeyPair, nargs='+', metavar='KEY=VAL')
    return parser.parse_args()


def resolve_config_ns(ns: Union[str, Sequence]):
    sub_config = GLOBAL_CONFIG
    if hasattr(ns, 'split'):
        ns = ns.split('.')
    for sub_key in ns:
        try:
            sub_config = sub_config[sub_key]
        except KeyError:
            raise KeyError(f'No key {sub_key} in keys {sub_config.keys()} '
                           f'when resolving {ns}')
    return sub_config


def configurable(key: str):
    def decorator(fn):
        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            sub_config = resolve_config_ns(key)
            new_kwargs = sub_config.copy()
            new_kwargs.update(kwargs)
            return fn(*args, **new_kwargs)
        return decorated
    return decorator


@contextmanager
def change_config(key: str, value: Any):
    parts = key.split('.')
    cfg = resolve_config_ns(parts[:-1])
    key = parts[-1]
    old_value = cfg[key]
    cfg[key] = value
    yield
    cfg[key] = old_value
