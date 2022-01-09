import jax.numpy as jnp
import jax
import haiku as hk
from typing import Callable


def cartesian_product(a: jnp.ndarray, b: jnp.ndarray):
    if len(a.shape) != 2:
        raise ValueError(f'Shape a needs to be rank 2, but was {a.shape}')
    if len(b.shape) != 2:
        raise ValueError(f'Shape b needs to be rank 2, but was {b.shape}')

    a_repeated = jnp.repeat(a, b.shape[0], axis=0)  # Each element is repeated
    b_tiled = jnp.tile(b, (a.shape[0], 1))  # The array as a whole is repeated
    product = jnp.concatenate((a_repeated, b_tiled), axis=1)
    return product


def bcast_local_devices(value):
    devices = jax.local_devices()
    return jax.tree_map(
        lambda v: jax.api.device_put_sharded(len(devices) * [v], devices), value)


def get_first(xs):
    return jax.tree_map(lambda x: x[0], xs)


def broadcast_minor(x, shape):
    assert x.shape == shape[:len(x.shape)]
    new_dims = list(x.shape) + ([1] * (len(shape) - len(x.shape)))
    # TODO explicit broadcast may not be necessary
    return jnp.broadcast_to(x.reshape(new_dims), shape)


def freeze_split(is_frozen: Callable, func: Callable):
    def call(params, *args, **kwargs):
        frozen, unfrozen = hk.data_structures.partition(is_frozen, params)
        return func(unfrozen, frozen, *args, **kwargs)
    return call


def freeze_merge(func: Callable):
    def call(unfrozen, frozen, *args, **kwargs):
        params = hk.data_structures.merge(unfrozen, frozen)
        return func(params, *args, **kwargs)
    return call
