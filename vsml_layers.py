import functools
import jax
import jax.numpy as jnp
import haiku as hk
import jaxutil


def dense(sub_rnn, reduce_fn, fwd_msg, bwd_msg, state):
    """Dense VSML layer.
    Args:
        sub_rnn: Rnn taking as inputs fwd_msg, bwd_msg, state
        reduce_fn: A function used to merge messages
        fwd_msg: Shape [in_channel, msg_size]
        bwd_msg: Shape [out_channel, msg_size]
        state: Shape [ic, oc, slow_size]
    """
    batched = hk.vmap(sub_rnn, in_axes=(None, 0, 0))
    batched = hk.vmap(batched, in_axes=(0, None, 0))
    fwd_msg, bwd_msg, state = batched(fwd_msg, bwd_msg, state)
    fwd_msg = reduce_fn(fwd_msg, axis=0)
    bwd_msg = reduce_fn(bwd_msg, axis=1)
    return fwd_msg, bwd_msg, state


def conv(base_func, reduce_fn, fwd_msg, bwd_msg, state, stride):
    # TODO generalize to arbitrary state pytrees
    kwidth = state[0].shape[0]
    pad_fwd_msg = jnp.pad(fwd_msg,
                          ((kwidth // 2, kwidth // 2),)
                          + ((0, 0),) * (fwd_msg.ndim - 1))
    width = fwd_msg.shape[0]
    pad_width = pad_fwd_msg.shape[0]

    # TODO This is inefficient
    # Shape [pad_width // stride, kwidth, in_channel, msg_size]
    gathered_fwd_msg = pad_fwd_msg[(jnp.arange(kwidth)[None]
                                   + jnp.arange(pad_width - kwidth + 1,
                                                step=stride)[:, None])]

    batched_kwidth = hk.vmap(base_func, in_axes=(0, None, 0))
    batched = hk.vmap(batched_kwidth, in_axes=(0, 0, None))

    fwd_msg, bwd_msg, state = batched(gathered_fwd_msg, bwd_msg, state)

    state = jax.tree_map(lambda s: reduce_fn(s, axis=0), state)

    # Reduce over kernel
    fwd_msg = reduce_fn(fwd_msg, axis=1)

    # Construct bwd_msg
    # TODO striding currently inefficient
    idx0 = jnp.arange(width)[:, None] + jnp.arange(kwidth)[None] - 1
    idx1 = jnp.broadcast_to(jnp.flip(jnp.arange(kwidth)[None, :]), (width, kwidth))
    msg = bwd_msg[(jnp.clip(idx0, 0, width - 1) // stride, idx1)]
    mask = jnp.logical_and(jnp.logical_and(idx0 >= 0, idx0 < width // stride),
                           idx0 % stride == 0).astype(jnp.int32)
    mask = jaxutil.broadcast_minor(mask, msg.shape)
    bwd_msg = reduce_fn(msg * mask, axis=1)

    return fwd_msg, bwd_msg, state


def conv1d(sub_rnn, reduce_fn, fwd_msg, bwd_msg, state, stride=1):
    """1D conv with channels. Currently only SAME padding.
    Args:
        sub_rnn: Rnn taking as inputs fwd_msg, bwd_msg, state
        reduce_fn: A function used to merge messages
        fwd_msg: Shape [width, in_channel, msg_size]
        bwd_msg: Shape [width // stride, out_channel, msg_size]
        state: Shape [kwidth, ic, oc, slow_size]
    """
    base_func = functools.partial(dense, sub_rnn, reduce_fn)
    return conv(base_func, reduce_fn, fwd_msg, bwd_msg, state, stride)


def conv2d(sub_rnn, reduce_fn, fwd_msg, bwd_msg, state, stride=1):
    """2D conv with channels. Currently only SAME padding.
    Args:
        sub_rnn: Rnn taking as inputs fwd_msg, bwd_msg, state
        reduce_fn: A function used to merge messages
        fwd_msg: Shape [height, width, in_channel, msg_size]
        bwd_msg: Shape [height // stride, width // stride, out_channel, msg_size]
        state: Shape [kheight, kwidth, ic, oc, slow_size]
    """
    base_func = functools.partial(conv1d, sub_rnn, reduce_fn, stride=stride)
    return conv(base_func, reduce_fn, fwd_msg, bwd_msg, state, stride)
