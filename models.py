import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import itertools
import functools
import optax

from typing import Tuple, Callable, List, Optional, Iterable, Any
from config import configurable
import vsml_layers
import recurrent


LayerSpec = Any


@chex.dataclass
class DenseSpec:
    in_size: int = -1
    out_size: int = -1


@chex.dataclass
class ConvSpec:
    in_width: int = -1
    in_height: int = -1
    in_channels: int = -1
    out_width: int = -1
    out_height: int = -1
    out_channels: int = -1
    kernel_size: int = 3
    stride: int = 1

    @property
    def out_size(self):
        return self.out_width * self.out_height * self.out_channels


SPEC_TYPES = {
    'dense': DenseSpec,
    'conv': ConvSpec,
}


def create_spec(cfg):
    cfg = cfg.copy()
    constr = SPEC_TYPES[cfg.pop('type')]
    return constr(**cfg)


def complete_specs(layer_specs, input_shape, output_size):
    last_idx = len(layer_specs) - 1
    for i, spec in enumerate(layer_specs):
        prev_spec = layer_specs[i - 1] if i > 0 else None
        if isinstance(spec, ConvSpec):
            if i == 0:
                spec.in_height, spec.in_width, spec.in_channels = input_shape
            else:
                spec.in_height = prev_spec.out_height
                spec.in_width = prev_spec.out_width
                spec.in_channels = prev_spec.out_channels
            spec.out_height = int(np.ceil(spec.in_height / spec.stride))
            spec.out_width = int(np.ceil(spec.in_width / spec.stride))
        elif isinstance(spec, DenseSpec):
            if i == 0:
                spec.in_size = np.prod(input_shape)
            else:
                spec.in_size = prev_spec.out_size
            if i == last_idx:
                spec.out_size = output_size
    return layer_specs


@chex.dataclass
class LayerState:
    lstm_state: hk.LSTMState
    incoming_fwd_msg: jnp.ndarray
    incoming_bwd_msg: jnp.ndarray


@configurable('model.sub_rnn')
class SubRNN(hk.Module):

    def __init__(self, slow_size: int, msg_size: int, init_rand_proportion: float, layer_norm: bool):
        super().__init__()
        self._lstm = recurrent.CustomLSTM(slow_size)
        self._fwd_messenger = hk.Linear(msg_size)
        self._bwd_messenger = hk.Linear(msg_size)
        if layer_norm:
            self._fwd_layer_norm = hk.LayerNorm((-1,), create_scale=True, create_offset=True)
            self._bwd_layer_norm = hk.LayerNorm((-1,), create_scale=True, create_offset=True)
        self.msg_size = msg_size
        self._init_rand_proportion = init_rand_proportion
        self._use_layer_norm = layer_norm

    def __call__(self, fwd_msg: jnp.ndarray, bwd_msg: jnp.ndarray,
                 lstm_state: hk.LSTMState) -> Tuple[jnp.ndarray, jnp.ndarray, hk.LSTMState]:
        inputs = jnp.concatenate([fwd_msg, bwd_msg], axis=-1)
        outputs, lstm_state = self._lstm(inputs, lstm_state)
        fwd_msg = self._fwd_messenger(outputs)
        bwd_msg = self._bwd_messenger(outputs)
        if self._use_layer_norm:
            fwd_msg = self._fwd_layer_norm(fwd_msg)
            bwd_msg = self._bwd_layer_norm(bwd_msg)
        return fwd_msg, bwd_msg, lstm_state

    def initial_state(self, layer_spec: LayerSpec) -> hk.LSTMState:
        if isinstance(layer_spec, DenseSpec):
            shape = (layer_spec.in_size, layer_spec.out_size)
        elif isinstance(layer_spec, ConvSpec):
            shape = (layer_spec.kernel_size,
                     layer_spec.kernel_size,
                     layer_spec.in_channels,
                     layer_spec.out_channels)
        return self._lstm.initial_vsml_state(shape, self._init_rand_proportion)


@configurable('model.vsml_rnn')
class VSMLRNN(hk.Module):

    def __init__(self, layer_specs: List[LayerSpec], num_micro_ticks: int,
                 loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 tanh_bound: float, output_idx: int, backward_pass: bool,
                 separate_backward_rnn: bool, feed_label: bool, layerwise_rnns: bool):
        super().__init__()
        self._layer_specs = layer_specs
        self._num_micro_ticks = num_micro_ticks
        self._tanh_bound = tanh_bound
        if layerwise_rnns:
            self._sub_rnns = [SubRNN() for _ in layer_specs]
        else:
            self._sub_rnns = [SubRNN()] * len(layer_specs)
        self._loss_func = loss_func
        self._loss_func_grad = jax.grad(loss_func)
        self._backward_pass = backward_pass
        self._feed_label = feed_label
        self._batched_tick = hk.vmap(
            functools.partial(self._tick, self._sub_rnns, reverse=False))
        if backward_pass:
            if separate_backward_rnn:
                if layerwise_rnns:
                    self._back_sub_rnns = [SubRNN() for _ in layer_specs]
                else:
                    self._back_sub_rnns = [SubRNN()] * len(layer_specs)
            else:
                self._back_sub_rnns = self._sub_rnns
            self._reverse_batched_tick = hk.vmap(
                functools.partial(self._tick, self._back_sub_rnns, reverse=True))
        self._output_idx = output_idx

    def _tick(self, sub_rnns, layer_states: List[LayerState], error: jnp.ndarray,
              inp: jnp.ndarray, reverse=False) -> Tuple[List[LayerState], jnp.ndarray]:
        if isinstance(self._layer_specs[0], DenseSpec):
            inp = inp.flatten()
        sub_rnn = sub_rnns[0]
        fwd_msg = jnp.pad(inp[..., None], (*[(0, 0)] * inp.ndim, (0, sub_rnn.msg_size - 1)))
        bwd_msg = jnp.pad(error, ((0, 0), (0, sub_rnn.msg_size - 2)))
        layer_states[0].incoming_fwd_msg = fwd_msg
        layer_states[-1].incoming_bwd_msg = bwd_msg
        output = None

        iterable = list(enumerate(zip(layer_states, self._layer_specs, sub_rnns)))
        if reverse:
            iterable = list(reversed(iterable))
        for i, (ls, lspec, srnn) in iterable:
            lstm_state, fwd_msg, bwd_msg = (ls.lstm_state,
                                            ls.incoming_fwd_msg,
                                            ls.incoming_bwd_msg)
            for _ in range(self._num_micro_ticks):
                args = (srnn, jnp.mean, fwd_msg, bwd_msg, lstm_state)
                if isinstance(lspec, DenseSpec):
                    out = vsml_layers.dense(*args)
                elif isinstance(lspec, ConvSpec):
                    out = vsml_layers.conv2d(*args, stride=lspec.stride)
                else:
                    raise ValueError(f'Invalid layer {lspec}')
                new_fwd_msg, new_bwd_msg, lstm_state = out
            ls.lstm_state = lstm_state
            if i > 0:
                shape = layer_states[i - 1].incoming_bwd_msg.shape
                layer_states[i - 1].incoming_bwd_msg = new_bwd_msg.reshape(shape)
            if i < len(layer_states) - 1:
                shape = layer_states[i + 1].incoming_fwd_msg.shape
                layer_states[i + 1].incoming_fwd_msg = new_fwd_msg.reshape(shape)
            else:
                output = new_fwd_msg[:, self._output_idx]
                if self._tanh_bound:
                    output = jnp.tanh(output / self._tanh_bound) * self._tanh_bound

        return layer_states, output

    def _create_layer_state(self, spec: LayerSpec) -> LayerState:
        sub_rnn = self._sub_rnns[0]
        lstm_state = sub_rnn.initial_state(spec)
        msg_size = sub_rnn.msg_size
        new_msg = functools.partial(jnp.zeros, dtype=lstm_state.hidden.dtype)
        if isinstance(spec, DenseSpec):
            incoming_fwd_msg = new_msg((spec.in_size, msg_size))
            incoming_bwd_msg = new_msg((spec.out_size, msg_size))
        elif isinstance(spec, ConvSpec):
            incoming_fwd_msg = new_msg((spec.in_height, spec.in_width,
                                        spec.in_channels, msg_size))
            incoming_bwd_msg = new_msg((spec.out_height, spec.out_width,
                                        spec.out_channels, msg_size))

        return LayerState(lstm_state=lstm_state,
                          incoming_fwd_msg=incoming_fwd_msg,
                          incoming_bwd_msg=incoming_bwd_msg)

    def _merge_layer_states(self, layer_states: List[LayerState]) -> List[LayerState]:
        def merge(state):
            s1, s2 = jnp.split(state, [state.shape[-1] // 2], axis=-1)
            merged_s1 = jnp.mean(s1, axis=0, keepdims=True)
            new_s1 = jnp.broadcast_to(merged_s1, s1.shape)
            return jnp.concatenate((new_s1, s2), axis=-1)
        for ls in layer_states:
            ls.lstm_state = jax.tree_map(merge, ls.lstm_state)
        return layer_states

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        layer_states = [self._create_layer_state(spec) for spec in self._layer_specs]
        layer_states = jax.tree_map(lambda ls: jnp.stack([ls] * inputs.shape[1]),
                                    layer_states)
        init_error = layer_states[-1].incoming_bwd_msg[..., :2]

        def scan_tick(carry, x):
            layer_states, error = carry
            inp, label = x
            if inp.shape[0] > 1:
                layer_states = self._merge_layer_states(layer_states)

            new_layer_states, out = self._batched_tick(layer_states, error, inp)
            new_error = self._loss_func_grad(out, label)
            label_input = label if self._feed_label else jnp.zeros_like(label)
            new_error = jnp.stack([new_error, label_input], axis=-1)

            if self._backward_pass:
                new_layer_states, _ = self._reverse_batched_tick(new_layer_states, new_error, inp)
                new_error = jnp.zeros_like(new_error)

            return (new_layer_states, new_error), out
        _, outputs = hk.scan(scan_tick, (layer_states, init_error),
                             (inputs, labels))
        return outputs


@configurable('model.meta_rnn')
class MetaRNN(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, num_micro_ticks: int, slow_size: int,
                 tanh_bound: float, use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(64, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(64, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(64, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._num_micro_ticks = num_micro_ticks
        self._tanh_bound = tanh_bound
        self._loss_func_grad = jax.grad(loss_func)
        self._lstm = hk.LSTM(slow_size)
        self._output_proj = hk.Linear(output_size)

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        output_size = self._output_proj.output_size
        lstm_state = self._lstm.initial_state(batch_size)
        init_error = jnp.zeros((batch_size, output_size))
        init_label = jnp.zeros((batch_size, output_size))

        # TODO merge states for batch_size > 1
        def scan_tick(carry, x):
            lstm_state, error, prev_label = carry
            inp, label = x
            if self._conv is not None:
                inp = self._conv(inp)
            inp = hk.Flatten(preserve_dims=1)(inp)
            inputs = jnp.concatenate([inp, error, prev_label], axis=-1)
            for _ in range(self._num_micro_ticks):
                out, lstm_state = self._lstm(inputs, lstm_state)
            out = self._output_proj(out)
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)
            return (lstm_state, new_error, label), out
        _, outputs = hk.scan(scan_tick, (lstm_state, init_error, init_label),
                             (inputs, labels))
        return outputs


@configurable('model.sgd')
class SGD(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, num_layers: int, hidden_size: int,
                 tanh_bound: float, optimizer: str, lr: float, use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._tanh_bound = tanh_bound
        self._loss_func = loss_func
        self._grad_func = jax.grad(self._loss, has_aux=True)
        self._network = functools.partial(self._network,
                                          output_size=output_size,
                                          num_layers=num_layers,
                                          hidden_size=hidden_size)
        self._network = hk.without_apply_rng(hk.transform(self._network))
        self._opt = getattr(optax, optimizer)(lr)

    def _network(self, x: jnp.ndarray, output_size, num_layers, hidden_size):
        if self._conv is not None:
            x = self._conv(x)
        x = hk.Flatten(preserve_dims=1)(x)
        for _ in range(num_layers - 1):
            x = hk.Linear(hidden_size)(x)
            x = jnp.tanh(x)
        x = hk.Linear(output_size)(x)
        if self._tanh_bound:
            x = jnp.tanh(x / self._tanh_bound) * self._tanh_bound
        return x

    def _loss(self, params, x, labels):
        logits = self._network.apply(params, x)
        loss = self._loss_func(logits, labels)
        return loss, logits

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        # TODO seed from outside
        rng = jax.random.PRNGKey(22)

        dummy_inp = inputs[0]
        params = self._network.init(rng, dummy_inp)
        opt_state = self._opt.init(params)

        def scan_tick(carry, x):
            params, opt_state = carry
            grads, out = self._grad_func(params, *x)
            updates, opt_state = self._opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), out
        _, outputs = jax.lax.scan(scan_tick, (params, opt_state), (inputs, labels))
        return outputs


class HebbianLinear(hk.Module):

    def __init__(self, output_size: int, use_oja: bool = False,
                 with_bias: bool = True,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 name: Optional[str] = None,):
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        if use_oja:
            self._fw_update = self._oja
        else:
            self._fw_update = self._hebb
        # Dim 1
        self._fw_update = jax.vmap(self._fw_update, in_axes=[0, None, None, 0])
        # Dim 2
        self._fw_update = jax.vmap(self._fw_update, in_axes=[0, None, 0, None])
        # Batch axis
        self._fw_update = jax.vmap(self._fw_update, in_axes=[None, None, 0, 0])

    def __call__(self, inputs: jnp.ndarray,
                 fast_weights: Optional[jnp.ndarray]) -> jnp.ndarray:
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        coeff = hk.get_parameter('coeff', [input_size, output_size], dtype,
                                 init=hk.initializers.Constant(0.01))
        fw_lr = hk.get_parameter('fw_lr', [], dtype,
                                 init=hk.initializers.Constant(-4.5))
        fw_lr = jax.nn.sigmoid(fw_lr)

        if fast_weights is None:
            fast_weights = jnp.zeros_like(w)
        out = jnp.dot(inputs, w + coeff * fast_weights)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        # Generate new fast weights
        # TODO make softmax optional
        new_fast_weights = self._fw_update(fast_weights, fw_lr, inputs,
                                           jax.nn.softmax(out))
        # Reduce batch axis
        new_fast_weights = jnp.mean(new_fast_weights, axis=0)

        return out, new_fast_weights

    def _hebb(self, fw, fw_lr, x, y) -> jnp.ndarray:
        return (1 - fw_lr) * fw + fw_lr * x * y

    def _oja(self, fw, fw_lr, x, y) -> jnp.ndarray:
        return fw + fw_lr * y * (x - y * fw)


@configurable('model.hebbian_fw')
class HebbianFW(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 input_shape: int, output_size: int, use_oja: bool, tanh_bound: float,
                 use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._layers = [
            HebbianLinear(output_size, use_oja)
        ]
        self.output_size = output_size
        self._tanh_bound = tanh_bound
        self._loss_func = loss_func
        self._loss_func_grad = jax.grad(loss_func)
        # Create parameters
        aux = [jnp.zeros([1, output_size])] * 2
        self._eval_layers(jnp.zeros([1, *input_shape]), itertools.repeat(None), aux)

    def _eval_layers(self, inputs: jnp.ndarray, fast_weights: Iterable[jnp.ndarray], aux):
        x = inputs
        if self._conv is not None:
            x = self._conv(x)
        x = hk.Flatten(preserve_dims=1)(x)
        x = jnp.concatenate([x, *aux], axis=-1)
        fws_out = []
        for layer, fws in zip(self._layers, fast_weights):
            x, new_fws = layer(x, fws)
            fws_out.append(new_fws)
        return x, fws_out

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        hebb_state = [jnp.zeros([layer.input_size, layer.output_size])
                      for layer in self._layers]
        init_error = jnp.zeros((batch_size, self.output_size))

        # TODO merge states for batch_size > 1
        def scan_tick(carry, x):
            hebb_state, error = carry
            inp, label = x

            aux = [jnp.zeros_like(error), jnp.zeros_like(label)]
            out, _ = self._eval_layers(inp, hebb_state, aux)
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)

            aux = [new_error, label]
            _, hebb_state = self._eval_layers(inp, hebb_state, aux)

            return (hebb_state, new_error), out
        _, outputs = hk.scan(scan_tick, (hebb_state, init_error),
                             (inputs, labels))
        return outputs


@configurable('model.fwp')
class FWP(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, fast_size: int, tanh_bound: float):
        super().__init__()
        self._tanh_bound = tanh_bound
        self._fast_size = fast_size
        self._output_size = output_size
        self._fast_shape = (fast_size, output_size)

        self._loss_func_grad = jax.grad(loss_func)
        size = 2 * fast_size + output_size + 1
        self._slow_net = hk.Linear(size)

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        fast_state = hk.initializers.VarianceScaling()(self._fast_shape, jnp.float32)
        init_carry = jnp.zeros((3, batch_size, self._output_size))

        def scan_tick(carry, x):
            fast_state, error, prev_label, prev_out = carry
            inp, label = x
            inp = hk.Flatten(preserve_dims=1)(inp)
            inputs = jnp.concatenate([inp, prev_out, error, prev_label], axis=-1)
            split_indices = np.cumsum([self._fast_size, self._output_size,
                                       self._fast_size])
            k, v, q, beta = jnp.split(self._slow_net(inputs), split_indices, axis=-1)
            beta = jax.nn.sigmoid(beta)
            prev_v = k @ fast_state
            fast_state = fast_state + k.T @ (beta * (v - prev_v))
            out = q @ fast_state
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)
            return (fast_state, new_error, label, out), out

        _, outputs = hk.scan(scan_tick, (fast_state, *init_carry), (inputs, labels))
        return outputs


@configurable('model.fw_memory')
class FWMemory(hk.Module):

    def __init__(self, loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 output_size: int, slow_size: int, tanh_bound: float, memory_size: int, use_conv: bool):
        super().__init__()
        if use_conv:
            self._conv = hk.Sequential([
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.relu,
                hk.Conv2D(8, 3, 2, padding='SAME'),
                jax.nn.tanh,
            ])
        else:
            self._conv = None
        self._tanh_bound = tanh_bound
        self._memory_size = memory_size
        self._loss_func_grad = jax.grad(loss_func)
        self._lstm = hk.LSTM(slow_size)
        self._output_proj = hk.Linear(output_size)
        self._write_head = hk.Linear(3 * memory_size + 1)
        self._read_head = hk.Linear(2 * memory_size)
        self._read_proj = hk.Linear(slow_size)
        self._layer_norm = hk.LayerNorm(-1, False, False)

    def __call__(self, inputs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        batch_size = inputs.shape[1]
        output_size = self._output_proj.output_size
        lstm_state = self._lstm.initial_state(batch_size)
        init_error = jnp.zeros((batch_size, output_size))
        init_label = jnp.zeros((batch_size, output_size))
        init_memory = jnp.zeros((self._memory_size, self._memory_size ** 2))

        # TODO merge states for batch_size > 1
        def scan_tick(carry, x):
            lstm_state, memory, error, prev_label = carry
            inp, label = x
            if self._conv is not None:
                inp = self._conv(inp)
            inp = hk.Flatten(preserve_dims=1)(inp)
            inputs = jnp.concatenate([inp, error, prev_label], axis=-1)

            out, lstm_state = self._lstm(inputs, lstm_state)
            write = self._write_head(out)
            beta = jax.nn.sigmoid(write[:, -1])
            k1, k2, v = jnp.split(jax.nn.tanh(write[:, :-1]), 3, axis=-1)
            # TODO this flatten doesn't work with batch dim
            key = jnp.outer(k1, k2).flatten()
            v_old = memory @ key
            memory = memory + beta * jnp.outer((v - v_old), key)
            memory = memory / jnp.maximum(1, jnp.linalg.norm(memory))
            n, e = jnp.split(jax.nn.tanh(self._read_head(out)), 2, axis=-1)
            # TODO optionally add multiple readouts
            n = self._layer_norm(memory @ jnp.outer(n, e).flatten())
            readout = self._read_proj(n)
            out = out + readout

            out = self._output_proj(out)
            if self._tanh_bound:
                out = jnp.tanh(out / self._tanh_bound) * self._tanh_bound
            new_error = self._loss_func_grad(out, label)
            return (lstm_state, memory, new_error, label), out
        _, outputs = hk.scan(scan_tick, (lstm_state, init_memory, init_error, init_label),
                             (inputs, labels))
        return outputs
