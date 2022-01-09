import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional, Tuple, Any, Sequence
from haiku import LSTMState


class CustomLSTM(hk.RNNCore):

    def __init__(self, hidden_size: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def _initializer(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        input_size = shape[0]
        stddev = 1. / np.sqrt(input_size)
        return hk.initializers.TruncatedNormal(stddev=stddev)(shape, dtype)

    def _lstm_initializer(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        return self._initializer(shape, dtype)

    def __call__(self, inputs: jnp.ndarray,
                 prev_state: LSTMState) -> Tuple[jnp.ndarray, LSTMState]:
        if len(inputs.shape) > 2 or not inputs.shape:
            raise ValueError("LSTM input must be rank-1 or rank-2.")
        x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
        gated = hk.Linear(4 * self.hidden_size,
                          w_init=self._lstm_initializer)(x_and_h)
        # i = input, g = cell_gate, f = forget_gate, o = output_gate
        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 5)
        i = jax.nn.sigmoid(i - 5)
        c = f * prev_state.cell + i * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, LSTMState(h, c)

    def initial_state(self, batch_size: Optional[int]) -> LSTMState:
        raise NotImplementedError()

    def initial_vsml_state(self, shape: Sequence[int], rand_proportion: float) -> LSTMState:
        shape = tuple(shape) + (self.hidden_size,)
        hidden = jnp.zeros(shape)
        rand_shape = shape[:-1] + (int(shape[-1] * rand_proportion),)
        zero_shape = shape[:-1] + (shape[-1] - rand_shape[-1],)
        cell = jnp.concatenate([self._initializer(rand_shape, jnp.float32),
                                jnp.zeros(zero_shape)], axis=-1)
        return LSTMState(hidden=hidden, cell=cell)
