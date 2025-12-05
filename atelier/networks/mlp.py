import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Sequence


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    init_last_layer_zeros: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.init_last_layer_zeros:
                x = nn.Dense(
                    size,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                )(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x
