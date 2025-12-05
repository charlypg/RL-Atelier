import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence

from atelier.networks.mlp import MLP

class DeepQNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    init_last_layer_zeros: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        action_values = MLP(
            hidden_dims=(*self.hidden_dims, self.action_dim),
            activations=self.activations,
            activate_final=self.activate_final,
            init_last_layer_zeros=self.init_last_layer_zeros
        )(observations)
        return action_values
