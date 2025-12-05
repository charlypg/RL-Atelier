import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen as nn
import optax
from typing import Optional, Sequence, Tuple

from atelier.types import Params

def random_key_split(rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jax.random.split(rng)

def init(
    model_def: nn.Module,
    inputs: Sequence[jnp.ndarray],
    tx: Optional[optax.GradientTransformation] = None,
) -> Tuple[Params, optax.OptState]:
    """Generic initialization function"""
    # Initialize params
    variables = model_def.init(*inputs)
    params = variables.pop("params")

    # Initialize optimizer state
    if tx is not None:
        opt_state = tx.init(params)
    else:
        opt_state = None

    return params, opt_state

def grad_norm(grad):
    flattened_grads, _ =  ravel_pytree(grad)
    return jax.numpy.linalg.norm(flattened_grads)

def get_shapes(params: Params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)

def apply_updates(
    optimizer: optax.GradientTransformation,
    grad: Params, 
    opt_state: optax.OptState, 
    params: Params,
) -> Tuple[Params, optax.OptState, Params]:
    updates, new_opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, updates

def ema_target_update(
    params: Params, target_params: Params, tau: float
) -> Params:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), params, target_params
    )

    return new_target_params

def huber(td: jnp.ndarray) -> jnp.ndarray:
    """Huber function."""
    abs_td = jnp.abs(td)
    return jnp.where(abs_td <= 1.0, jnp.square(td), abs_td)
