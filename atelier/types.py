import numpy as np
import jax.numpy as jnp
import flax
from typing import Any, Dict, NamedTuple, Union

Params = flax.core.FrozenDict[str, Any]
Metrics = Dict[str, Any]

class Batch(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    terminateds: jnp.ndarray
    next_observations: jnp.ndarray

def dict_to_batch(
    dico: Dict[str, Union[np.ndarray, jnp.ndarray]],
    actions_are_int: bool = False
) -> Batch:
    if actions_are_int:
        actions = jnp.array(dico["action"], dtype=jnp.int32)
    else:
        actions = jnp.array(dico["action"])
    
    return Batch(
        observations=jnp.array(dico["observation"]),
        actions=actions,
        rewards=jnp.array(dico["reward"]),
        terminateds=jnp.array(dico["terminated"]),
        next_observations=jnp.array(dico["next_observation"])
    )
