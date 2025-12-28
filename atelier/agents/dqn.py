import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple

from atelier.networks.dqn import DeepQNetwork
from atelier.types import Metrics, Params
from atelier.utils import (
    init, apply_updates, huber, get_shapes, grad_norm
)


class DQN:
    """Class implementing a DQN algorithm."""
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        learning_rate = 3e-4,
        update_params_every: int = 4,
        update_target_every_x_steps: int = 10_000,
        network_features: dict = dict(hidden_dims=(128, 128)),
        eps_greedy_hp: dict = dict(
            eps_decrease="exponential",
            eps_end=0.01,
            eps_end_at=4e5
        )
    ):
        # Observation and action spaces
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Discount factor
        self.gamma = gamma

        # Epsilon-greedy parameters
        self.eps_greedy_hp = eps_greedy_hp.copy()
        if self.eps_greedy_hp["eps_decrease"] == "exponential":
            self.eps_dec = self.eps_greedy_hp["eps_end"]**(
                1 / self.eps_greedy_hp["eps_end_at"]
            )
            print("DQN.eps_dec (exponential):", self.eps_dec)
            self.update_epsilon = self.update_epsilon_exponential
        elif self.eps_greedy_hp["eps_decrease"] == "linear":
            self.eps_dec = (
                self.eps_greedy_hp["eps_end"] - 1.0
            ) / self.eps_greedy_hp["eps_end_at"]
            print("DQN.eps_dec (linear):", self.eps_dec)
            self.update_epsilon = self.update_epsilon_linear
        else:
            raise NotImplementedError

        # Optimization
        self.optimizer = optax.adam(learning_rate=learning_rate)
        self.update_params_every = update_params_every
        self.update_target_every_x_steps = update_target_every_x_steps

        # Neural network
        self.network = self.make_network(network_features)
    
    def make_network(self, network_features: dict) -> DeepQNetwork:
        return DeepQNetwork(
            action_dim=self.action_dim,
            **network_features
        )
    
    def init(self, rng: jnp.ndarray) -> Tuple[Params, optax.OptState, Params, float]:
        init_obs = jnp.ones((1, 1, self.observation_dim))

        params, opt_state = init(
            model_def=self.network,
            inputs=[rng, init_obs],
            tx=self.optimizer
        )
        print(get_shapes(params))

        target_params, _ = init(
            model_def=self.network,
            inputs=[rng, init_obs]
        )
        print(get_shapes(target_params))

        epsilon = 1.0
        print("epsilon:", epsilon)

        return params, opt_state, target_params, epsilon
    
    @functools.partial(
        jax.jit,
        static_argnames=(
            "self",
        )
    )
    def select_action(
        self,
        params: Params,
        observation: jnp.ndarray
    ) -> jnp.ndarray:
        print("COMPILE: DQN.select_action")
        actions_values = self.network.apply({"params": params}, observation)
        return jnp.argmax(actions_values, axis=-1)
    
    def select_single_action(
        self,
        params: Params,
        observation: np.ndarray
    ) -> jnp.ndarray:
        return int(
            jnp.squeeze(
                self.select_action(
                    params=params,
                    observation=jnp.expand_dims(jnp.array(observation), axis=0)
                )
            )
        )
    
    def dqn_loss( # TODO: comments and error metric
        self,
        params: Params,
        target_params: Params,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        masks: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Metrics]:
        actions_values = self.network.apply({"params": params}, observations)
        actions = jnp.array(actions, dtype=jnp.int32)
        current_action_values = jnp.take_along_axis(
            actions_values, actions, axis=-1
        ).squeeze(axis=-1)
        next_actions_values = self.network.apply({"params": target_params}, next_observations)
        next_values = jnp.max(next_actions_values, axis=-1)
        rewards = jnp.squeeze(rewards, axis=-1)
        masks = jnp.squeeze(masks, axis=-1)
        target_values = rewards + self.gamma * masks * next_values
        print(rewards.shape, masks.shape, next_values.shape)
        print(current_action_values.shape, target_values.shape)
        td_error = current_action_values - target_values
        batch_loss = huber(td_error)
        loss = jnp.mean(batch_loss)
        return loss, {
            "loss": loss,
            "q_mean": jnp.mean(current_action_values),
            "next_q_mean": jnp.mean(next_values),
            "abs_td_error": jnp.abs(td_error)
        }
    
    @functools.partial(
        jax.jit,
        static_argnames=(
            "self",
        )
    )
    def gradient_step(
        self,
        params: Params,
        opt_state: optax.OptState,
        target_params: Params,
        batch: Dict[str, np.ndarray]
    ) -> tuple:
        print("COMPILE: DQN.gradient_step")
        grad, metrics = jax.grad(self.dqn_loss, has_aux=True)(
            params,
            target_params=target_params,
            observations=batch["observation"],
            actions=batch["action"],
            next_observations=batch["next_observation"],
            rewards=batch["reward"],
            masks=1-batch["terminated"]
        )
        params, opt_state, updates = apply_updates(
            optimizer=self.optimizer,
            grad=grad,
            opt_state=opt_state,
            params=params
        )
        return params, opt_state, updates, grad, metrics
    
    def update_epsilon_exponential(self, epsilon: float) -> float:
        return max(self.eps_greedy_hp["eps_end"], self.eps_dec * epsilon)
    
    def update_epsilon_linear(self, epsilon: float) -> float:
        return max(self.eps_greedy_hp["eps_end"], self.eps_dec + epsilon)
    
    def update_target_params(
        self,
        params: Params,
        target_params: Params,
        step: int
    ) -> Params:
        if step % self.update_target_every_x_steps == 0:
            return params
        else:
            return target_params
    
    def update(
        self,
        params: Params,
        opt_state: optax.OptState,
        target_params: Params,
        batch: Dict[str, np.ndarray],
        epsilon: float,
        step: int
    ):
        if step % self.update_params_every == 0:
            # Perform gradient descent step
            params, opt_state, updates, grad, grad_metrics = self.gradient_step(
                params=params,
                opt_state=opt_state,
                target_params=target_params,
                batch=batch
            )
        else:
            updates = None
            grad = None
            grad_metrics = None

        # Update target params
        target_params = self.update_target_params(
            params=params,
            target_params=target_params,
            step=step
        )

        # Update epsilon
        epsilon = self.update_epsilon(epsilon)

        return (
            params,
            opt_state,
            target_params,
            epsilon,
            updates,
            grad,
            grad_metrics
        )


class DoubleDQN(DQN):
    def dqn_loss(
        self,
        params: Params,
        target_params: Params,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        masks: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Metrics]:
        # Compute current value
        actions_values = self.network.apply({"params": params}, observations)
        actions = jnp.array(actions, dtype=jnp.int32)
        current_action_values = jnp.take_along_axis(
            actions_values, actions, axis=-1
        ).squeeze(axis=-1)
        # Compute next actions according to current q function
        next_actions_values = jax.lax.stop_gradient(
            self.network.apply({"params": params}, next_observations)
        )
        next_actions = jnp.argmax(next_actions_values, axis=-1, keepdims=True)
        # Compute target values according to https://arxiv.org/pdf/1509.06461
        target_next_actions_values = self.network.apply(
            {"params": target_params}, next_observations
        )
        next_values = jnp.take_along_axis(
            target_next_actions_values, next_actions, axis=-1
        ).squeeze(axis=-1)
        rewards = jnp.squeeze(rewards, axis=-1)
        masks = jnp.squeeze(masks, axis=-1)
        target_values = rewards + self.gamma * masks * next_values
        print(rewards.shape, masks.shape, next_values.shape)
        print(current_action_values.shape, target_values.shape)
        # Compute loss
        td_error = current_action_values - target_values
        batch_loss = huber(td_error)
        loss = jnp.mean(batch_loss)
        return loss, {
            "loss": loss,
            "q_mean": jnp.mean(current_action_values),
            "next_q_mean": jnp.mean(next_values),
            "abs_td_error": jnp.abs(td_error)
        }
