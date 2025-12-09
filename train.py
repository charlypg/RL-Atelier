import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import jax
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple

from atelier.agents.dqn import DQN, DoubleDQN
from atelier.buffers.xpag.buffer import DefaultBuffer
from atelier.tools.csv_logging import CSVLogger
from atelier.samplers.xpag.sampler import DefaultSampler
from atelier.types import Params, Metrics

from utils_config import (
    resolve_tuple, merge_base_variant_cli, get_str_date, two_hashes_from_dict
)

OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


def eval_episode(
    eval_env: gym.Env,
    agent: DQN,
    params: Params
) -> float:
    obs, info = eval_env.reset()
    done = False
    sum_of_rewards = 0
    while not(done):
        action = agent.select_single_action(params, obs)
        next_obs, reward, term, trunc, info = eval_env.step(action)
        sum_of_rewards += reward

        done = term or trunc
        obs = next_obs
    return sum_of_rewards

def eval(
    eval_env: gym.Env,
    agent: DQN,
    params: Params,
    nb_episodes: int
) -> Metrics:
    returns = np.zeros((nb_episodes,))
    for i in range(nb_episodes):
        returns[i] = eval_episode(
            eval_env=eval_env, agent=agent, params=params
        )
    return {
        "return_mean": np.mean(returns),
        "return_median": np.median(returns),
        "return_25": np.quantile(returns, 0.25),
        "return_75": np.quantile(returns, 0.75)
    }

def plot_from_dataframe(
    fig, ax,
    df: pd.DataFrame,
    x_key: str, y_key: str,
    fill_between: Optional[Tuple[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
):
    x = pd.to_numeric(df[x_key], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_key], errors="coerce").to_numpy()
    
    if fill_between is not None:
        ymin = pd.to_numeric(df[fill_between[0]], errors="coerce").to_numpy()
        ymax = pd.to_numeric(df[fill_between[1]], errors="coerce").to_numpy()
        ax.fill_between(x, ymin, ymax, alpha=0.2)
    
    ax.plot(x, y)
    ax.grid()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig, ax

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(hydra_config):
    # Get hyperparameters from configs
    hydra_config = merge_base_variant_cli(hydra_config)
    cfg = OmegaConf.to_container(hydra_config, resolve=True)
    for key, val in zip(cfg.keys(), cfg.values()):
        if type(val) == dict:
            print("{0}:".format(key))
            print(val)
            print()
        else:
            print("{0}: {1}".format(key, val))

    # Environment
    env = gym.make(
        cfg["env"]["env_name"],
        render_mode=None,
        **cfg["env"]["env_kwargs"]
    )
    num_eval_episodes = 20
    if num_eval_episodes == 1:
        eval_render_mode = "human"
    else:
        eval_render_mode = None
    eval_env = gym.make(
        cfg["env"]["env_name"],
        render_mode=eval_render_mode,
        **cfg["env"]["env_kwargs"]
    )
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # JAX random state
    seed = cfg["seed"]
    np_rand_state = np.random.RandomState(seed)
    rng = jax.random.PRNGKey(seed)

    # Agent
    if cfg["agent"]["method"] == "dqn":
        print("method: DQN")
        agent_class = DQN
    elif cfg["agent"]["method"] == "doubledqn":
        print("method: DoubleDQN")
        agent_class = DoubleDQN
    else:
        raise NotImplementedError
    
    agent = agent_class(
        observation_dim=observation_dim, action_dim=action_dim,
        gamma=cfg["agent"]["gamma"],
        learning_rate=cfg["agent"]["learning_rate"],
        update_target_every_x_steps=cfg["agent"]["update_target_every_x_steps"],
        network_features=cfg["agent"]["network_features"],
        eps_greedy_hp=cfg["agent"]["eps_greedy_hp"]
    )
    rng, sub = jax.random.split(rng)
    params, opt_state, target_params, epsilon = agent.init(sub)

    # Buffer and sampler
    buffer = DefaultBuffer(
        buffer_size=cfg["buffer"]["buffer_size"],
        sampler=DefaultSampler()
    )

    # cfg id and save_dir
    hashes = two_hashes_from_dict(cfg)
    save_dir_no_seed = os.path.join(
        cfg["logs"]["save_dir"],
        hashes["hash_no_seed"]
    )
    variant_path = os.path.join(save_dir_no_seed, "config.yaml")
    save_dir = os.path.join(
        save_dir_no_seed,
        get_str_date() + '_' + hashes["hash"]
    )
    exp_path = os.path.join(save_dir, "exp.yaml")
    os.makedirs(save_dir, exist_ok=True)
    # Save config
    with open(exp_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    with open(variant_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(hashes["config_no_seed"])))
    
    # Main loop
    logger = CSVLogger(save_dir=save_dir, filename="metrics.csv")
    metrics = None
    observation, info = env.reset()
    for step in tqdm(range(cfg["alg_general"]["max_steps"])):
        # Evaluation
        if step % cfg["alg_general"]["eval_every_x_steps"] == 0:
            eval_metrics = eval(
                eval_env=eval_env,
                agent=agent,
                params=params,
                nb_episodes=num_eval_episodes
            )
            if metrics is not None:
                logger.log({**eval_metrics, **metrics}, step=step)
                print(
                    "step:", step, ";",
                    "epsilon: {:.2f}".format(epsilon), ";",
                    "return_median: {:.3f}".format(eval_metrics["return_median"]), ";",
                    "return_25: {:.3f}".format(eval_metrics["return_25"]), ";",
                    "return_75: {:.3f}".format(eval_metrics["return_75"]), ";",
                    "loss:", metrics["loss"],
                    "q_mean:", metrics["q_mean"],
                    "next_q_mean:", metrics["next_q_mean"]
                )
            else:
                logger.log(eval_metrics, step=step)
                print(
                    "step:", step, ";",
                    "epsilon: {:.2f}".format(epsilon), ";",
                    "return_median: {:.3f}".format(eval_metrics["return_median"]), ";",
                    "return_25: {:.3f}".format(eval_metrics["return_25"]), ";",
                    "return_75: {:.3f}".format(eval_metrics["return_75"]), ";"
                )

        # Select action
        if np_rand_state.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.select_single_action(params, observation)
        
        # Perform step
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Store transition
        transition = {
            "observation": np.expand_dims(observation, axis=0),
            "action": np.array([[action]], dtype=np.int64),
            "next_observation": np.expand_dims(next_observation, axis=0),
            "reward": np.array([[reward]]),
            "terminated": np.array([[terminated]])
        }
        buffer.insert(transition)

        # Update if necessary
        if step > cfg["alg_general"]["start_training_after_x_steps"]:
            # Sample batch
            batch = buffer.sample(cfg["alg_general"]["batch_size"])

            if step % cfg["alg_general"]["update_params_every"] == 0:
                # Perform gradient descent step
                params, opt_state, updates, grad, metrics = agent.gradient_step(
                    params=params,
                    opt_state=opt_state,
                    target_params=target_params,
                    batch=batch
                )
            # Update target params
            target_params = agent.update_target_params(
                params=params,
                target_params=target_params,
                step=step
            )
            # Update epsilon
            epsilon = agent.update_epsilon(epsilon)

        # Prepare next iter
        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation

    # Some figures
    df = logger.get_dataframe()
    fig, ax = plt.subplots()
    fig, ax = plot_from_dataframe(
        fig, ax, df,
        x_key="step", y_key="return_median",
        fill_between=("return_25", "return_75"),
        xlabel="Step",
        ylabel="Return"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "return.pdf"))
    plt.close(fig)

    for y_key in df.keys():
        if y_key != "step" and not("return" in y_key):
            fig, ax = plt.subplots()
            fig, ax = plot_from_dataframe(
                fig, ax, df,
                x_key="step", y_key=y_key,
                xlabel="Step",
                ylabel=y_key
            )
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "{}.pdf".format(y_key)))
            plt.close(fig)


if __name__ == "__main__":
    main()
