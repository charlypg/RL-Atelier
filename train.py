import hydra
from omegaconf import OmegaConf
import os
import numpy as np
import jax
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from atelier.agents.dqn import DQN, DoubleDQN
from atelier.buffers.xpag.buffer import DefaultBuffer
from atelier.tools.csv_logging import CSVLogger
from atelier.samplers.xpag.sampler import DefaultSampler
from atelier.types import Params, Metrics
from atelier.wrappers.envs import make_envs
from plotting import plot_from_dataframe
from utils_config import (
    resolve_tuple, merge_base_variant_cli, get_str_date, two_hashes_from_dict
)

OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


def evaluate(
    eval_env: gym.vector.AsyncVectorEnv,
    agent: DQN,
    params: Params,
    np_rand_state: np.random.RandomState,
    max_episode_steps: int
) -> Metrics:
    shape_env = (eval_env.num_envs,)
    obs, info = eval_env.reset(seed=np_rand_state.randint(1_000_000))
    done = np.zeros(shape_env, dtype=np.bool_)
    sum_of_rewards = np.zeros(shape_env)
    flags = np.zeros(shape_env)
    for _ in range(max_episode_steps):
        action = np.array(agent.select_action(params, obs))
        next_obs, reward, term, trunc, info = eval_env.step(action)

        done = np.logical_or(term, trunc)
        flags = np.logical_or(flags, done)
        sum_of_rewards += reward * (1 - flags)
        obs = next_obs

    return {
        "sum_rewards_mean": np.mean(sum_of_rewards),
        "sum_rewards_median": np.median(sum_of_rewards),
        "sum_rewards_25": np.quantile(sum_of_rewards, 0.25),
        "sum_rewards_75": np.quantile(sum_of_rewards, 0.75)
    }

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
    eval_env, _ = make_envs(
        env_name=cfg["env"]["env_name"],
        num_envs=cfg["env"]["num_eval_env"],
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
        update_params_every=cfg["agent"]["update_params_every"],
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
    logger_eval = CSVLogger(save_dir=save_dir, filename="eval.csv")
    logger_learning_metrics = CSVLogger(
        save_dir=save_dir, filename="learning_metrics.csv"
    )
    learning_metrics = None
    observation, info = env.reset()
    for step in tqdm(range(cfg["alg_general"]["max_steps"])):
        # Evaluation
        if step % cfg["alg_general"]["eval_every_x_steps"] == 0:
            eval_metrics = evaluate(
                eval_env=eval_env,
                agent=agent,
                params=params,
                np_rand_state=np_rand_state,
                max_episode_steps=cfg["env"]["env_kwargs"]["max_episode_steps"]
            )
            logger_eval.log(eval_metrics, step=step)
            if learning_metrics is not None:
                logger_learning_metrics.log(learning_metrics, step=step)
                print(
                    "step:", step, ";",
                    "epsilon: {:.2f}".format(epsilon), ";",
                    "sum_rewards_median: {:.3f}".format(eval_metrics["sum_rewards_median"]), ";",
                    "sum_rewards_25: {:.3f}".format(eval_metrics["sum_rewards_25"]), ";",
                    "sum_rewards_75: {:.3f}".format(eval_metrics["sum_rewards_75"]), ";",
                    "loss:", learning_metrics["loss"],
                    "q_mean:", learning_metrics["q_mean"],
                    "next_q_mean:", learning_metrics["next_q_mean"]
                )
            else:
                print(
                    "step:", step, ";",
                    "epsilon: {:.2f}".format(epsilon), ";",
                    "sum_rewards_median: {:.3f}".format(eval_metrics["sum_rewards_median"]), ";",
                    "sum_rewards_25: {:.3f}".format(eval_metrics["sum_rewards_25"]), ";",
                    "sum_rewards_75: {:.3f}".format(eval_metrics["sum_rewards_75"]), ";"
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
        # Insert transition in buffer
        buffer.insert(transition)

        # Update if necessary
        if step > cfg["alg_general"]["start_training_after_x_steps"]:
            # Sample batch
            batch, batch_info = buffer.sample(cfg["alg_general"]["batch_size"])
            
            # Update agent
            (
                params,
                opt_state,
                target_params,
                epsilon,
                updates,
                grad,
                grad_metrics
            ) = agent.update(
                params=params,
                opt_state=opt_state,
                target_params=target_params,
                batch=batch,
                epsilon=epsilon,
                step=step
            )

            if grad_metrics is not None:
                # Update learning metrics
                learning_metrics = grad_metrics.copy()
                
                # Update sampler
                batch_info["per_priorities"] = grad_metrics["abs_td_error"]
                buffer.update_sampler(**batch_info)

        # Prepare next iter
        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation

    # Some figures
    import pandas as pd
    df = pd.concat(
        [logger_eval.get_dataframe(), logger_learning_metrics.get_dataframe()],
        ignore_index=True
    )
    fig, ax = plt.subplots()
    fig, ax = plot_from_dataframe(
        fig, ax, df,
        x_key="step", y_key="sum_rewards_median",
        fill_between=("sum_rewards_25", "sum_rewards_75"),
        xlabel="Step",
        ylabel="Sum of rewards"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sum_rewards.pdf"))
    plt.close(fig)

    for y_key in df.keys():
        if y_key != "step" and not("sum_rewards" in y_key):
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
