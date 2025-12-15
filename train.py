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
from atelier.tools.logging import CSVLogger, PrintLogger
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
    for _ in range(max_episode_steps):
        action = np.array(agent.select_action(params, obs))
        next_obs, reward, term, trunc, info = eval_env.step(action.squeeze())

        done = np.logical_or(term, trunc)
        sum_of_rewards += reward * (1 - done)
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

    if not("max_episode_steps" in cfg["env"]["kwargs"]):
        raise "NO max_episode_steps in env.kwargs"
    env, env_info = make_envs(
        env_name=cfg["env"]["env_name"],
        num_envs=cfg["env"]["num_envs"],
        **cfg["env"]["kwargs"]
    )
    eval_env, _ = make_envs(
        env_name=cfg["env"]["env_name"],
        num_envs=cfg["env"]["eval_num_envs"],
        **cfg["env"]["kwargs"]
    )
    print(env_info)

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
        observation_dim=env_info["observation_dim"], 
        action_dim=env_info["action_dim"],
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
    print_logger = PrintLogger(debug=True)
    logger_eval = CSVLogger(save_dir=save_dir, filename="eval.csv")
    logger_learning_metrics = CSVLogger(
        save_dir=save_dir, filename="learning_metrics.csv"
    )
    metrics = None
    observation, info = env.reset()
    num_iterations = cfg["alg_general"]["max_steps"] // cfg["env"]["num_envs"]
    assert(cfg["env"]["num_envs"] % cfg["alg_general"]["update_params_every"] == 0)
    assert(cfg["env"]["num_envs"] >= cfg["alg_general"]["update_params_every"])
    num_gd_steps_per_step = cfg["env"]["num_envs"] // cfg["alg_general"]["update_params_every"]
    print("num_gd_steps_per_step:", num_gd_steps_per_step)
    for i in tqdm(range(num_iterations)):
        # Evaluation
        if (i * cfg["env"]["num_envs"]) % cfg["alg_general"]["eval_every_x_steps"] == 0:
            eval_metrics = evaluate(
                eval_env=eval_env,
                agent=agent,
                params=params,
                np_rand_state=np_rand_state,
                max_episode_steps=cfg["env"]["kwargs"]["max_episode_steps"]
            )
            logger_eval.log(eval_metrics, step=i * cfg["env"]["num_envs"])
            if metrics is not None:
                logger_learning_metrics.log(metrics, step=i * cfg["env"]["num_envs"])
                print_logger.log({**eval_metrics, **metrics}, step=i * cfg["env"]["num_envs"])
            else:
                print_logger.log(eval_metrics, step=i * cfg["env"]["num_envs"])

        # Select action
        if np_rand_state.uniform() < epsilon:
            action = np.expand_dims(env.action_space.sample(), axis=-1)
        else:
            action = np.array(agent.select_action(params, observation))
        
        # Perform step
        next_observation, reward, terminated, truncated, info = env.step(action.squeeze())

        # Store transition
        transition = {
            "observation": observation,
            "action": action,
            "next_observation": next_observation,
            "reward": np.expand_dims(reward, axis=-1),
            "terminated": np.expand_dims(terminated, axis=-1)
        }
        buffer.insert(transition)

        # Update if necessary
        if i * cfg["env"]["num_envs"] > cfg["alg_general"]["start_training_after_x_steps"]:
            # Sample batch
            batch = buffer.sample(cfg["alg_general"]["batch_size"])

            # TODO: encapsulate updates
            for _ in range(num_gd_steps_per_step):
                # Perform gradient descent step
                params, opt_state, updates, grad, metrics = agent.gradient_step(
                    params=params,
                    opt_state=opt_state,
                    target_params=target_params,
                    batch=batch
                )
                # logger_learning_metrics.log(metrics, step=step) # DEBUG
            for k in range(cfg["env"]["num_envs"]):
                # Update target params
                target_params = agent.update_target_params(
                    params=params,
                    target_params=target_params,
                    step=i*cfg["env"]["num_envs"]+k
                )
                # Update epsilon
                epsilon = agent.update_epsilon(epsilon)
            metrics["epsilon"] = epsilon

        # Prepare next iter
        done = np.logical_or(terminated, truncated)
        if np.max(done):
            observation, info = env.reset(
                options={"reset_mask": done}, 
                seed=np_rand_state.randint(1_000_000)
            )
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
