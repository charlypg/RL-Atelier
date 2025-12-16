import gymnasium as gym
from typing import Any, Dict, Tuple

# TODO: add render_mode for one env
def make_envs(
    env_name: str,
    num_envs: int,
    **kwargs
) -> Tuple[gym.vector.AsyncVectorEnv, Dict[str, Any]]:
    dummy_env = gym.make(env_name, **kwargs)
    env_info = dict()
    
    # Observation space info
    observation_space = dummy_env.observation_space
    env_info["discrete_obs_space"] = isinstance(observation_space, gym.spaces.Discrete)
    if env_info["discrete_obs_space"]:
        env_info["observation_dim"] = observation_space.n
    else:
        assert(isinstance(dummy_env.observation_space, gym.spaces.Box))
        env_info["observation_dim"] = observation_space.shape[0]
    
    # Action space info
    action_space = dummy_env.action_space
    env_info["discrete_act_space"] = isinstance(action_space, gym.spaces.Discrete)    
    if env_info["discrete_act_space"]:
        env_info["action_dim"] = action_space.n
    else:
        assert(isinstance(dummy_env.action_space, gym.spaces.Box))
        env_info["action_dim"] = action_space.shape[0]
    
    # Return vectorized env
    return gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(env_name, **kwargs)
        ] * num_envs,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED
    ), env_info
