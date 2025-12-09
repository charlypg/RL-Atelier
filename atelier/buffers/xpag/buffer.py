# This file is adapted from xpag
# Original source: https://github.com/perrin-isir/xpag/xpag/buffers/buffer.py
# Commit: fad4d9cf77053cf29b42f322091bdf182012a501
# Copyright (c) 2022-2023, CNRS - Licensed under BSD 3-Clause License.


from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
from typing import Union, Dict, Any, Optional
import joblib
import os

from atelier.samplers.xpag.sampler import Sampler


class DataType(Enum):
    NUMPY = "data represented as numpy arrays"
    JAX = "data represented as jax.numpy arrays"


def get_datatype(x: Union[np.ndarray, jnp.ndarray]) -> DataType:
    if isinstance(x, jnp.ndarray):
        return DataType.JAX
    elif isinstance(x, np.ndarray):
        return DataType.NUMPY
    else:
        raise TypeError(f"{type(x)} not handled.")


def datatype_convert(
    x: Union[np.ndarray, jnp.ndarray, list, float],
    datatype: Union[DataType, None] = DataType.NUMPY,
) -> Union[np.ndarray, jnp.ndarray]:
    if datatype is None:
        return x
    elif datatype == DataType.NUMPY:
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)
    elif datatype == DataType.JAX:
        if isinstance(x, jnp.ndarray):
            return x
        else:
            return jnp.array(x)

def batch_shapes(
    batch: Dict[str, Union[np.ndarray, jnp.ndarray]]
) -> Dict[str, tuple]:
    return jax.tree_map(lambda x: x.shape, batch)

def batch_reshape_by_gd_steps(
    batch: Dict[str, Union[np.ndarray, jnp.ndarray]],
    num_gd_steps: int, 
    batch_size: int
) -> Dict[str, tuple]:
    return jax.tree_map(
        lambda x: x.reshape((num_gd_steps, batch_size, x.shape[-1])), batch
    )


class Buffer(ABC):
    """Base class for buffers"""

    def __init__(
        self,
        buffer_size: int,
        sampler: Optional[Sampler] = None,
    ):
        self.buffer_size = buffer_size
        self.sampler = sampler

    @abstractmethod
    def insert(self, step: Dict[str, Any]):
        """Inserts a transition in the buffer"""
        pass

    @abstractmethod
    def sample(self, batch_size) -> Dict[str, Union[np.ndarray, jnp.ndarray]]:
        """Uses the sampler to returns a batch of transitions"""
        pass


class EpisodicBuffer(Buffer):
    """Base class for episodic buffers"""

    def __init__(
        self,
        buffer_size: int,
        sampler: Sampler,
    ):
        super().__init__(buffer_size, sampler)

    @abstractmethod
    def store_done(self, done):
        """Stores the episodes that are done"""
        pass


class DefaultBuffer(Buffer):
    def __init__(
        self,
        buffer_size: int,
        sampler: Sampler,
    ):
        super().__init__(buffer_size, sampler)
        self.current_size = 0
        self.buffers = {}
        self.size = buffer_size
        self.dict_sizes = None
        self.num_envs = None
        self.keys = None
        self.zeros = None
        self.where = None
        self.first_insert_done = False

    def init_buffer(self, step: Dict[str, Any]):
        self.dict_sizes = {}
        self.keys = list(step.keys())
        assert "terminated" in self.keys
        for key in self.keys:
            if isinstance(step[key], dict):
                for k in step[key]:
                    assert len(step[key][k].shape) == 2
                    self.dict_sizes[key + "." + k] = step[key][k].shape[1]
            else:
                assert len(step[key].shape) == 2
                self.dict_sizes[key] = step[key].shape[1]
        self.num_envs = step["terminated"].shape[0]
        for key in self.dict_sizes:
            self.buffers[key] = np.zeros([self.size, self.dict_sizes[key]])
        self.zeros = lambda i: np.zeros(i).astype("int")
        self.where = np.where
        self.first_insert_done = True

    def insert(self, step: Dict[str, Any]):
        if not self.first_insert_done:
            self.init_buffer(step)
        idxs = self._get_storage_idx(inc=self.num_envs)
        for key in self.keys:
            if isinstance(step[key], dict):
                for k in step[key]:
                    self.buffers[key + "." + k][idxs, :] = datatype_convert(
                        step[key][k], DataType.NUMPY
                    ).reshape((self.num_envs, self.dict_sizes[key + "." + k]))
            else:
                self.buffers[key][idxs, :] = datatype_convert(
                    step[key], DataType.NUMPY
                ).reshape((self.num_envs, self.dict_sizes[key]))

    def pre_sample(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][: self.current_size]
        return temp_buffers

    def sample(self, batch_size):
        return self.sampler.sample(self.pre_sample(), batch_size)

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        return idx

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        list_vars = [
            ("current_size", self.current_size),
            ("buffers", self.buffers),
            ("size", self.size),
            ("dict_sizes", self.dict_sizes),
            ("num_envs", self.num_envs),
            ("keys", self.keys),
            ("first_insert_done", self.first_insert_done),
        ]
        for cpl in list_vars:
            with open(os.path.join(directory, cpl[0] + ".joblib"), "wb") as f_:
                joblib.dump(cpl[1], f_)

    def load(self, directory: str):
        self.current_size = joblib.load(os.path.join(directory, "current_size.joblib"))
        self.buffers = joblib.load(os.path.join(directory, "buffers.joblib"))
        self.size = joblib.load(os.path.join(directory, "size.joblib"))
        self.dict_sizes = joblib.load(os.path.join(directory, "dict_sizes.joblib"))
        self.num_envs = joblib.load(os.path.join(directory, "num_envs.joblib"))
        self.keys = joblib.load(os.path.join(directory, "keys.joblib"))
        self.first_insert_done = joblib.load(
            os.path.join(directory, "first_insert_done.joblib")
        )
        self.zeros = lambda i: np.zeros(i).astype("int")
        self.where = np.where


class DefaultEpisodicBuffer(EpisodicBuffer):
    def __init__(
        self,
        max_episode_steps: int,
        buffer_size: int,
        sampler: Optional[Sampler] = None,
    ):
        super().__init__(buffer_size, sampler)
        self.current_size = 0
        self.buffers = {}
        self.T = max_episode_steps
        self.size = int(buffer_size // self.T)
        self.dict_sizes = None
        self.num_envs = None
        self.keys = None
        self.current_t = None
        self.zeros = None
        self.where = None
        self.current_idxs = None
        self.first_insert_done = False

    def init_buffer(self, step: Dict[str, Any]):
        self.dict_sizes = {}
        self.keys = list(step.keys())
        assert "terminated" in self.keys
        self.num_envs = step["terminated"].shape[0]
        for key in self.keys:
            if isinstance(step[key], dict):
                for k in step[key]:
                    assert len(step[key][k].shape) == 2
                    self.dict_sizes[key + "." + k] = step[key][k].shape[1]
            else:
                assert len(step[key].shape) == 2, (
                    f"step[{key}] must be 2-dimensional (e.g. shape "
                    f"({self.num_envs}, 1) instead of ({self.num_envs},) for scalar "
                    f"entries)"
                )
                self.dict_sizes[key] = step[key].shape[1]
        self.dict_sizes["episode_length"] = 1
        for key in self.dict_sizes:
            self.buffers[key] = np.zeros([self.size, self.T, self.dict_sizes[key]])
        self.current_t = np.zeros(self.num_envs).astype("int")
        self.zeros = lambda i: np.zeros(i).astype("int")
        self.where = np.where
        self.current_idxs = DefaultEpisodicBuffer._get_storage_idx(
            self=self, inc=self.num_envs
        )
        self.first_insert_done = True

    def insert(self, step: Dict[str, Any]):
        if not self.first_insert_done:
            self.init_buffer(step)
        for key in self.keys:
            if isinstance(step[key], dict):
                for k in step[key]:
                    self.buffers[key + "." + k][
                        self.current_idxs, self.current_t, :
                    ] = datatype_convert(step[key][k], DataType.NUMPY).reshape(
                        (self.num_envs, self.dict_sizes[key + "." + k])
                    )
            else:
                self.buffers[key][
                    self.current_idxs, self.current_t, :
                ] = datatype_convert(step[key], DataType.NUMPY).reshape(
                    (self.num_envs, self.dict_sizes[key])
                )
        self.current_t += 1
        self.buffers["episode_length"][
            self.current_idxs, self.zeros(self.num_envs), :
        ] = self.current_t.reshape((self.num_envs, 1))

    def store_done(self, done):
        if done.max():
            where_done = self.where(datatype_convert(done, DataType.NUMPY) == 1)[0]
            k_envs = len(where_done)
            new_idxs = self._get_storage_idx(inc=k_envs)
            self.current_idxs[where_done] = new_idxs.reshape((1, len(new_idxs)))
            self.current_t[where_done] = 0

    def pre_sample(self):
        temp_buffers = {}
        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][: self.current_size]
        return temp_buffers

    def sample(self, batch_size):
        return self.sampler.sample(self.pre_sample(), batch_size)

    def sample_with_ext_sampler(self, batch_size, sampler: Sampler):
        return sampler.sample(self.pre_sample(), batch_size)

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.buffers["episode_length"][idx, self.zeros(inc), :] = self.zeros(
            inc
        ).reshape((inc, 1))
        self.current_size = min(self.size, self.current_size + inc)
        return idx

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        list_vars = [
            ("buffer_size", self.buffer_size),
            ("current_size", self.current_size),
            ("buffers", self.buffers),
            ("T", self.T),
            ("size", self.size),
            ("dict_sizes", self.dict_sizes),
            ("num_envs", self.num_envs),
            ("keys", self.keys),
            ("current_t", self.current_t),
            ("current_idxs", self.current_idxs),
            ("first_insert_done", self.first_insert_done),
        ]
        for cpl in list_vars:
            with open(os.path.join(directory, cpl[0] + ".joblib"), "wb") as f_:
                joblib.dump(cpl[1], f_)

    def load(self, directory: str):
        self.buffer_size = joblib.load(os.path.join(directory, "buffer_size.joblib"))
        self.current_size = joblib.load(os.path.join(directory, "current_size.joblib"))
        self.buffers = joblib.load(os.path.join(directory, "buffers.joblib"))
        self.T = joblib.load(os.path.join(directory, "T.joblib"))
        self.size = joblib.load(os.path.join(directory, "size.joblib"))
        self.dict_sizes = joblib.load(os.path.join(directory, "dict_sizes.joblib"))
        self.num_envs = joblib.load(os.path.join(directory, "num_envs.joblib"))
        self.keys = joblib.load(os.path.join(directory, "keys.joblib"))
        self.current_t = joblib.load(os.path.join(directory, "current_t.joblib"))
        self.current_idxs = joblib.load(os.path.join(directory, "current_idxs.joblib"))
        self.first_insert_done = joblib.load(
            os.path.join(directory, "first_insert_done.joblib")
        )
        self.zeros = lambda i: np.zeros(i).astype("int")
        self.where = np.where
