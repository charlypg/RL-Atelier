import numpy as np
import jax.numpy as jnp
from typing import Dict, Union, Tuple

from atelier.samplers.xpag.sampler import Sampler


class SumTree:
    """
    SumTree that samples integers according to a softmax over priorities.
    Internally stores exp(priority) so that sampling probability is softmax.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        init_max_priority: float = 1.0
    ):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=np.int64)
        self.raw_priorities = np.zeros(capacity, dtype=np.float64)

        self.alpha = alpha # How much we prioritize
        self.beta = beta # Importance sampling hyperparameter
        self.init_max_priority = init_max_priority

        self.write = 0
        self.size = 0
    
    def _propagate(self, idx, change):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, value):
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx  # leaf reached

            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1

    def max_priority(self):
        """
        Return the maximum raw priority currently stored in the tree.
        """
        if self.size == 0:
            return self.init_max_priority
        return np.max(self.raw_priorities[:self.size])


    def total(self) -> float:
        return self.tree[0]

    def priority_to_weight(self, priorities: np.ndarray) -> np.ndarray:
        return np.exp(self.alpha * priorities)
    
    def add_batch(self, values, priorities):
        """
        Insert a batch of values with priorities.
        Priorities are converted to exp(priority) internally.
        """
        values = np.asarray(values, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)

        for v, p in zip(values, priorities):
            tree_idx = self.write + self.capacity - 1

            self.data[self.write] = v
            self.raw_priorities[self.write] = p

            new_weight = self.priority_to_weight(p) #np.exp(self.alpha * p)
            delta = new_weight - self.tree[tree_idx]

            self.tree[tree_idx] = new_weight
            self._propagate(tree_idx, delta)

            self.write = (self.write + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def update_batch(self, indices, priorities):
        """
        Update priorities of an existing batch of data.
        
        indices: indices in [0, self.capacity)
        priorities: new priority values
        """
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)

        for i, p in zip(indices, priorities):
            if i < 0 or i >= self.size:
                # continue  # ignore invalid indices
                raise "invalid index"

            tree_idx = i + self.capacity - 1

            old_weight = self.tree[tree_idx]
            new_weight = self.priority_to_weight(p) #np.exp(self.alpha * p)

            delta = new_weight - old_weight

            self.tree[tree_idx] = new_weight
            self.raw_priorities[i] = p

            self._propagate(tree_idx, delta)


    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        """
        Sample a batch of values according to softmax(priorities).
        Uses stratified sampling for lower variance.
        """
        sampled_values = []
        sampled_priorities = []

        segment = self.total() / batch_size

        for i in range(batch_size):
            try:
                s = np.random.uniform(segment * i, segment * (i + 1))
            except:
                print("AAAAAAAAHHHHHH", segment)
            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1

            sampled_values.append(self.data[data_idx])
            sampled_priorities.append(self.raw_priorities[data_idx])

        idxs_info = dict(
            idxs=np.array(sampled_values),
            sampled_priorities=np.array(sampled_priorities)
        )
        # idxs_info["sampled_priorities"] = np.maximum(idxs_info["sampled_priorities"], 1e-6)
        # idxs_info["log_probs"] = np.log(idxs_info["sampled_priorities"]) - np.log(self.total())
        # idxs_info["log_IS_weights"] = -self.beta * (np.log(self.size) + idxs_info["log_probs"])
        # idxs_info["IS_weights"] = np.exp(idxs_info["log_IS_weights"])
        # idxs_info["IS_weights"] /= idxs_info["IS_weights"].max()
        
        idxs_info["sampled_weights"] = self.priority_to_weight(idxs_info["sampled_priorities"])
        idxs_info["sampled_weights_norm"] = idxs_info["sampled_weights"] / self.total()
        idxs_info["IS_weights"] = (self.size * idxs_info["sampled_weights_norm"]) ** (-self.beta)

        return idxs_info


class PERSampler(Sampler):
    def __init__(
        self,
        buffer_size: int,
        alpha: float,
        beta: float,
        init_max_priority: float = 1.0,
        seed: Union[int, None] = None
    ):
        Sampler.__init__(self, seed=seed)
        
        # Encapsulate a SumTree for prioritized sampling
        self.sum_tree = SumTree(
            capacity=buffer_size,
            alpha=alpha,
            beta=beta,
            init_max_priority=init_max_priority
        )

    def insert(self, idxs: np.ndarray, **kwargs):
        # Insert new transitions according to proportional PER
        # Cf Algorithm 1 https://arxiv.org/pdf/1511.05952
        max_priority = self.sum_tree.max_priority()
        priorities = max_priority * np.ones(shape=idxs.shape)
        self.sum_tree.add_batch(idxs, priorities)

    def update(self, idxs: np.ndarray, per_priorities: np.ndarray, **kwargs):
        # Update priorities according to proportional PER
        # Cf Algorithm 1 https://arxiv.org/pdf/1511.05952
        self.sum_tree.update_batch(idxs, per_priorities)

    def sample(
        self,
        buffer,
        batch_size: int,
    ) -> Tuple[Dict[str, Union[np.ndarray, jnp.ndarray]], dict]:
        """Return a batch of transitions"""
        idxs_info = self.sum_tree.sample(batch_size)
        transitions = {key: buffer[key][idxs_info["idxs"]] for key in buffer.keys()}
        return transitions, idxs_info
