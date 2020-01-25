"""A wrapper for MT10 and MT50 Metaworld environments."""
import numpy as np

from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy

class MetaWorldWrapper(MultiEnvWrapper):
    """A wrapper for MT10 and MT50 environments.

    The functionallity added to MT envs includes:
        - Observations augmented with one hot encodings.
        - task sampling strategies (round robin for MT envs).
        - Ability to easily retrieve the one-hot to env_name mappings.

    Args:
        MTBenchmarkClass (Metaworld.Benchmark): The Metaworld BenchmarkClass being used.
        sample_strategy (Garage.env.multi_env_wrapper.sample_strategy):
            The sample strategy for alternating between tasks.
    """
    def __init__(self, MTBenchmarkClass, sample_strategy=round_robin_strategy):
        envs = MTBenchmarkClass._task_envs
        self._env_names = MTBenchmarkClass._task_names
        super().__init__(envs, sample_strategy)

    def _compute_env_one_hot(self, task_number):
        """Returns the one-hot encoding of task_number
        Args:
            task_number (int): The number of the task
        """
        one_hot = np.zeros(self.task_space.shape)
        one_hot[task_number] = self.task_space.high[task_number]
        return one_hot

    @property
    def task_name_to_one_hot(self):
        """Returns a :class:`dict` of the different envs and their one-hot mappings."""
        ret = {}
        for (number, name) in enumerate(self._env_names):
            ret[name] = self._compute_env_one_hot(number)

        return ret
