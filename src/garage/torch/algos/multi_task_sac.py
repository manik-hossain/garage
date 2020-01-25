"""An implementation of MT-Sac first described in Metaworlds."""
import copy

import numpy as np
import torch

from dowel import logger, tabular
from garage.torch.utils import np_to_torch, torch_to_np
from collections import deque

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.algos import SAC

class MultiTaskSAC(OffPolicyRLAlgorithm):

    def __init__(self,
                 env_spec,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 gradient_steps_per_itr,
                 alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 use_automatic_entropy_tuning=True,
                 discount=0.99,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 smooth_return=True,
                 input_include_goal=False):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.replay_buffer = replay_buffer
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.initial_log_entropy = initial_log_entropy
        self.gradient_steps = gradient_steps_per_itr
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=self.gradient_steps,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         smooth_return=smooth_return)
        self.reward_scale = reward_scale
        # use 2 target q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf1_optimizer = optimizer(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer(self.qf2.parameters(), lr=self.qf_lr)
        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning and not alpha:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            import ipdb; ipdb.set_trace()
            self._num_tasks = self.env_spec.num_tasks
            self.log_alpha = torch.tensor([self.initial_log_entropy] * self._num_tasks, dtype=torch.float, requires_grad=True)
            self.alpha_optimizer = optimizer([self.log_alpha], lr=self.policy_lr)
        else:
            self.alpha = [alpha] * self._num_tasks

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            if self.replay_buffer.n_transitions_stored < self.min_buffer_size:
                batch_size = self.min_buffer_size
            else:
                batch_size = None
            runner.step_path = runner.obtain_samples(runner.step_itr, batch_size)
            for sample in runner.step_path:
                self.replay_buffer.store(obs=sample.observation,
                                        act=sample.action,
                                        rew=sample.reward,
                                        next_obs=sample.next_observation,
                                        done=sample.terminal)
            self.episode_rewards.append(sum([sample.reward for sample in runner.step_path]))
            for _ in range(self.gradient_steps):
                last_return, policy_loss, qf1_loss, qf2_loss = self.train_once(runner.step_itr,
                                              runner.step_path)
            self.evaluate_performance(
                runner.step_itr,
                self._obtain_evaluation_samples(runner.get_env_copy(), num_trajs=10))
            self.log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """
        """
        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
            samples = self.replay_buffer.sample(self.buffer_batch_size)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(itr, samples)
            self.update_targets()

        return 0, policy_loss, qf1_loss, qf2_loss