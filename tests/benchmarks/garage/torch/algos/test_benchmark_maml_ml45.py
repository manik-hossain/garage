"""This script creates a regression test over garage-MAML and ProMP-TRPO.

Unlike garage, baselines doesn't set max_path_length. It keeps steps the action
until it's done. So we introduced tests.wrappers.AutoStopEnv wrapper to set
done=True when it reaches max_path_length. We also need to change the
garage.tf.samplers.BatchSampler to smooth the reward curve.
"""
import datetime
import os.path as osp
import random
import sys

import numpy as np
import dowel
from dowel import logger as dowel_logger
import pytest
import torch
import tensorflow as tf

from metaworld.benchmarks import ML45WithPinnedGoal

from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline as PM_LinearFeatureBaseline
from meta_policy_search.envs.normalized_env import normalize as PM_normalize
from meta_policy_search.meta_algos.trpo_maml import TRPOMAML
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger as PM_logger

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner, SnapshotConfig
from garage.np.baselines import LinearFeatureBaseline
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy

from tests import benchmark_helper
import tests.helpers as Rh

test_garage = True
test_promp = False

# Same as promp:full_code/config/trpo_maml_config.json
hyper_parameters = {
    'hidden_sizes': [100, 100],
    'max_kl': 0.01,
    'inner_lr': 0.1,
    'gae_lambda': 1.0,
    'discount': 0.99,
    'max_path_length': 150,
    'fast_batch_size': 10,  # num of rollouts per task
    'meta_batch_size': 20,  # num of tasks
    'n_epochs': 1700,
    # 'n_epochs': 1,
    'n_trials': 3,
    'num_grad_update': 1,
    'n_parallel': 1,
    'inner_loss': 'log_likelihood'
}


class TestBenchmarkMAML:  # pylint: disable=too-few-public-methods
    """Compare benchmarks between garage and baselines."""

    @pytest.mark.huge
    def test_benchmark_maml(self):  # pylint: disable=no-self-use
        """Compare benchmarks between garage and baselines."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = './data/local/benchmarks/maml-ml45/%s/' % timestamp
        result_json = {}
        env_id = 'ML45'
        meta_env = ML45WithPinnedGoal.get_train_tasks()

        seeds = random.sample(range(100), hyper_parameters['n_trials'])
        task_dir = osp.join(benchmark_dir, env_id)
        plt_file = osp.join(benchmark_dir, '{}_benchmark.png'.format(env_id))
        promp_csvs = []
        garage_csvs = []

        for trial in range(hyper_parameters['n_trials']):
            seed = seeds[trial]
            trial_dir = task_dir + '/trial_%d_seed_%d' % (trial + 1, seed)
            garage_dir = trial_dir + '/garage'
            promp_dir = trial_dir + '/promp'

            if test_garage:
                # Run garage algorithm
                env = GarageEnv(normalize(meta_env, expected_action_scale=10.))
                garage_csv = run_garage(env, seed, garage_dir)
                garage_csvs.append(garage_csv)
                env.close()

            if test_promp:
                with tf.Graph().as_default():
                    # Run promp algorithm
                    promp_env = PM_normalize(meta_env)
                    promp_csv = run_promp(promp_env, seed, promp_dir)
                promp_csvs.append(promp_csv)
                promp_env.close()

        if test_garage and test_promp:
            benchmark_helper.plot_average_over_trials(
                [promp_csvs, promp_csvs, garage_csvs, garage_csvs],
                ys=[
                    'Step_0-AverageReturn', 'Step_1-AverageReturn',
                    'Update_0/AverageReturn', 'Update_1/AverageReturn'
                ],
                xs=[
                    'n_timesteps', 'n_timesteps', 'TotalEnvSteps',
                    'TotalEnvSteps'
                ],
                plt_file=plt_file,
                env_id=env_id,
                x_label='TotalEnvSteps',
                y_label='AverageReturn',
                names=['ProMP_0', 'ProMP_1', 'garage_0', 'garage_1'],
            )

            batch_size = hyper_parameters[
                'meta_batch_size'] * hyper_parameters['max_path_length']
            result_json[env_id] = benchmark_helper.create_json(
                [promp_csvs, promp_csvs, garage_csvs, garage_csvs],
                seeds=seeds,
                trials=hyper_parameters['n_trials'],
                xs=[
                    'n_timesteps', 'n_timesteps', 'TotalEnvSteps',
                    'TotalEnvSteps'
                ],
                ys=[
                    'Step_0-AverageReturn', 'Step_1-AverageReturn',
                    'Update_0/AverageReturn', 'Update_1/AverageReturn'
                ],
                factors=[batch_size] * 4,
                names=['ProMP_0', 'ProMP_1', 'garage_0', 'garage_1'])

            Rh.write_file(result_json, 'MAML')


def run_garage(env, seed, log_dir):
    """Create garage PyTorch MAML model and training.

    Args:
        env (GarageEnv): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    deterministic.set_seed(seed)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hyper_parameters['hidden_sizes'],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(env=env,
                    policy=policy,
                    baseline=baseline,
                    max_path_length=hyper_parameters['max_path_length'],
                    discount=hyper_parameters['discount'],
                    gae_lambda=hyper_parameters['gae_lambda'],
                    meta_batch_size=hyper_parameters['meta_batch_size'],
                    inner_lr=hyper_parameters['inner_lr'],
                    max_kl_step=hyper_parameters['max_kl'],
                    num_grad_updates=hyper_parameters['num_grad_update'])

    # Set up logger since we are not using run_experiment
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    dowel_logger.add_output(dowel.StdOutput())
    dowel_logger.add_output(dowel.CsvOutput(tabular_log_file))
    dowel_logger.add_output(dowel.TensorBoardOutput(log_dir))

    snapshot_config = SnapshotConfig(snapshot_dir=log_dir,
                                     snapshot_mode='all',
                                     snapshot_gap=1)

    runner = LocalRunner(snapshot_config=snapshot_config)
    runner.setup(algo, env, sampler_args=dict(n_envs=5))
    runner.train(n_epochs=hyper_parameters['n_epochs'],
                 batch_size=(hyper_parameters['fast_batch_size'] *
                             hyper_parameters['max_path_length']))

    dowel_logger.remove_all()

    return tabular_log_file


def run_promp(env, seed, log_dir):
    """Create ProMP model and training.

    Args:
        env (dict): Environment of the task.
        seed (int): Random positive integer for the trial.
        log_dir (str): Log dir path.

    Returns:
        str: Path to output csv file

    """
    deterministic.set_seed(seed)

    # configure logger
    PM_logger.configure(dir=log_dir,
                        format_strs=['stdout', 'log', 'csv', 'tensorboard'],
                        snapshot_mode='all')

    baseline = PM_LinearFeatureBaseline()

    policy = MetaGaussianMLPPolicy(
        name='meta-policy',
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=hyper_parameters['meta_batch_size'],
        hidden_sizes=hyper_parameters['hidden_sizes'],
    )

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=hyper_parameters['fast_batch_size'],
        meta_batch_size=hyper_parameters['meta_batch_size'],
        max_path_length=hyper_parameters['max_path_length'],
        parallel=hyper_parameters['n_parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=hyper_parameters['discount'],
        gae_lambda=hyper_parameters['gae_lambda'],
        normalize_adv=True,
    )

    algo = TRPOMAML(
        policy=policy,
        step_size=hyper_parameters['max_kl'],
        inner_type=hyper_parameters['inner_loss'],
        inner_lr=hyper_parameters['inner_lr'],
        meta_batch_size=hyper_parameters['meta_batch_size'],
        num_inner_grad_steps=hyper_parameters['num_grad_update'],
        exploration=False,
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=hyper_parameters['n_epochs'],
        num_inner_grad_steps=hyper_parameters['num_grad_update'],
    )

    trainer.train()
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    sampler.close()

    return tabular_log_file


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_garage = sys.argv[1] in ('both', 'garage')
        test_promp = sys.argv[1] in ('both', 'promp')

    test_cls = TestBenchmarkMAML()
    test_cls.test_benchmark_maml()
