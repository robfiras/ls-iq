import os
from time import perf_counter
from contextlib import contextmanager

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.core.logger.logger import Logger


from imitation_lib.imitation import LSIQ
from imitation_lib.utils import FullyConnectedNetwork
from imitation_lib.utils import prepare_expert_data, BestAgentSaver


from experiment_launcher import run_experiment


def _create_agent(mdp, expert_data, sw, lr_critic, lr_actor, plcy_loss_mode,
                  regularizer_mode, use_target, lossQ_type, use_cuda, tau,
                  learnable_alpha, init_alpha, reg_mult, Q_exp_loss, gamma,
                  loss_mode_exp, log_std_min, log_std_max, delay_Q, n_fits,
                  logging_iter):

    # calculate the minimum and maximum Q-function
    Q_max = 1.0 / (reg_mult * (1 - gamma))
    Q_min = - 1.0 / (reg_mult * (1 - gamma))

    # Settings
    initial_replay_size = 10000
    max_replay_size = 1000000
    batch_size = 256     # the real batch size is double the size as an expert batch is going to be added
    warmup_transitions = 15000

    lr_alpha = 2e-6
    weight_decay_actor = 0.0
    weight_decay_critic = 0.0

    target_entropy = -22.0

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_output_shape = (mdp.info.action_space.shape[0]*2,)
    actor_params = dict(network=FullyConnectedNetwork,
                        n_features=[256, 256],
                        input_shape=actor_input_shape,
                        output_shape=actor_output_shape,
                        activations=["relu", "relu", "identity"],
                        use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor, 'weight_decay': weight_decay_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=FullyConnectedNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic, 'weight_decay': weight_decay_critic}},
                         n_features=[256, 256],
                         input_shape=critic_input_shape,
                         activations=["relu", "relu", "identity"],
                         squeeze_out=False,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # create IQfO agent
    agent = LSIQ(mdp_info=mdp.info, batch_size=batch_size, initial_replay_size=initial_replay_size,
                 max_replay_size=max_replay_size, demonstrations=expert_data, sw=sw, use_target=use_target,
                 warmup_transitions=warmup_transitions, tau=tau, lr_alpha=lr_alpha, actor_params=actor_params,
                 actor_optimizer=actor_optimizer, critic_params=critic_params, delay_Q=delay_Q, lossQ_type=lossQ_type,
                 target_entropy=target_entropy, critic_fit_params=None, plcy_loss_mode=plcy_loss_mode,
                 regularizer_mode=regularizer_mode, learnable_alpha=learnable_alpha, init_alpha=init_alpha,
                 reg_mult=reg_mult, Q_min=Q_min, Q_max=Q_max, log_std_min=log_std_min, log_std_max=log_std_max,
                 loss_mode_exp=loss_mode_exp, Q_exp_loss=Q_exp_loss, n_fits=n_fits, logging_iter=logging_iter)

    return agent


def experiment(env_id: str = "HalfCheetah-v2",
               n_epochs: int = 500,
               n_steps_per_epoch: int = 10000,
               n_steps_per_fit: int = 1,
               n_eval_episodes: int = 50,
               n_epochs_save: int = 100,
               logging_iter: int = 100,
               expert_data_path: str = None,
               use_cuda: bool = False,
               lr_critic: float = 3e-4,
               lr_actor: float = 3e-5,
               results_dir: str = "./logs",
               plcy_loss_mode: str = "value",
               regularizer_mode: str = "exp_and_plcy",
               reg_mult: float = 0.5,
               Q_exp_loss: str = "MSE",
               n_fits: int = 1,
               loss_mode_exp: str = "fix",
               log_std_min: float = -5.0,
               log_std_max: float = 2.0,
               learnable_alpha: bool = False,
               use_target: bool = True,
               init_alpha: float = 0.001,
               tau: float = 0.005,
               delay_Q: int = 1,
               lossQ_type: str = "sqil_like",
               gamma: float = 0.99,
               horizon: int = 1000,
               seed: int = 0):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    logger_stoch = Logger(results_dir=results_dir, log_name="stochastic_logging", seed=seed, append=True)
    logger_deter = Logger(results_dir=results_dir, log_name="deterministic_logging", seed=seed, append=True)

    results_dir = os.path.join(results_dir, str(seed))

    env_params = dict(name=env_id, horizon=horizon, gamma=gamma)

    mdp = Gym(**env_params)

    # load expert data
    expert_data = prepare_expert_data(data_path=expert_data_path)

    # logging stuff
    tb_writer = SummaryWriter(log_dir=results_dir)
    agent_saver = BestAgentSaver(save_path=results_dir, n_epochs_save=n_epochs_save)

    # create agent and core
    agent = _create_agent(mdp, expert_data,  sw=tb_writer, lr_critic=lr_critic, lr_actor=lr_actor,
                          plcy_loss_mode=plcy_loss_mode, regularizer_mode=regularizer_mode,
                          use_cuda=use_cuda, use_target=use_target, lossQ_type=lossQ_type,
                          delay_Q=delay_Q, tau=tau, learnable_alpha=learnable_alpha, init_alpha=init_alpha,
                          reg_mult=reg_mult, gamma=gamma, Q_exp_loss=Q_exp_loss,
                          loss_mode_exp=loss_mode_exp, log_std_min=log_std_min,
                          n_fits=n_fits, log_std_max=log_std_max, logging_iter=logging_iter)

    core = Core(agent, mdp)

    # iqfo train loop
    for epoch in range(n_epochs):
        with catchtime() as t:
            # training
            core.learn(n_steps=n_steps_per_epoch, n_steps_per_fit=n_steps_per_fit, quiet=True)
            print('Epoch %d | Time %fs ' % (epoch + 1, float(t())))

            # evaluate with deterministic policy
            agent.policy.use_mean = True
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            R_mean = np.mean(compute_J(dataset))
            J_mean = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            logger_deter.log_numpy(Epoch=epoch, R_mean=R_mean, J_mean=J_mean, L=L)
            tb_writer.add_scalar("Eval_R-deterministic", R_mean, epoch)
            tb_writer.add_scalar("Eval_J-deterministic", J_mean, epoch)
            tb_writer.add_scalar("Eval_L-deterministic", L, epoch)
            agent.policy.use_mean = False

            # evaluate with stochastic policy
            dataset = core.evaluate(n_episodes=n_eval_episodes)
            R_mean_stoch = np.mean(compute_J(dataset))
            J_mean_stoch = np.mean(compute_J(dataset, gamma=gamma))
            L = np.mean(compute_episodes_length(dataset))
            logger_stoch.log_numpy(Epoch=epoch, R_mean=R_mean_stoch, J_mean=J_mean_stoch, L=L)
            tb_writer.add_scalar("Eval_R-stochastic", R_mean_stoch, epoch)
            tb_writer.add_scalar("Eval_J-stochastic", J_mean_stoch, epoch)
            tb_writer.add_scalar("Eval_L-stochastic", L, epoch)

            print("R_mean (deter): %f | R_mean (stoch): %f" % (R_mean, R_mean_stoch))

            # save agent if needed
            agent_saver.save(core.agent, J_mean)

    agent_saver.save_curr_best_agent()
    print("Finished.")

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


if __name__ == "__main__":

    # Leave unchanged
    run_experiment(experiment)
