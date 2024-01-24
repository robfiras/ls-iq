from copy import deepcopy

import torch
import numpy as np

from mushroom_rl.core import Dataset
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils import TorchUtils
from mushroom_rl.rl_utils.value_functions import compute_gae
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.trpo import TRPO
from mushroom_rl.utils.minibatches import minibatch_generator

from imitation_lib.utils import GailDiscriminatorLoss


class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning (GAIL) implementation.

    "Generative Adversarial Imitation Learning"
    Ho, J., & Ermon, S. (2016).

    """

    def __init__(self, mdp_info, policy_class, policy_params, sw, discriminator_params, critic_params,
                 train_D_n_th_epoch=3, ent_coeff=0., max_kl=.01, lam=0.97, n_epochs_line_search=10, n_epochs_cg=10,
                 cg_damping=1e-1, cg_residual_tol=1e-10, logging_iter=1, demonstrations=None, env_reward_frac=0.0,
                 backend="torch", state_mask=None, act_mask=None, use_next_states=False, critic_fit_params=None,
                 discriminator_fit_params=None, loss=GailDiscriminatorLoss()):

        # initialize TRPO agent
        policy = policy_class(**policy_params)
        super(GAIL, self).__init__(mdp_info, policy, critic_params,
                                   ent_coeff, max_kl, lam, n_epochs_line_search,
                                   n_epochs_cg, cg_damping, cg_residual_tol, backend=backend,
                                   critic_fit_params=critic_fit_params)

        # discriminator params
        self._discriminator_fit_params = (dict() if discriminator_fit_params is None
                                          else discriminator_fit_params)

        self._loss = loss
        discriminator_params.setdefault("loss", deepcopy(self._loss))
        self._D = Regressor(TorchApproximator, **discriminator_params)
        self._train_D_n_th_epoch = train_D_n_th_epoch

        self._env_reward_frac = env_reward_frac
        self._demonstrations = {key: torch.from_numpy(val).to(dtype=torch.float32, device=TorchUtils.get_device())
                                for key, val in demonstrations.items()}
        assert 0.0 <= env_reward_frac <= 1.0, "Environment reward must be between [0,1]"
        assert demonstrations is not None or env_reward_frac == 1.0, "No demonstrations have been loaded"

        # select which observations / actions to discriminate
        if not ("actions" in demonstrations):
            act_mask = []

        self._state_mask = np.arange(demonstrations["states"].shape[1]) \
            if state_mask is None else np.array(state_mask, dtype=np.int64)

        self._act_mask = np.arange(demonstrations["actions"].shape[1]) \
            if act_mask is None else np.array(act_mask, dtype=np.int64)

        self._use_next_state = use_next_states

        self._epoch_counter = 1
        assert logging_iter % train_D_n_th_epoch == 0, "logging_iter has to be a multitude of train_D_n_th_epoch."
        self._logging_iter = logging_iter

        if sw:
            self._sw = sw
            setattr(self._sw, '__deepcopy__', lambda self: None)    # dont need to be copyable, otherwise pickle error
        else:
            self._sw = None

        self._add_save_attr(
            discriminator_fit_params='pickle',
            _loss='torch',
            _train_n_th_epoch ='pickle',
            _D='mushroom',
            _env_reward_frac='pickle',
            _demonstrations='pickle!',
            _act_mask='pickle',
            _state_mask='pickle',
            _use_next_state='pickle',
            _train_D_n_th_epoch='pickle',
            _logging_iter="primitive"
        )

    def fit(self, dataset, **info):
        state, action, reward, next_state, absorbing, last = dataset.parse(to='torch')
        state, action, reward, next_state = state.to(torch.float32), action.to(torch.float32),\
                                            reward.to(torch.float32), next_state.to(torch.float32)

        state, next_state = self._agent_preprocess(state), self._agent_preprocess(next_state)

        # update reward
        if self._env_reward_frac < 1.0:
            # create reward from the discriminator (can use fraction of environment reward)
            reward_disc = self.make_discrim_reward(state, action, next_state)
            reward = reward * self._env_reward_frac + reward_disc * (1 - self._env_reward_frac)
            new_dataset = Dataset.from_array(state, action, reward, next_state, absorbing, last, backend='torch')
        else:
            new_dataset = dataset.copy()

        v_target, adv = compute_gae(self._V, state, next_state, reward, absorbing, last,
                                    self.mdp_info.gamma, self._lambda())
        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        adv = adv.detach()
        v_target = v_target.detach()

        # Policy update
        self._old_policy = deepcopy(self.policy)
        old_pol_dist = self._old_policy.distribution_t(state)
        old_log_prob = self._old_policy.log_prob_t(state, action).detach()

        TorchUtils.zero_grad(self.policy.parameters())
        loss = self._compute_loss(state, action, adv, old_log_prob)

        prev_loss = loss.item()

        # Compute gradient
        loss.backward()
        g = TorchUtils.get_gradient(self.policy.parameters())

        # Compute direction through conjugate gradient
        stepdir = self._conjugate_gradient(g, state, old_pol_dist)

        # Line search
        self._line_search(state, action, adv, old_log_prob, old_pol_dist, prev_loss, stepdir)

        # Value function update
        self._V.fit(state, v_target, **self._critic_fit_params)

        # fit discriminator
        self._fit_discriminator(state, action, next_state)

        # update statistics of standardizer
        self._update_agent_preprocessor(state)

        # make logging
        self._logging_sw(dataset, new_dataset, v_target, old_pol_dist)

        self._iter += 1

    def _fit_discriminator(self, plcy_obs, plcy_act, plcy_n_obs):
        plcy_obs = plcy_obs[:, self._state_mask]
        plcy_act = plcy_act[:, self._act_mask]
        plcy_n_obs = plcy_n_obs[:, self._state_mask]

        if self._iter % self._train_D_n_th_epoch == 0:

            # get batch of data to discriminate
            if self._use_next_state and not self._act_mask.size > 0:
                demo_obs, demo_n_obs = next(minibatch_generator(plcy_obs.shape[0],
                                                                self._demonstrations["states"],
                                                                self._demonstrations["next_states"]))
                demo_obs = demo_obs[:, self._state_mask]
                demo_n_obs = demo_n_obs[:, self._state_mask]
                input_states = torch.concatenate([plcy_obs, demo_obs])
                input_next_states = torch.concatenate([plcy_n_obs, demo_n_obs])
                inputs = (input_states, input_next_states)
            elif self._act_mask.size > 0 and not self._use_next_state:
                demo_obs, demo_act = next(minibatch_generator(plcy_obs.shape[0],
                                                              self._demonstrations["states"],
                                                              self._demonstrations["actions"]))
                demo_obs = demo_obs[:, self._state_mask]
                demo_act = demo_act[:, self._act_mask]
                input_states = torch.concatenate([plcy_obs, demo_obs])
                input_actions = torch.concatenate([plcy_act, demo_act])
                inputs = (input_states, input_actions)
            elif self._act_mask.size > 0 and self._use_next_state:
                raise ValueError("Discriminator with states, actions and next states "
                                 "as input currently not supported.")
            else:
                demo_obs = next(minibatch_generator(plcy_obs.shape[0],
                                                    self._demonstrations["states"]))[0]
                demo_obs = demo_obs[:, self._state_mask]
                input_states = torch.concatenate([plcy_obs, demo_obs])
                inputs = (input_states,)

            # create label targets
            plcy_target = torch.zeros((plcy_obs.shape[0], 1)).to(device=TorchUtils.get_device())
            demo_target = torch.ones((plcy_obs.shape[0], 1)).to(device=TorchUtils.get_device())

            targets = torch.concatenate([plcy_target, demo_target])

            self._D.fit(*inputs, targets, **self._discriminator_fit_params)

            self._discriminator_logging(inputs, targets)

    @staticmethod
    def logging_interval(func):
        def wrapper(self, *args, **kwargs):
            if self._sw and self._iter % self._logging_iter == 0:
                return func(self, *args, **kwargs)
        return wrapper

    @logging_interval
    def _discriminator_logging(self, inputs, targets):
        plcy_inputs, demo_inputs = self.divide_data_to_demo_and_plcy(inputs)

        # calculate the accuracies
        dout_exp = torch.sigmoid(self.discrim_output(*demo_inputs, apply_mask=False)).detach().cpu().numpy()
        dout_plcy = torch.sigmoid(self.discrim_output(*plcy_inputs, apply_mask=False)).detach().cpu().numpy()
        accuracy_exp = np.mean((dout_exp > 0.5))
        accuracy_gen = np.mean((dout_plcy < 0.5))
        self._sw.add_scalar('Accuracy of Discriminator on Policy Samples', accuracy_gen, self._iter)
        self._sw.add_scalar('Output of Discriminator on Policy Samples', np.mean(dout_plcy), self._iter)
        self._sw.add_scalar('Accuracy of Discriminator on Expert Samples', accuracy_exp, self._iter)
        self._sw.add_scalar('Output of Discriminator on Expert Samples', np.mean(dout_exp), self._iter)

        # calculate losses
        loss = deepcopy(self._loss)
        loss_eval = loss.forward(self._D(*inputs), targets)
        bernoulli_ent = torch.mean(loss.logit_bernoulli_entropy(self.discrim_output(*inputs, apply_mask=False)))
        neg_bernoulli_ent_loss = -loss.entcoeff * bernoulli_ent
        self._sw.add_scalar('Discriminator Loss', loss_eval, self._iter // self._train_D_n_th_epoch)
        self._sw.add_scalar('Bernoulli Entropy', bernoulli_ent, self._iter // self._train_D_n_th_epoch)
        self._sw.add_scalar('Neg. Bernoulli Entropy Loss (included in Discriminator Loss)',
                            neg_bernoulli_ent_loss, self._iter // self._train_D_n_th_epoch)

    @logging_interval
    def _logging_sw(self, dataset, new_dataset, v_target, old_pol_dist):
        v_pred = self._V(dataset.state)
        v_err = torch.mean(torch.square(v_pred - v_target))

        logging_ent = self.policy.entropy(dataset.state)
        new_pol_dist = self.policy.distribution(dataset.state)
        logging_kl = torch.distributions.kl.kl_divergence(old_pol_dist, new_pol_dist).mean()

        avg_rwd = dataset.compute_J().mean()
        avg_rwd_new = new_dataset.compute_J().mean()
        L = dataset.episodes_length.mean()

        self._sw.add_scalar('Mean Reward from Environment (for validation only)', avg_rwd, self._iter)
        self._sw.add_scalar('Mean Reward from Discriminator', avg_rwd_new, self._iter)
        self._sw.add_scalar('Mean Episode Length', L, self._iter)
        self._sw.add_scalar('Value Function Loss', v_err, self._iter)
        self._sw.add_scalar('Entropy', logging_ent, self._iter)
        self._sw.add_scalar('KL-Divergence', logging_kl, self._iter)

    def divide_data_to_demo_and_plcy(self, inputs):
        if self._act_mask.size > 0:
            input_states, input_actions = inputs
            plcy_obs = input_states[0:len(input_states)//2]
            plcy_act = input_actions[0:len(input_actions)//2]
            plcy_inputs = (plcy_obs, plcy_act)
            demo_obs = input_states[len(input_states)//2:]
            demo_act = input_actions[len(input_actions)//2:]
            demo_inputs = (demo_obs, demo_act)
        elif self._use_next_state:
            input_states, input_next_states = inputs
            plcy_obs = input_states[0:len(input_states)//2]
            plcy_n_obs = input_next_states[0:len(input_next_states)//2]
            plcy_inputs = (plcy_obs, plcy_n_obs)
            demo_obs = input_states[len(input_states)//2:]
            demo_n_obs = input_next_states[len(input_next_states)//2:]
            demo_inputs = (demo_obs, demo_n_obs)
        else:
            input_states = inputs[0]
            plcy_inputs = (input_states[0:len(input_states)//2],)
            demo_inputs = (input_states[len(input_states)//2:],)
        return plcy_inputs, demo_inputs

    def prepare_discrim_inputs(self, inputs, apply_mask=True):
        if self._use_next_state and not self._act_mask.size > 0:
            states, next_states = inputs
            states = states[:, self._state_mask] if apply_mask else states
            next_states = next_states[:, self._state_mask] if apply_mask else next_states
            inputs = (states, next_states)
        elif self._act_mask.size > 0 and not self._use_next_state:
            states, actions = inputs
            states = states[:, self._state_mask] if apply_mask else states
            actions = actions[:, self._act_mask] if apply_mask else actions
            inputs = (states, actions)
        elif self._act_mask.size > 0 and self._use_next_state:
            raise ValueError("Discriminator with states, actions and next states as input currently not supported.")
        else:
            states = inputs[0][:, self._state_mask] if apply_mask else inputs[0]
            inputs = (states,)
        return inputs

    def discrim_output(self, *inputs, apply_mask=True):
        inputs = self.prepare_discrim_inputs(inputs, apply_mask=apply_mask)
        d_out = self._D(*inputs)
        return d_out

    @torch.no_grad()
    def make_discrim_reward(self, state, action, next_state, apply_mask=True):
        if self._use_next_state:
            d = self.discrim_output(state, next_state, apply_mask=apply_mask)
        else:
            d = self.discrim_output(state, action, apply_mask=apply_mask)
        plcy_prob = 1/(1 + torch.exp(-d))     # sigmoid
        return torch.squeeze(-torch.log(1 - plcy_prob + 1e-8))
