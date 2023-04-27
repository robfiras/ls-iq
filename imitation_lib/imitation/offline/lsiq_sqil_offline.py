from copy import deepcopy
import torch
from torch.functional import F
import numpy as np
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.torch import to_float_tensor
from imitation_lib.imitation import LSIQ_SQIL


class LSIQ_SQIL_Offline(LSIQ_SQIL):

    def fit(self, dataset):
        raise AttributeError("This is the offline implementation of IQ, it is not supposed to use the fit function. "
                             "Use the fit_offline function instead.")

    def fit_offline(self, n_steps):

        for i in range(n_steps):

            # sample batch of same size from expert replay buffer and concatenate with samples from own policy
            assert self._act_mask.size > 0, "IQ-Learn needs demo actions!"
            demo_obs, demo_act, demo_nobs, demo_absorbing = next(minibatch_generator(self._batch_size(),
                                                                 self._demonstrations["states"],
                                                                 self._demonstrations["actions"],
                                                                 self._demonstrations["next_states"],
                                                                 self._demonstrations["absorbing"]))

            # prepare plcy data
            plcy_state = deepcopy(demo_obs.astype(np.float32)[:, self._state_mask])
            plcy_action = self.policy.draw_action(plcy_state)
            plcy_next_state = np.zeros_like(plcy_state)    # just used as a place holder
            plcy_absorbing = np.ones_like(demo_absorbing)    # all are absorbing, hence the placeholders

            # prepare data for IQ update
            input_states = to_float_tensor(np.concatenate([plcy_state, demo_obs.astype(np.float32)[:, self._state_mask]]))
            input_actions = to_float_tensor(np.concatenate([plcy_action, demo_act.astype(np.float32)]))
            input_n_states = to_float_tensor(np.concatenate([plcy_next_state,
                                                             demo_nobs.astype(np.float32)[:, self._state_mask]]))
            input_absorbing = to_float_tensor(np.concatenate([plcy_absorbing, demo_absorbing.astype(np.float32)]))
            is_expert = torch.concat([torch.zeros(len(plcy_state), dtype=torch.bool),
                                      torch.ones(len(plcy_state), dtype=torch.bool)])

            # make IQ update
            self.iq_update(input_states, input_actions, input_n_states, input_absorbing, is_expert)

            self._iter += 1
            self.policy.iter += 1


    def _lossQ(self, obs, act, next_obs, absorbing, is_expert):

        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        gamma = to_float_tensor(self.mdp_info.gamma).cuda() if self._use_cuda else to_float_tensor(self.mdp_info.gamma)
        absorbing = torch.tensor(absorbing).cuda() if self._use_cuda else absorbing
        current_Q = self._critic_approximator(obs, act, output_tensor=True)
        if not self._use_target:
            next_v = self.getV(next_obs)
        else:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs).detach()
        absorbing = torch.unsqueeze(absorbing, 1)
        y = (1 - absorbing) * gamma.detach() * self._Q_Q_multiplier * torch.clip(next_v, self._Q_min, self._Q_max)

        # define the rewards
        if self._treat_absorbing_states:
            r_max = (1 - absorbing) * ((1 / self._reg_mult)) \
                    + absorbing * (1 / (1 - gamma.detach())) * ((1 / self._reg_mult))
            r_min = (1 - absorbing) * (-(1 / self._reg_mult))\
                    + absorbing * (1 / (1 - gamma.detach())) * (-(1 / self._reg_mult))
        else:
            r_max = torch.ones_like(absorbing) * ((1 / self._reg_mult))
            r_min = torch.ones_like(absorbing) * (-(1 / self._reg_mult))

        r_max = r_max[is_expert]
        r_min = r_min[~is_expert]

        # expert part
        if self._loss_mode_exp == "bootstrap":
            if self._Q_exp_loss == "MSE":
                loss_term1 = torch.mean(torch.square(current_Q[is_expert] - (r_max + y[is_expert])))
            elif self._Q_exp_loss == "Huber":
                loss_term1 = F.huber_loss(current_Q[is_expert], (r_max + y[is_expert]))
            else:
                raise ValueError("Unknown loss.")
        elif self._loss_mode_exp == "fix":
            if self._Q_exp_loss == "MSE":
                loss_term1 = F.mse_loss(current_Q[is_expert], torch.ones_like(current_Q[is_expert]) * self._Q_max)
            elif self._Q_exp_loss == "Huber":
                loss_term1 = F.huber_loss(current_Q[is_expert], torch.ones_like(current_Q[is_expert]) * self._Q_max)
            else:
                raise ValueError("Unknown loss.")
        else:
            raise ValueError("Unknown expert loss mode.")

        # policy part
        if self._Q_exp_loss == "MSE":
            loss_term2 = torch.mean(torch.square(current_Q[~is_expert] - (r_min + y[~is_expert])))
        elif self._Q_exp_loss == "Huber":
            loss_term2 = F.huber_loss(current_Q[~is_expert], (r_min + y[~is_expert]))
        else:
            raise ValueError("Unknown loss.")

        # do the logging
        reward = (self._Q_Q_multiplier * current_Q - y)
        self.logging_loss(current_Q, y, reward, is_expert, obs, act, absorbing)

        # add gradient penalty if needed
        if self._gp_lambda > 0:
            with torch.no_grad():
                act_plcy, _ = self.policy.compute_action_and_log_prob_t(obs[is_expert])
            loss_gp = self._gradient_penalty(obs[is_expert], act[is_expert],
                                             obs[is_expert], act_plcy, self._gp_lambda)
        else:
            loss_gp = 0.0

        loss_Q = loss_term1 + loss_term2 + loss_gp
        self.update_Q_parameters(loss_Q)

        grads = []
        for param in self._critic_approximator.model.network.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        norm = grads.norm(dim=0, p=2)
        if self._iter % self._logging_iter == 0:
            self.sw_add_scalar('Gradients/Norm2 Gradient LossQ wrt. Q-parameters', norm, self._iter)

        return loss_term1, loss_term2, 0.0