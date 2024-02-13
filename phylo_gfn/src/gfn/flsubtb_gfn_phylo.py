import math
import torch
import numpy as np
from src.gfn.base import GeneratorBase
from src.model.build import build_model
from torch.nn import functional as F
from copy import deepcopy

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0)
}


class FLSubTBGFlowNetGenerator(GeneratorBase):
    def __init__(self, gfn_cfg, state2input, env, device=None):
        super().__init__(gfn_cfg, state2input, env)
        self.gfn_model_cfg = gfn_model_cfg = gfn_cfg.MODEL
        if device is None:
            self.all_device = [torch.device('cpu')]
        elif isinstance(device, torch.device):
            self.all_device = [device]
        elif isinstance(device, list) and all([isinstance(device_, torch.device) for device_ in device]):
            self.all_device = device
        else:
            raise ValueError(f'Unrecognized {device}')

        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE

        # load model
        self.env_type = env.type
        self.model = build_model(gfn_cfg, self.env_type)
        self.model.to(self.all_device[0])

        # target network
        self.use_target_net = gfn_model_cfg.USE_TARGET_NET
        if self.use_target_net:
            self.update_target_net()

        # maintain two separate parameter groups for the main model and the partition
        self.opt = torch.optim.Adam(
            [{'params': list(self.model.parameters()), 'lr': gfn_model_cfg.LR_MODEL}],
            weight_decay=gfn_model_cfg.L2_REG, betas=(0.9, 0.999), amsgrad=True)

        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.grad_clip = gfn_model_cfg.GRAD_CLIP
        self.grad_norm = lambda model: math.sqrt(sum(
            [p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))
        self.param_norm = lambda model: math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))
        self.loss = 0
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.gradient_clipping_params = list(self.parameters())

        # subtb in convolution style
        traj = self.env.sample(1)[0]
        traj_length = len(traj.transitions)
        self.lamb = gfn_model_cfg.SUBTB_LAMBDA

        path_filters, flow_filters, filter_mask = [], [], []
        subtb_weights = []
        for size in range(1, traj_length + 1):
            # filter for summing up pf, pb and partial rewards along a subtrajectory
            path_filter = torch.cat([torch.ones(1, 1, size), torch.zeros(1, 1, traj_length - size)], dim=-1)
            path_filters.append(path_filter)
            # filter for start state flow subtracting end state flow
            flow_filter = torch.zeros(1, 1, traj_length + 1)
            flow_filter[0, 0, size] = -1.
            if not self.use_target_net:
                flow_filter[0, 0, 0] = 1.
            flow_filters.append(flow_filter)
            # mask to remove the paddings
            filter_mask.append(torch.tensor(np.array([False] * (traj_length - size + 1) + [True] * (size - 1))))
            subtb_weights.extend([self.lamb ** size] * (traj_length - size + 1))
        self.path_filters = torch.cat(path_filters, dim=0).to(self.all_device[0])
        self.flow_filters = torch.cat(flow_filters, dim=0).to(self.all_device[0])
        self.filter_mask = torch.stack(filter_mask, dim=0).to(self.all_device[0])
        self.subtb_weights = torch.tensor(np.array(subtb_weights, dtype=np.float32)).to(self.all_device[0])

    def accumulate_loss_amp(self, input_batch, factor=1.0, shared_array=None):

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = self.get_loss(input_batch, shared_array)
            loss = loss / factor
        self.grad_scaler.scale(loss).backward()
        self.loss += loss.item()

    def update_model_amp(self):
        info = {'grad_norm': self.grad_norm(self.model),
                'param_norm': self.param_norm(self.model),
                'loss': self.loss}

        self.grad_scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.grad_scaler.step(self.opt)
        self.grad_scaler.update()
        self.opt.zero_grad()
        self.loss = 0
        return info

    def accumulate_loss(self, input_batch, factor=1.0, shared_array=None):
        loss = self.get_loss(input_batch, shared_array)
        loss = loss / factor
        loss.backward()
        self.loss += loss.item()  # to free up GPU

    def update_model(self):

        info = {'grad_norm': self.grad_norm(self.model),
                'param_norm': self.param_norm(self.model),
                'loss': self.loss}
        torch.nn.utils.clip_grad_norm_(self.gradient_clipping_params, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad()
        self.loss = 0
        return info

    def save(self, path):
        # we need to include optimizers state_dict as well
        torch.save({
            'generator_state_dict': self.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }, path)

    def load(self, path):
        # loading all state dicts
        all_state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(all_state_dict['generator_state_dict'])
        self.opt.load_state_dict(all_state_dict['opt_state_dict'])
        self.grad_scaler.load_state_dict(all_state_dict['grad_scaler_state_dict'])

    def train_step(self, input_batch, shared_array=None):
        self.opt.zero_grad()
        loss = self.get_loss(input_batch, shared_array)  # compute loss for all trajectories at once
        loss.backward()
        info = {'grad_norm': self.grad_norm(self.model),
                'param_norm': self.param_norm(self.model),
                'loss': loss.detach().cpu().numpy().tolist()}
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return info

    def compute_log_Z(self, scale=None):

        if self.condition_on_scale:
            states = [self.env.get_initial_state() for _ in range(scale.shape[0])]
        else:
            states = [self.env.get_initial_state()]
        batch_combined_dict = self.state2input.states2inputs(states)
        if self.condition_on_scale:
            batch_combined_dict['scale'] = scale
            batch_combined_dict['batch_size'] = len(states)
        ret = self(batch_combined_dict)
        return ret['log_flow']

    def Z(self, scale=None):
        return np.exp(self.log_Z(scale))

    def log_Z(self, scale=None):
        with torch.no_grad():
            log_z = self.compute_log_Z(scale)
            return log_z.cpu().numpy()

    def compute_log_paths_pf(self, logits, seq_len_mask, input_dict):
        """
        parse forward output and compute log pf
        """
        if self.env_type == 'ONE_STEP_BINARY_TREE':
            pairwise_mask_tensor = input_dict['pairwise_mask_tensor']
            logits = logits.masked_fill(pairwise_mask_tensor, float('-inf'))
            batch_action = input_dict['batch_pairwise_action']
        else:
            logits = logits.masked_fill(seq_len_mask, float('-inf'))
            batch_action = input_dict['batch_action']
        log_path_pfs = torch.nn.functional.log_softmax(logits, dim=1).gather(
            1, batch_action.unsqueeze(-1)).squeeze(-1)
        return log_path_pfs

    def compute_log_pf(self, generator_ret_dict, input_dict):
        """
        parse forward output and compute log pf
        used in loss computation as well as correlation evaluation
        :return: log pf of all trajectories
        """
        logits = generator_ret_dict['logits']
        seq_len_mask = generator_ret_dict['mask']

        log_path_pfs = self.compute_log_paths_pf(logits, seq_len_mask, input_dict)
        batch_size = input_dict['batch_size']
        log_pf = torch.zeros(batch_size, dtype=torch.float32).to(logits.device). \
            scatter_add_(0, input_dict['batch_traj_idx'], log_path_pfs)
        return log_pf

    def get_loss(self, batch_combined_dict, shared_array=None):

        ret = self(batch_combined_dict)
        logits = ret['logits']
        log_flow = ret['log_flow']
        seq_len_mask = ret['mask']
        log_paths_pf = self.compute_log_paths_pf(logits, seq_len_mask, batch_combined_dict)
        trajs_num = batch_combined_dict['batch_size']
        log_paths_pb = batch_combined_dict['batch_pb_log']

        log_paths_pf = log_paths_pf.reshape(trajs_num, -1)
        traj_length = log_paths_pf.shape[1]
        log_flow = log_flow.reshape(trajs_num, -1)
        log_flow_reward = F.pad(log_flow, (0, 1))  # log_flow_rewards last dim is trajs_length + 1

        log_partial_rewards = batch_combined_dict['log_partial_rewards']
        fl_c = self.env.fl_c
        if 'scale' in batch_combined_dict:
            scale = batch_combined_dict['scale'] + 0.0
            log_partial_rewards /= scale.reshape(-1, 1)
            fl_c = fl_c / scale.reshape(-1, 1)
        edges_log_partial_rewards = log_partial_rewards[:, :-1] - fl_c - log_partial_rewards[:, 1:]

        log_paths_pf = F.pad(log_paths_pf, (0, traj_length - 1, 0, 0))
        log_paths_pb = F.pad(log_paths_pb, (0, traj_length - 1, 0, 0))
        edges_log_partial_rewards = F.pad(edges_log_partial_rewards, (0, traj_length - 1, 0, 0))
        # convolution output: nb_trajs x nb_filters x traj_length  (nb_filters equals traj_length)
        sub_pf = F.conv1d(log_paths_pf.unsqueeze(1), self.path_filters)
        sub_pb = F.conv1d(log_paths_pb.unsqueeze(1), self.path_filters)
        sub_log_partial_rewards = F.conv1d(edges_log_partial_rewards.unsqueeze(1), self.path_filters)

        # flattened and removed invalid entries from the output
        sub_pf = sub_pf[~self.filter_mask.repeat(trajs_num, 1, 1)]
        sub_pb = sub_pb[~self.filter_mask.repeat(trajs_num, 1, 1)]
        sub_log_partial_rewards = sub_log_partial_rewards[~self.filter_mask.repeat(trajs_num, 1, 1)]

        # state flow subtraction
        if self.use_target_net:
            target_ret = self(batch_combined_dict, with_target_net=True)
            target_log_flow = target_ret['log_flow']
            target_log_flow = target_log_flow.reshape(trajs_num, -1)
            target_log_flow_reward_with_padding = F.pad(target_log_flow, (0, traj_length, 0, 0))
            neg_end_state_flow = F.conv1d(target_log_flow_reward_with_padding.unsqueeze(1), self.flow_filters)
            start_state_flow = log_flow_reward[:, :traj_length].unsqueeze(1).repeat(1, neg_end_state_flow.shape[1], 1)
            state_flow_first_subtract_last = start_state_flow + neg_end_state_flow
        else:
            log_flow_reward = F.pad(log_flow_reward, (0, traj_length - 1, 0, 0))
            state_flow_first_subtract_last = F.conv1d(log_flow_reward.unsqueeze(1), self.flow_filters)
        state_flow_first_subtract_last = state_flow_first_subtract_last[~self.filter_mask.repeat(trajs_num, 1, 1)]

        err = sub_pf - sub_pb + state_flow_first_subtract_last + sub_log_partial_rewards
        err = err.pow(2).reshape(trajs_num, -1)
        weights = self.subtb_weights.unsqueeze(0).repeat(trajs_num, 1)
        loss = (err * weights).sum(-1) / weights.sum(-1)

        if shared_array is not None:
            # shared_array: multiprocessing.sharedctypes.SynchronizedArray
            shared_array.acquire()
            counter = int(shared_array[0])
            loss_list = loss.detach().cpu().numpy().tolist()
            nb_vals = len(loss_list)
            shared_array[1 + counter: 1 + counter + nb_vals] = loss_list
            shared_array[0] = float(counter + nb_vals)
            shared_array.release()

        final_loss = loss.mean()
        return final_loss

    def update_target_net(self):
        self.target_net = deepcopy(self.model)
        self.target_net.to(self.all_device[0])
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

    def forward(self, inputs, rollout_scale=None, with_target_net=False):
        """
        assume all input states have the same input/output dimension and the same state type
        """
        # inputs can be a batch of mixed intermediate and non-intermediate states
        if isinstance(inputs, dict) and 'batch_input' in inputs:
            # for computing loss on sampled trajectories, dense version
            input_dict = inputs
        elif isinstance(inputs, list):
            input_dict = self.state2input.states2inputs(inputs)
            if self.condition_on_scale:
                B = input_dict['batch_input'].shape[0]
                input_dict['scale'] = torch.tensor(np.array([rollout_scale] * B, dtype=np.float32))
        elif isinstance(inputs, dict) and 'batch_input_coo_idx' in inputs:
            # for computing loss on sampled trajectories, sparse version,
            # not used as of currently, but may come in handy in the future (to save RAM for example)
            input_dict = inputs
            batch_input = torch.sparse_coo_tensor(
                indices=input_dict['batch_input_coo_idx'],
                values=torch.ones(input_dict['batch_input_coo_idx'].shape[1]),
                size=[input_dict['nb_state'], input_dict['max_nb_seq'], self.gfn_model_cfg.TRANSFORMER.INPUT_SIZE]
            )
            # torch Linear (or torch.matmul) cannot handle 3D sparse tensor
            input_dict['batch_input'] = batch_input.to_dense()
        else:
            raise ValueError('Unrecognized inputs')

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                if k in ['scale', 'batch_input']:
                    # for model forward
                    input_dict[k] = v.to(self.all_device[0]).float()
                elif k in ['batch_nb_seq', 'batch_intermediate_flag']:
                    # for model forward
                    input_dict[k] = v.to(self.all_device[0])
                else:
                    # for the loss computation
                    input_dict[k] = v.to(self.all_device[-1])

        if with_target_net:
            return self.target_net(**input_dict)
        else:
            return self.model(**input_dict)
