import math
import torch
import numpy as np
from src.gfn.base import GeneratorBase
from src.model.edges_model.build import build_edge_model
from src.model.build import build_model
from src.model.mlp import MLP
from src.utils.lr_schedulers.build import build_scheduler

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0)
}


class TBGFlowNetGenerator(GeneratorBase):
    def __init__(self, gfn_cfg, state2input, env, device=None):
        super().__init__(gfn_cfg, state2input, env)
        self.gfn_model_cfg = gfn_model_cfg = gfn_cfg.MODEL
        # self.apply_fast_Z = gfn_model_cfg.TB_FAST_Z

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
        if not self.env.parsimony_problem:
            self.edges_model = build_edge_model(gfn_cfg)
            self.edges_model.to(self.all_device[0])

        params = [
            {'params': list(self.parameters()), 'lr': gfn_model_cfg.LR_MODEL}
        ]

        # trajs = env.sample(1000)
        # max_reward_seen = np.max([x.reward['log_reward'] for x in trajs])
        # self.max_reward_seen = max_reward_seen

        # if condition on scale, Z is calculated using MLP, otherwise Z is directly calculated
        if self.condition_on_scale:
            self.Z_module = MLP(gfn_cfg.MODEL.Z_MLP).to(self.all_device[0])
            params.append({'params': list(self.Z_module.parameters()), 'lr': gfn_model_cfg.LR_Z})
        else:
            self._Z = torch.nn.Parameter(  # in log
                torch.ones(256, device=self.all_device[0]) * gfn_model_cfg.Z_PARTITION_INIT / 256, requires_grad=True
            )
            params.append(
                {'params': [self._Z], 'lr': gfn_model_cfg.LR_Z}
            )
        # maintain two separate parameter groups for the main model and the partition
        self.opt = torch.optim.Adam(
            params,
            weight_decay=gfn_model_cfg.L2_REG, betas=(0.9, 0.999), amsgrad=True)

        # learning rate scheduler, important for the likelihood problem
        if gfn_cfg.MODEL.USE_LR_SCHEDULER:
            self.scheduler = build_scheduler(self.opt, gfn_cfg.MODEL.LR_SCHEDULER)
        else:
            self.scheduler = None

        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.grad_clip = gfn_model_cfg.GRAD_CLIP
        self.grad_norm = lambda model: math.sqrt(sum(
            [p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))
        self.param_norm = lambda model: math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))
        self.loss = 0
        self.grad_scaler = torch.cuda.amp.GradScaler()
        if not self.condition_on_scale:
            self.gradient_clipping_params = list(self.model.parameters())
        else:
            self.gradient_clipping_params = list(self.parameters())
        if not self.env.parsimony_problem:
            self.gradient_clipping_params += list(self.edges_model.parameters())

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
        to_save = {
            'generator_state_dict': self.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        if self.scheduler is not None:
            to_save['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(to_save, path)

    def load(self, path):
        # loading all state dicts
        all_state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(all_state_dict['generator_state_dict'])
        self.opt.load_state_dict(all_state_dict['opt_state_dict'])
        self.grad_scaler.load_state_dict(all_state_dict['grad_scaler_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(all_state_dict['scheduler_state_dict'])

        # manual LR setting upon resuming experiments
        # if not self.gfn_model_cfg.USE_LR_SCHEDULER:
        #
        #     if self.opt.param_groups[0]['lr'] != self.gfn_model_cfg.LR_MODEL:
        #         self.opt.param_groups[0]['lr'] = self.gfn_model_cfg.LR_MODEL
        #
        #     if self.opt.param_groups[1]['lr'] != self.gfn_model_cfg.LR_Z:
        #         self.opt.param_groups[1]['lr'] = self.gfn_model_cfg.LR_Z

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
            scale_embs = self.model.scale_module(scale)
            log_z = self.Z_module(scale_embs)
            return log_z
        else:
            return self._Z.sum()

    def Z(self, scale=None):
        return np.exp(self.log_Z(scale))

    def log_Z(self, scale=None):
        with torch.no_grad():
            log_z = self.compute_log_Z(scale)
            # check
            # if len(log_z) > 0:
            #     log_z = log_z[0]
            return log_z.cpu().numpy()

    def compute_log_pf(self, generator_ret_dict, input_dict):
        """
        parse forward output and compute log pf
        used in loss computation as well as correlation evaluation
        :return: log pf of all trajectories
        """
        logits = generator_ret_dict['logits']

        if self.env_type == 'ONE_STEP_BINARY_TREE':  # this also covers the one_step_likelihood env type
            pairwise_mask_tensor = input_dict['pairwise_mask_tensor']
            logits = logits.masked_fill(pairwise_mask_tensor, float('-inf'))
            batch_action = input_dict['batch_pairwise_action']
        else:
            mask = generator_ret_dict['mask']
            logits = logits.masked_fill(mask, float('-inf'))
            batch_action = input_dict['batch_action']
        log_path_pfs = torch.nn.functional.log_softmax(logits, dim=1).gather(
            1, batch_action.unsqueeze(-1)).squeeze(-1)
        # group by sum aggregate according to trajectory ids
        batch_size = input_dict['batch_size']
        log_pf = torch.zeros(batch_size, dtype=torch.float32).to(logits.device). \
            scatter_add_(0, input_dict['batch_traj_idx'], log_path_pfs)

        if not self.env.parsimony_problem:
            # edge_model compute_log_pf should probably be put inside the generator
            # note, the backward prob of edge sampling is annihilated
            edges_log_pf = self.edges_model.compute_log_pf(generator_ret_dict['edges_ret'], input_dict)
            log_pf = log_pf + edges_log_pf
        return log_pf

    def get_loss(self, batch_combined_dict, shared_array=None):

        ret = self(batch_combined_dict, compute_edge=True)
        log_pf = self.compute_log_pf(ret, batch_combined_dict)
        # we directly load log reward to avoid overflow issues
        rewards_log = batch_combined_dict['batch_log_reward']

        if self.condition_on_scale:
            scale = batch_combined_dict['scale']
            rewards_log /= scale.squeeze()

        log_z = self.compute_log_Z(batch_combined_dict.get('scale')).reshape(-1).to(log_pf.device)
        loss = (log_z + log_pf - rewards_log - batch_combined_dict['batch_pb_log']).pow(2)

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

    def forward(self, inputs, rollout_scale=None, compute_edge=False):
        """
        assume all input states have the same input/output dimension and the same state type
        """
        # inputs can be a batch of mixed intermediate and non-intermediate states
        if isinstance(inputs, dict) and 'batch_input' in inputs:
            # for computing loss on sampled trajectories, dense version
            input_dict = inputs
        elif isinstance(inputs, list):
            # not being used by anything really
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

        ret = self.model(**input_dict)

        # for likelihood problem continue the computation
        if not self.env.parsimony_problem and compute_edge:
            ret['edges_ret'] = self.calculate_edge_data(ret['summary_reps'], ret['trees_reps'], False, input_dict)

        return ret

    def calculate_edge_data(self, summary_reps, tree_representations, sample, input_dict):
        # TODO CLEAN THE INPUT ARGS
        """
        perform forward inference on edge model
        :param summary_reps: summary token representation obtained by the main phylo model
        :param tree_representations: tree representation obtained by the main phylo model
        :param sample: whether to perform sampling
        :param input_dict: additional inputs for edge model (batch_nb_seq, tree_actions, and possibly list_rand_action_prob)
        :return:
        """
        # get tree pairs
        tree_pairs = self.env.retrieve_tree_pairs(input_dict['batch_nb_seq'], input_dict['batch_action'])
        left_trees_indices = [x[0] for x in tree_pairs]
        right_trees_indices = [x[1] for x in tree_pairs]
        n = len(tree_pairs)
        left_trees_reps = tree_representations[torch.arange(n), left_trees_indices]
        right_trees_reps = tree_representations[torch.arange(n), right_trees_indices]
        edges_ret = self.edges_model(summary_reps, left_trees_reps, right_trees_reps, sample, input_dict)
        return edges_ret
