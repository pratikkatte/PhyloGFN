import torch
import torch.nn as nn
from src.model.mlp import MLP
from torch.distributions import Categorical
import numpy as np

"""
Categorical 1 : left and right edges model
"""


class EdgesModelCategorical(nn.Module):

    def __init__(self, edge_cat_cfg):
        super(EdgesModelCategorical, self).__init__()
        # model for first n-1 steps left and right edges
        self.lr_model = MLP(edge_cat_cfg.HEAD)
        # model last step edge
        self.root_edge_model = MLP(edge_cat_cfg.ROOT_EDGE_HEAD)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.lr_actions_list = np.arange(edge_cat_cfg.HEAD.OUTPUT_SIZE)
        self.root_edge_actions_list = np.arange(edge_cat_cfg.ROOT_EDGE_HEAD.OUTPUT_SIZE)

    def forward(self, summary_reps, left_trees, right_trees, sample, input_dict):

        # for now the representation is the concatenation of left and right tree
        rep = torch.cat([summary_reps, left_trees, right_trees], dim=1)
        batch_nb_seq = input_dict['batch_nb_seq']
        root_edges_flag = batch_nb_seq == 2
        first_edges_flag = batch_nb_seq > 2

        ret = {}
        edge_actions = torch.zeros(len(batch_nb_seq)).long().to(batch_nb_seq)
        if first_edges_flag.sum().item() > 0:
            first_edges_reps = rep[first_edges_flag]
            first_edges_logits = self.lr_model(first_edges_reps)
            first_edges_ret = {
                'logits': first_edges_logits
            }
            if sample:
                random_p = input_dict['list_random_action_prob'][first_edges_flag.detach().cpu().numpy()]
                actions = self.sample(first_edges_ret, self.lr_actions_list, random_p)
                ret['first_edges_actions'] = actions
                edge_actions[first_edges_flag] = actions
            ret['first_edges_ret'] = first_edges_ret

        if root_edges_flag.sum().item() > 0:
            root_edges_reps = rep[root_edges_flag]
            root_edges_logits = self.root_edge_model(root_edges_reps)
            root_edges_ret = {
                'logits': root_edges_logits
            }
            if sample:
                random_p = input_dict['list_random_action_prob'][root_edges_flag.detach().cpu().numpy()]
                actions = self.sample(root_edges_ret, self.root_edge_actions_list, random_p)
                ret['root_edges_actions'] = actions
                edge_actions[root_edges_flag] = actions
            ret['root_edges_ret'] = root_edges_ret

        if sample:
            ret['edge_actions'] = edge_actions
        return ret

    def sample(self, ret, actions_list, random_p):
        logits = ret['logits']
        distribution = Categorical(logits=logits)
        edge_action = distribution.sample()
        # random_action_prob
        rand_flag = np.random.rand(edge_action.shape[0]) <= random_p
        rand_num = rand_flag.sum()
        if rand_num > 0:
            rand_actions = torch.tensor(np.random.choice(actions_list, rand_num)).to(edge_action)
            edge_action[torch.tensor(rand_flag)] = rand_actions
        return edge_action

    def compute_log_pf(self, ret, input_dict):
        log_paths_pf = self.compute_log_path_pf(ret, input_dict)
        # batch_traj_idx and scatter_add would also work here
        batch_size = input_dict['batch_size']
        log_pf = log_paths_pf.reshape(batch_size, -1).sum(-1)
        return log_pf

    def compute_log_path_pf(self, ret, input_dict):

        batch_nb_seq = input_dict['batch_nb_seq']
        root_edges_flag = batch_nb_seq == 2
        first_edges_flag = batch_nb_seq > 2
        log_paths_pf = torch.zeros(len(input_dict['batch_input'])).to(input_dict['batch_input'])

        if first_edges_flag.sum().item() > 0:
            first_edges_ret = ret['first_edges_ret']
            first_edges_actions = input_dict['batch_edge_action'][first_edges_flag]
            log_p = self.logsoftmax(first_edges_ret['logits'])
            log_paths_pf[first_edges_flag] = log_p.gather(1, first_edges_actions.unsqueeze(-1)).squeeze(-1)

        if root_edges_flag.sum().item() > 0:
            root_edges_ret = ret['root_edges_ret']
            root_edges_actions = input_dict['batch_edge_action'][root_edges_flag]
            log_p = self.logsoftmax(root_edges_ret['logits'])
            log_paths_pf[root_edges_flag] = log_p.gather(1, root_edges_actions.unsqueeze(-1)).squeeze(-1)

        return log_paths_pf
