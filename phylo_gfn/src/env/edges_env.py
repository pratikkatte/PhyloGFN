import itertools
import numpy as np
import random
import math


class EdgeEnvCategorical(object):

    def __init__(self, edge_cat_cfg):
        self.categorical_bin_size = edge_cat_cfg.BIN_SIZE
        self.categorical_bins = edge_cat_cfg.BINS
        self.lr_actions_pairs = list(
            itertools.product(np.arange(self.categorical_bins), np.arange(self.categorical_bins)))
        self.lr_actions_pairs_indices = {
            pair: idx for idx, pair in enumerate(self.lr_actions_pairs)
        }
        self.max_edge_length = float(self.categorical_bins * self.categorical_bin_size)

    def lr_actions_2_edges(self, edge_action):
        """

        :param edge_action: one action in N^2 of (l,r) pairs
        :return:
        """
        left_length_action, right_length_action = self.lr_actions_pairs[edge_action]
        left_length = (left_length_action + 1) * self.categorical_bin_size
        right_length = (right_length_action + 1) * self.categorical_bin_size
        return left_length, right_length

    def root_edge_actions_2_edges(self, edge_action):
        """
        calculate l r edge length at root level, since at the root we only care about total length
        since at the root level we only care about the total length, return l/2 for left and right
        :param edge_action: one action in N
        :return:
        """
        edge_length = (1 + edge_action) * self.categorical_bin_size
        return edge_length/2, edge_length/2

    def lr_edges_2_actions(self, left_length, right_length):
        left_length_action = int(round((left_length / self.categorical_bin_size))) - 1
        right_length_action = int(round((right_length / self.categorical_bin_size))) - 1
        action = self.lr_actions_pairs_indices[(left_length_action, right_length_action)]
        return action

    def root_edge_2_actions(self, left_length, right_length):
        length = left_length + right_length
        action = int(round((length / self.categorical_bin_size))) - 1
        return action

    def generate_random_actions_lr(self):
        edge_action = random.randint(0, len(self.lr_actions_pairs) - 1)
        return edge_action

    def generate_random_actions_root(self):
        edge_action = random.randint(0, self.categorical_bins - 1)
        return edge_action

    def actions_2_edges(self, action, **other_input):

        if other_input['at_root']:
            return self.root_edge_actions_2_edges(action)
        else:
            return self.lr_actions_2_edges(action)

    def edges_2_actions(self, left_length, right_length, **other_input):

        if other_input['at_root']:
            action = self.root_edge_2_actions(left_length, right_length)
        else:
            action = self.lr_edges_2_actions(left_length, right_length)
        return action

    def generate_random_actions(self, **other_input):
        if other_input['at_root']:
            return self.generate_random_actions_root()
        else:
            return self.generate_random_actions_lr()


# class EdgeEnvContinuous(object):
#
#     def __init__(self, edge_cont_cfg):
#         self.max_edge_length = edge_cont_cfg.MAX_EDGE_LENGTH
#
#     def actions_2_edges(self, edge_action):
#         left_edge_action, right_edge_action = edge_action
#         left_length = left_edge_action * self.max_edge_length
#         right_length = right_edge_action * self.max_edge_length
#         return left_length, right_length
#
#     def edges_2_actions(self, left_length, right_length):
#         left_edge_action = left_length / self.max_edge_length
#         right_edge_action = right_length / self.max_edge_length
#         return left_edge_action, right_edge_action
#
#     def generate_random_actions(self):
#         left_edge_action, right_edge_action = random.random(), random.random()
#         return left_edge_action, right_edge_action
#
#     def get_possible_splits(self, total_edge_length):
#         """
#         Given total edge length, find possible range of left cut
#         :param total_edge_length:
#         :return:
#         """
#         left_edge_min = max(0.01, total_edge_length - self.max_edge_length)
#         left_edge_max = total_edge_length - left_edge_min
#         return True, (total_edge_length, left_edge_min, left_edge_max)
#
#
# class EdgeEnvContinuous2(object):
#
#     def __init__(self, edge_cont_cfg):
#         self.max_total_length = edge_cont_cfg.MAX_TOTAL_LENGTH
#
#     def actions_2_edges(self, edge_action):
#         total_length_action, left_portion_action = edge_action
#
#         total_length = total_length_action * self.max_total_length
#         left_length = total_length * left_portion_action
#
#         right_length = total_length - left_length
#         return left_length, right_length
#
#     def edges_2_actions(self, left_length, right_length):
#         total_length = left_length + right_length
#         left_portion = left_length / total_length
#         total_length_action = min(max(0.001, total_length / self.max_total_length), 0.999)
#         left_portion_action = min(max(0.001, left_portion), 0.999)
#         return total_length_action, left_portion_action
#
#     def generate_random_actions(self):
#         total_portion, left_portion = random.uniform(0.001, 0.999), random.uniform(0.001, 0.999)
#         return total_portion, left_portion
#
#
# class EdgeEnvCategoricalIndependent(object):
#
#     def __init__(self, edge_cat_cfg):
#         self.categorical_bin_size = edge_cat_cfg.BIN_SIZE
#         self.categorical_bins = edge_cat_cfg.BINS
#         self.max_edge_length = float(self.categorical_bins * self.categorical_bin_size)
#
#     def actions_2_edges(self, edge_action):
#         left_length_action, right_length_action = edge_action
#         left_length = (left_length_action + 1) * self.categorical_bin_size
#         right_length = (right_length_action + 1) * self.categorical_bin_size
#         return left_length, right_length
#
#     def edges_2_actions(self, left_length, right_length):
#         left_length_action = int(round((left_length / self.categorical_bin_size))) - 1
#         right_length_action = int(round((right_length / self.categorical_bin_size))) - 1
#         return left_length_action, right_length_action
#
#     def generate_random_actions(self):
#         left_length_action = random.randint(0, self.categorical_bins - 1)
#         right_length_action = random.randint(0, self.categorical_bins - 1)
#         return left_length_action, right_length_action
#
#     def get_possible_splits(self, total_edge_length):
#         """
#         Given total edge length, find all possible splits
#         :param total_edge_length:
#         :return:
#         """
#         if total_edge_length / self.categorical_bin_size < 2:
#             return False, None
#
#         max_edge_length = self.categorical_bin_size * self.categorical_bins
#         min_available_cuts = max(1, round((total_edge_length - max_edge_length) / self.categorical_bin_size))
#         max_available_cuts = min(round(total_edge_length / self.categorical_bin_size) - 1, self.categorical_bins)
#         available_left_length = np.arange(min_available_cuts, max_available_cuts + 1) * self.categorical_bin_size
#         available_right_length = total_edge_length - available_left_length
#         possible_edge_pairs = list(zip(available_left_length, available_right_length))
#         return True, possible_edge_pairs
#
#
# class EdgeEnvCategorical2(object):
#
#     def __init__(self, edge_cat_cfg):
#         self.total_length_bin_size = edge_cat_cfg.TOTAL_LENGTH_BIN_SIZE
#         self.left_portion_bin_size = edge_cat_cfg.LEFT_PORTION_BIN_SIZE
#         self.total_length_bins = edge_cat_cfg.TOTAL_LENGTH_BINS
#         self.left_portion_bins = edge_cat_cfg.LEFT_PORTION_BINS
#
#     def actions_2_edges(self, edge_action):
#         total_length_action, left_portion_action = edge_action
#         total_length = self.total_length_bin_size * (1 + total_length_action)
#         left_portion = (1 + left_portion_action) * self.left_portion_bin_size
#         left_length = total_length * left_portion
#         right_length = total_length * (1 - left_portion)
#         return left_length, right_length
#
#     def edges_2_actions(self, left_length, right_length):
#         total_length = left_length + right_length
#         total_length_action = int(round(total_length / self.total_length_bin_size)) - 1
#         left_length_portion = left_length / total_length
#         left_portion_action = int(round(left_length_portion / self.left_portion_bin_size) - 1)
#         action = total_length_action, left_portion_action
#         return action
#
#     def generate_random_actions(self):
#         total_length_action = random.randint(0, self.total_length_bins - 1)
#         left_portion_action = random.randint(0, self.left_portion_bins - 1)
#         return total_length_action, left_portion_action
#
#
# class EdgeEnvCategorical2Independent(object):
#
#     def __init__(self, edge_cat_cfg):
#         self.total_length_bin_size = edge_cat_cfg.TOTAL_LENGTH_BIN_SIZE
#         self.left_portion_bin_size = edge_cat_cfg.LEFT_PORTION_BIN_SIZE
#         self.total_length_bins = edge_cat_cfg.TOTAL_LENGTH_BINS
#         self.left_portion_bins = edge_cat_cfg.LEFT_PORTION_BINS
#
#     def actions_2_edges(self, edge_action):
#         total_length_action, left_portion_action = edge_action
#         total_length = self.total_length_bin_size * (1 + total_length_action)
#         left_portion = (1 + left_portion_action) * self.left_portion_bin_size
#         left_length = total_length * left_portion
#         right_length = total_length * (1 - left_portion)
#         return left_length, right_length
#
#     def edges_2_actions(self, left_length, right_length):
#         total_length = left_length + right_length
#         total_length_action = int(round(total_length / self.total_length_bin_size)) - 1
#         left_length_portion = left_length / total_length
#         left_portion_action = int(round(left_length_portion / self.left_portion_bin_size) - 1)
#         action = total_length_action, left_portion_action
#         return action
#
#     def generate_random_actions(self):
#         total_length_action = random.randint(0, self.total_length_bins - 1)
#         left_portion_action = random.randint(0, self.left_portion_bins - 1)
#         return total_length_action, left_portion_action


def build_edge_env(cfg):
    edges_cfg = cfg.GFN.MODEL.EDGES_MODELING
    dist = edges_cfg.DISTRIBUTION
    assert dist in ['CATEGORICAL']
    edge_env = EdgeEnvCategorical(edges_cfg.CATEGORICAL)
    return edge_env
    # assert dist in ['CATEGORICAL', 'CONTINUOUS2', 'CATEGORICAL_INDEPENDENT']
    # if dist == 'CATEGORICAL':
    #     edge_env = EdgeEnvCategorical(edges_cfg.CATEGORICAL)
    # elif dist == 'CATEGORICAL_INDEPENDENT':
    #     edge_env = EdgeEnvCategoricalIndependent(edges_cfg.CATEGORICAL_INDEPENDENT)
    # else:
    #     edge_env = EdgeEnvContinuous2(edges_cfg.CONTINUOUS2)
    # return edge_env
