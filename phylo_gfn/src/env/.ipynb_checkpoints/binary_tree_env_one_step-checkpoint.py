import torch
import random
import numpy as np
from typing import List, Union
import torch.nn.functional as F
from src.env.trajectory import Trajectory
import networkx as nx
from copy import deepcopy

CHARACTERS_MAPS = {
    'DNA': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        'N': [1., 1., 1., 1.]
    },
    'RNA': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'N': [1., 1., 1., 1.]
    },
    'DNA_WITH_GAP': {
        'A': [1., 0., 0., 0., 0.],
        'C': [0., 1., 0., 0., 0.],
        'G': [0., 0., 1., 0., 0.],
        'T': [0., 0., 0., 1., 0.],
        '-': [0., 0., 0., 0., 1.],
        'N': [1., 1., 1., 1., 0.]
    },
    'RNA_WITH_GAP': {
        'A': [1., 0., 0., 0., 0.],
        'C': [0., 1., 0., 0., 0.],
        'G': [0., 0., 1., 0., 0.],
        'U': [0., 0., 0., 1., 0.],
        '-': [0., 0., 0., 0., 1.],
        'N': [1., 1., 1., 1., 0.]
    }
}


class PhyloTree(object):

    def __init__(self, left_tree=None, right_tree=None, root_seq_data=None):
        """
        BINARY TREE REPRESENTATION OF THE PHYLOGENETIC TREE
        """
        # assert (left_tree is not None and right_tree is not None) or (root_seq_data is not None)

        self.left_tree = left_tree
        self.right_tree = right_tree
        if root_seq_data is not None:
            root_seq, seq_indices = root_seq_data
            self.root_seq = root_seq
            self.seq_indices = seq_indices
            self.total_mutations = 0
        else:
            left_idx = left_tree.seq_indices[0]
            right_idx = right_tree.seq_indices[0]
            if left_idx > right_idx:
                self.left_tree = right_tree
                self.right_tree = left_tree
            self.root_seq, self.total_mutations, self.seq_indices = self.build_root_sequence()
        self.indices_str = str(self.seq_indices)

    @property
    def is_leaf(self):
        return self.left_tree is None and self.right_tree is None

    @property
    def is_internal(self):
        return self.left_tree is not None or self.right_tree is not None

    def __str__(self):
        if self.left_tree is None:
            return str(self.seq_indices) + '\n'

        l = self.left_tree
        r = self.right_tree
        left_str = str(l)
        right_str = str(r)
        left_str_parts = left_str.split('\n')[:-1]
        right_str_parts = right_str.split('\n')[:-1]

        data = [self.indices_str]
        data.append('├── ' + left_str_parts[0])
        for p in left_str_parts[1:]:
            data.append('│   ' + p)
        data.append('└── ' + right_str_parts[0])
        for p in right_str_parts[1:]:
            data.append('    ' + p)

        new_tree_str = '\n'.join(data) + '\n'
        return new_tree_str

    def build_root_sequence(self):
        """
        build root sequence based on left right subtrees
        """
        total_mutations = self.left_tree.total_mutations + self.right_tree.total_mutations
        left_seq = self.left_tree.root_seq
        right_seq = self.right_tree.root_seq

        seq_overlap = left_seq * right_seq
        seq_union = (left_seq + right_seq) > 0
        has_overlap = seq_overlap.sum(axis=1) > 0
        has_overlap = has_overlap.reshape(-1, 1)

        # if there is overlap, keep the overlap, otherwise keep the union
        seq = (1 - has_overlap) * seq_union + has_overlap * seq_overlap
        seq = seq.astype(left_seq.dtype)
        mutations = (1 - has_overlap).sum()
        total_mutations += mutations
        seq_indices = self.left_tree.seq_indices + self.right_tree.seq_indices
        seq_indices = sorted(seq_indices)
        return seq, total_mutations, seq_indices

    def breadth_first_traversal(self):
        # traversal includes the leaves
        traversed_list = []
        queue = [self]

        while len(queue) > 0:
            node = queue.pop(0)
            traversed_list.append(node)
            if node.left_tree is not None:
                queue.append(node.left_tree)
            if node.right_tree is not None:
                queue.append(node.right_tree)

        return traversed_list

    def postorder_traversal(self):
        # leaves not included in the traversal
        stack = [self]
        stack_rev_postorder = []

        while len(stack) > 0:
            node = stack.pop()
            if node.is_internal:
                # only keep internal nodes
                stack_rev_postorder.append(node)

            if node.left_tree is not None:
                stack.append(node.left_tree)
            if node.right_tree is not None:
                stack.append(node.right_tree)

        return stack_rev_postorder[::-1]

    def get_internal_node_leafsets(self, include_root=True):
        traversed_list = self.breadth_first_traversal()
        if not include_root:
            traversed_list = traversed_list[1:]
        internal_node_leafsets = []
        for node in traversed_list:
            seq_indices = node.seq_indices
            if len(seq_indices) > 1:
                internal_node_leafsets.append(seq_indices)
        return internal_node_leafsets

    @property
    def leafsets_signature(self):
        internal_node_leafsets = self.get_internal_node_leafsets(include_root=False)
        nb_species = len(self.seq_indices)
        set_all_species = frozenset(list(range(nb_species)))
        leafsets_signature = []
        for leafset in internal_node_leafsets:
            if len(leafset) > nb_species - 2:
                continue
            leafsets_signature.append(frozenset(leafset))
            leafset_complement = set_all_species.difference(leafset)
            leafsets_signature.append(leafset_complement)
        leafsets_signature = list(set(leafsets_signature))
        leafsets_signature_str = [str(sorted(list(leafset))) for leafset in leafsets_signature]
        leafsets_signature_str = sorted(leafsets_signature_str)
        leafsets_signature_str = '|'.join(leafsets_signature_str)
        return leafsets_signature_str

    def to_unrooted_tree(self):
        traversed_list = self.postorder_traversal()  # this list only contains internal nodes
        nb_internal_nodes = len(traversed_list)

        nx_graph = nx.Graph()
        for idx, node in enumerate(traversed_list):

            # add leaves to the graph
            if node.left_tree.is_leaf:
                nx_graph.add_node(node.left_tree.indices_str,
                                  seq_indices=node.left_tree.seq_indices, root_seq=node.left_tree.root_seq)
            if node.right_tree.is_leaf:
                nx_graph.add_node(node.right_tree.indices_str,
                                  seq_indices=node.right_tree.seq_indices, root_seq=node.right_tree.root_seq)

            if idx == nb_internal_nodes - 1:
                # the root node
                nx_graph.add_edge(node.left_tree.indices_str, node.right_tree.indices_str)
            else:
                # add the internal node
                nx_graph.add_node(node.indices_str, seq_indices=None, root_seq=None)
                nx_graph.add_edge(node.left_tree.indices_str, node.indices_str)
                nx_graph.add_edge(node.right_tree.indices_str, node.indices_str)

        return UnrootedPhyloTree(nx_graph, self.leafsets_signature, self.total_mutations)


class UnrootedPhyloTree(object):

    def __init__(self, nx_graph, leafsets_signature=None, total_mutations=None):
        # the last two nodes are the children of the original root node
        self.nx_graph = nx_graph
        self.leafsets_signature = leafsets_signature
        self.total_mutations = total_mutations

    def nearest_neighbour_interchange(self, edge_idx=None):
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_internal_edges = nb_nodes // 2 - 2
        if edge_idx is None:
            edge_idx = random.choice(range(nb_internal_edges))
        else:
            assert edge_idx < nb_internal_edges, \
                f'edge_idx exceeds {nb_internal_edges} (max internal edges) in the current graph'

        nx_graph = deepcopy(self.nx_graph)
        node_1_id, node_2_id = list(filter(lambda x: nx_graph.degree(x[0]) == 3 and nx_graph.degree(x[1]) == 3,
                                           nx_graph.edges))[edge_idx]
        node_1_nei, node_2_nei = list(nx_graph.neighbors(node_1_id)), list(nx_graph.neighbors(node_2_id))
        node_1_nei.remove(node_2_id)
        node_2_nei.remove(node_1_id)
        sampled_nei = random.choice(node_1_nei)
        nx_graph.remove_edge(sampled_nei, node_1_id)
        nx_graph.remove_edge(node_2_nei[0], node_2_id)
        nx_graph.add_edge(sampled_nei, node_2_id)
        nx_graph.add_edge(node_2_nei[0], node_1_id)

        return UnrootedPhyloTree(nx_graph)

    def subtree_pruning_regrafting(self, edge_idx_for_pruning=None, edge_idx_for_regrafting=None,
                                   allow_idx_adjustment=True):
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_edges = nb_nodes - 1
        if edge_idx_for_pruning is None or edge_idx_for_regrafting is None:
            edge_idx_for_pruning, edge_idx_for_regrafting = random.sample(range(nb_edges), 2)
        else:
            assert edge_idx_for_pruning < nb_edges and edge_idx_for_regrafting < nb_edges, \
                f'edge_idx exceeds {nb_edges} (all edges) in the current graph'

        nx_graph = deepcopy(self.nx_graph)
        all_edges = list(nx_graph.edges)
        node_id_to_prune = all_edges[edge_idx_for_pruning]
        node_id_to_regraft = all_edges[edge_idx_for_regrafting]

        # in case of overlap between pruning edge and regrafting edge
        while node_id_to_prune[0] in node_id_to_regraft or node_id_to_prune[1] in node_id_to_regraft:
            if allow_idx_adjustment:
                edge_idx_for_regrafting = (edge_idx_for_regrafting + 1) % nb_edges
                node_id_to_regraft = all_edges[edge_idx_for_regrafting]
            else:
                return self

        # determine which is the node that can move on the graph
        nx_graph.remove_edge(*node_id_to_prune)
        if nx.has_path(nx_graph, node_id_to_prune[0], node_id_to_regraft[0]):
            node_to_reattach = node_id_to_prune[0]
            another_node = node_id_to_prune[1]
        else:
            node_to_reattach = node_id_to_prune[1]
            another_node = node_id_to_prune[0]
        nx_graph.add_edge(*node_id_to_prune)

        # remove the 'reattach' from the graph, and reconnect its two other neighbours
        node_to_detach = []
        for nei_node_id in nx_graph.neighbors(node_to_reattach):
            if nei_node_id != another_node:
                node_to_detach.append(nei_node_id)
        nx_graph.remove_edge(node_to_reattach, node_to_detach[0])
        nx_graph.remove_edge(node_to_reattach, node_to_detach[1])
        nx_graph.add_edge(*node_to_detach)

        # regrafting step
        nx_graph.remove_edge(*node_id_to_regraft)
        nx_graph.add_edge(node_id_to_regraft[0], node_to_reattach)
        nx_graph.add_edge(node_id_to_regraft[1], node_to_reattach)

        return UnrootedPhyloTree(nx_graph)

    def bisection_reconnection(self, edge_idx_1=None, edge_idx_2=None, edge_idx_3=None, allow_idx_adjustment=True):
        # edge in the middle of the three will be used for bisection, the other two for reconnection
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_edges = nb_nodes - 1
        nx_graph = deepcopy(self.nx_graph)
        all_edges, all_internal_edges_idx = [], []
        for i, edge in enumerate(nx_graph.edges):
            all_edges.append(edge)
            if nx_graph.degree(edge[0]) == 3 and nx_graph.degree(edge[1]) == 3:
                all_internal_edges_idx.append(i)

        if edge_idx_1 is None or edge_idx_2 is None or edge_idx_3 is None:
            # edge_idx_1 will be the bisection edge
            edge_idx_1 = random.choice(all_internal_edges_idx)
            # select one reconnection edge
            list_edge_idx_reconnection = list(range(nb_edges))
            list_edge_idx_reconnection.remove(edge_idx_1)
            edge_idx_2 = random.choice(list_edge_idx_reconnection)
            list_edge_idx_reconnection.remove(edge_idx_2)
            # select another reconnection edge
            random.shuffle(list_edge_idx_reconnection)
            nx_graph.remove_edge(*all_edges[edge_idx_1])
            for edge_idx in list_edge_idx_reconnection:
                if not nx.has_path(nx_graph, all_edges[edge_idx][0], all_edges[edge_idx_2][0]):
                    break
            edge_idx_3 = edge_idx
            nx_graph.add_edge(*all_edges[edge_idx_1])
        else:
            assert edge_idx_1 < nb_edges and edge_idx_2 < nb_edges and edge_idx_3 < nb_edges, \
                f'edge_idx exceeds {nb_edges} (all edges) in the current graph'

        list_edge_idx = [edge_idx_1, edge_idx_2, edge_idx_3]

        found_middle_edge = False
        for i, edge_idx in enumerate(list_edge_idx):
            node_id = all_edges[edge_idx]
            nx_graph.remove_edge(*node_id)
            other_node_id_1 = all_edges[list_edge_idx[(i + 1) % 3]]
            other_node_id_2 = all_edges[list_edge_idx[(i + 2) % 3]]
            if not nx.has_path(nx_graph, other_node_id_1[0], other_node_id_2[0]):
                # means we've found the bisection edge
                if nx.has_path(nx_graph, node_id[0], other_node_id_2[0]):
                    tmp = other_node_id_1
                    other_node_id_1 = other_node_id_2
                    other_node_id_2 = tmp
                found_middle_edge = True
                nx_graph.add_edge(*node_id)
                break
            nx_graph.add_edge(*node_id)

        if found_middle_edge is False:
            # this is possible, e.g. if none of them are internal edges
            return self
        node_bisection_id = node_id
        node_reconnection_id_1 = other_node_id_1
        node_reconnection_id_2 = other_node_id_2

        if node_bisection_id[0] not in node_reconnection_id_1:
            node_to_detach = []
            for nei_node_id in nx_graph.neighbors(node_bisection_id[0]):
                if nei_node_id != node_bisection_id[1]:
                    node_to_detach.append(nei_node_id)
            nx_graph.remove_edge(node_bisection_id[0], node_to_detach[0])
            nx_graph.remove_edge(node_bisection_id[0], node_to_detach[1])
            nx_graph.add_edge(*node_to_detach)

            nx_graph.remove_edge(*node_reconnection_id_1)
            nx_graph.add_edge(node_bisection_id[0], node_reconnection_id_1[0])
            nx_graph.add_edge(node_bisection_id[0], node_reconnection_id_1[1])

        if node_bisection_id[1] not in node_reconnection_id_2:
            node_to_detach = []
            for nei_node_id in nx_graph.neighbors(node_bisection_id[1]):
                if nei_node_id != node_bisection_id[0]:
                    node_to_detach.append(nei_node_id)
            nx_graph.remove_edge(node_bisection_id[1], node_to_detach[0])
            nx_graph.remove_edge(node_bisection_id[1], node_to_detach[1])
            nx_graph.add_edge(*node_to_detach)

            nx_graph.remove_edge(*node_reconnection_id_2)
            nx_graph.add_edge(node_bisection_id[1], node_reconnection_id_2[0])
            nx_graph.add_edge(node_bisection_id[1], node_reconnection_id_2[1])

        return UnrootedPhyloTree(nx_graph)

    def to_rooted_tree(self, edge_idx):
        """
        edge idx to insert the root node, transforming the tree into a rooted tree
        can be easily modified to insert nodes between two node indices.
        
        """
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_leaves = nb_nodes // 2 + 1
        nb_edges = nb_nodes - 1
        assert edge_idx < nb_edges, f'edge_idx exceeds {nb_edges} in the current graph'

        nx_graph = deepcopy(self.nx_graph)
        node_1_id, node_2_id = list(nx_graph.edges)[edge_idx]

        root_seq_indices = list(range(nb_leaves))
        root_id = str(root_seq_indices)
        nx_graph.add_node(root_id, seq_indices=root_seq_indices, root_seq=None)
        nx_graph.remove_edge(node_1_id, node_2_id)
        nx_graph.add_edge(root_id, node_1_id)
        nx_graph.add_edge(root_id, node_2_id)

        for node_id in nx.dfs_postorder_nodes(nx_graph, source=root_id):
            seq_indices, root_seq = nx_graph.nodes[node_id]['seq_indices'], nx_graph.nodes[node_id]['root_seq']
            if root_seq is not None:
                node = PhyloTree(root_seq_data=[root_seq, seq_indices])
            else:
                child_nodes = [nx_graph.nodes[nei_id]['rooted_tree_node'] for nei_id in nx.neighbors(nx_graph, node_id)
                               if 'rooted_tree_node' in nx_graph.nodes[nei_id]]
                node = PhyloTree(left_tree=child_nodes[0], right_tree=child_nodes[1])
            nx_graph.nodes[node_id]['rooted_tree_node'] = node

        # postorder traversal returns the root at the end
        return node

    def __lt__(self, obj):
        return self.total_mutations > obj.total_mutations

    def __eq__(self, obj):
        return self.total_mutations == obj.total_mutations


class PhylogenticTreeState(object):

    def __init__(self, subtrees: List[PhyloTree]):
        self.subtrees = subtrees
        self.is_done = (len(self.subtrees) == 1)
        self.num_trees = len(self.subtrees)
        trees = [x for x in self.subtrees if x.left_tree is not None]
        self.is_initial = len(trees) == 0

    def order_subtrees(self):
        # we need this to compare if two states are identical — helps to accelerate beam search
        # return the signatures comprising of the candidate trees and the ordered subtrees
        # as well as a dictionary to convert the original batch action to the new index under the ordered subtrees
        str_rep = ''
        sorted_idx = np.argsort([x.seq_indices[0] for x in self.subtrees])
        subtrees_ordered = []
        conversion_dict = {}
        for i, idx in enumerate(sorted_idx):
            tree = self.subtrees[idx]
            subtrees_ordered.append(tree)
            conversion_dict[idx] = i
            str_rep += '-' + str(tree.seq_indices)
        self.subtrees = subtrees_ordered

        return str_rep, conversion_dict

    def display(self):

        tree_reps = [str(x) for x in self.subtrees]
        tree_reps_splitted = [x.split('\n') for x in tree_reps]
        num_lines = max([len(x) for x in tree_reps_splitted])
        reps = [''] * num_lines

        for splitted_rep in tree_reps_splitted:
            max_width = max([len(x) for x in splitted_rep])
            for idx in range(num_lines):
                if idx < len(splitted_rep):
                    line = splitted_rep[idx]
                else:
                    line = ''
                reps[idx] = reps[idx] + line + ' ' * (max_width - len(line))
                reps[idx] = reps[idx] + '\t'

        reps = '\n'.join(reps)
        print(reps)


class PhylogenticTreeState2Input(object):

    def __init__(self, cfg, max_num_seqs):
        self.mode = cfg.GFN.MODEL.SEQ_LEN

        self.pairwise_masking_data = {}
        for n in range(2, max_num_seqs + 1):
            self.pairwise_masking_data[n] = self.build_pairs_masks_mapping(n)

    def state2input(self, state: PhylogenticTreeState):
        """
        handles one state
        """
        seqs = np.array([x.root_seq for x in state.subtrees], dtype=np.bool_)
        seqs = seqs.reshape(len(seqs), -1)
        seqs = torch.tensor(seqs)
        return seqs

    def states2inputs(self, states: List[PhylogenticTreeState]):
        """
        handles multiple states, each  with different number of subtrees or in different action modes
        """
        batch_input = []

        for state in states:
            input_state = self.state2input(state).unsqueeze(0)  # 1, nb_seq, char_num * seq_len
            batch_input.append(input_state)
        batch_nb_seq = np.array([input_state.shape[1] for input_state in batch_input])
        max_nb_seq = max(batch_nb_seq)

        # action_mapping_tensor: idx with triu padding to idx without triu padding
        # actions_mapping_reverse_tensor: idx without triu padding to idx with triu padding
        mask_tensor, action_mapping_tensor, actions_mapping_reverse_tensor = self.pairwise_masking_data[max_nb_seq]
        mask_tensor = torch.tensor(mask_tensor[batch_nb_seq - 2])
        action_mapping_tensor = torch.tensor(action_mapping_tensor[batch_nb_seq - 2])
        actions_mapping_reverse_tensor = torch.tensor(actions_mapping_reverse_tensor[batch_nb_seq - 2])

        for i, input_state in enumerate(batch_input):
            batch_input[i] = F.pad(input_state, (0, 0, 0, max_nb_seq - batch_nb_seq[i]), "constant", False)
        batch_input = torch.vstack(batch_input)
        batch_nb_seq = torch.tensor(batch_nb_seq).long()

        return {
            'batch_input': batch_input,
            'batch_nb_seq': batch_nb_seq,
            'pairwise_mask_tensor': mask_tensor,
            'pairwise_action_tensor': action_mapping_tensor,
            'pairwise_action_reverse_tensor': actions_mapping_reverse_tensor
        }

    def build_pairs_masks_mapping(self, max_num_seqs):
        rows, cols = np.triu_indices(max_num_seqs, k=1)
        data = []
        for index, (i, j) in enumerate(zip(rows, cols)):
            data.append([index, i, j])

        pairwise_valid_tensor = []
        for i in range(2, max_num_seqs + 1):  # "i" represents the "effective_num_seqs"
            value = i - 1
            pairwise_valid_tensor.append([x[1] <= value and x[2] <= value for x in data])
        pairwise_valid_tensor = np.array(pairwise_valid_tensor)
        # pairwise_valid_tensor shape: effective_num_seqs x triu_size

        x_list, y_list = np.where(pairwise_valid_tensor)
        actions_mapping_tensor = np.ones_like(pairwise_valid_tensor, dtype=np.int64) * -1
        actions_mapping_reverse_tensor = np.ones_like(pairwise_valid_tensor, dtype=np.int64) * -1
        counter = 0
        current_x = 0
        for x, y in zip(x_list, y_list):

            if x != current_x:
                current_x = x
                counter = 0

            actions_mapping_tensor[x, y] = counter
            actions_mapping_reverse_tensor[current_x, counter] = y
            counter += 1

        masks_tensor = ~ pairwise_valid_tensor
        return masks_tensor, actions_mapping_tensor, actions_mapping_reverse_tensor


class PhyloTreeReward(object):

    def __init__(self, reward_cfg):
        self.C = reward_cfg.C
        assert reward_cfg.RESHAPE_METHOD in ['C-MUTATIONS', 'EXPONENTIAL']
        self.reshape_method = reward_cfg.RESHAPE_METHOD
        self.power = reward_cfg.POWER
        self.scale = reward_cfg.SCALE
        self.reward_exp_min = reward_cfg.EXP_MIN
        self.reward_exp_max = reward_cfg.EXP_MAX

    def exponential_reward(self, state: PhylogenticTreeState):
        """
        r = np.exp((C - total_mutations) / scale)
        NOTE: we changed the notation to make it similar to the earlier reward definition
        :param state:
        :return:
        """
        total_mutations = sum(x.total_mutations for x in state.subtrees)
        log_reward = (self.C - total_mutations) / self.scale
        if log_reward > 709:
            # to avoid overflow warning message
            reward = self.reward_exp_max
        elif log_reward < -745:
            # to avoid overflow warning message
            reward = self.reward_exp_min
        else:
            reward = np.exp(log_reward)
        return {
            'reward': reward,
            'log_reward': log_reward
        }

    def c_minus_mutation_reward(self, state: PhylogenticTreeState):
        """
        r = ((c - total_mutations) / scale) ** power if c > total_mutations
        r = np.exp(total_mutations - 1) ** power if c <= total_mutations
        :param state:
        :return:
        """
        total_mutations = state.subtrees[0].total_mutations
        if self.C > total_mutations:
            score = (self.C - total_mutations + 0.0) / self.scale
        else:
            score = np.exp(self.C - total_mutations - 1)
        score = score ** self.power
        return {
            'reward': score,
            'log_reward': np.log(score)
        }

    def __call__(self, state: PhylogenticTreeState):
        """
        compute reward
        """
        if not state.is_done:
            return None
        if self.reshape_method == 'C-MUTATIONS':
            return self.c_minus_mutation_reward(state)
        else:
            return self.exponential_reward(state)

    def compute_fl_partial_reward(self, state: PhylogenticTreeState):
        """
        compute partial state reward for FL GFN,
        NOTE the C IS NOT INCLUDED HERE !!!!!!
        :param state:
        :return:
        """
        total_mutations = sum(x.total_mutations for x in state.subtrees)
        log_reward = -total_mutations / self.scale

        return log_reward


class PhylogeneticTreeEnv(object):

    def __init__(self, cfg, sequences):
        self.cfg = cfg
        self.sequences = sequences
        self.reward_fn = PhyloTreeReward(cfg.ENV.REWARD)
        self.chars_dict = CHARACTERS_MAPS[cfg.ENV.SEQUENCE_TYPE]
        self.seq_arrays = [self.seq2array(seq) for seq in self.sequences]
        self.type = cfg.ENV.ENVIRONMENT_TYPE
        self.compute_partial_reward = (cfg.GFN.LOSS_TYPE in ('FLDB', 'FLSUBTB'))
        # store for each number of trees, what are the possible combination of pairs
        self.tree_pairs_dict = {}
        for n in range(2, len(self.sequences) + 1):
            self.tree_pairs_dict[n] = np.stack(np.triu_indices(n, k=1), axis=1)
        # need to update this self.fl_c in unconditional model
        self.fl_c = cfg.ENV.REWARD.C / cfg.ENV.REWARD.SCALE / (len(self.sequences) - 1)
        self.parsimony_problem = True

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq, dtype=np.bool_)
        return data

    def sample(self, n):

        trajectories = []
        for _ in range(n):
            trajectories.append(self.generate_random_trajectory())
        return trajectories

    def generate_random_trajectory(self):

        state = self.get_initial_state()
        trajectory = Trajectory(state)
        while not state.is_done:
            n = len(state.subtrees)
            actions = list(range(int(n * (n - 1) / 2)))
            a = random.sample(actions, 1)[0]
            next_state = self.apply_action(state, a)
            r = self.reward_fn(next_state)
            trajectory.update(next_state, a, r, next_state.is_done)
            state = next_state

        return trajectory

    def get_initial_state(self):

        phylo_trees = [PhyloTree(None, None, [seq, [idx]]) for idx, seq in enumerate(self.seq_arrays)]
        state = PhylogenticTreeState(phylo_trees)
        if self.compute_partial_reward:
            state.log_partial_reward = self.reward_fn.compute_fl_partial_reward(state)
        return state

    def transition(self, state: PhylogenticTreeState, action):

        assert not state.is_done
        new_state = self.apply_action(state, action)
        done = new_state.is_done
        reward = self.reward_fn(new_state)

        return state, new_state, action, reward, done

    def determine_done(self, state: PhylogenticTreeState):

        return state.is_done

    def get_parent_states(self, state: PhylogenticTreeState):
        """
        Given state, return a list of possible parents
        """
        state_subtrees = state.subtrees

        possible_parents = []
        for idx, subtree in enumerate(state_subtrees):
            if subtree.left_tree is not None:
                trees = state_subtrees[:idx] + state_subtrees[idx + 1:] + [subtree.left_tree, subtree.right_tree]
                n = len(trees)
                action = int((n * (n - 1) / 2) - 1)
                possible_parents.append([PhylogenticTreeState(trees), action])

        return possible_parents

    def get_all_parents_trajectory(self, trajectory: Trajectory):
        """
        for each state in the trajectory, get a list of possible parents
        """
        transitions = trajectory.transitions
        all_possible_parents = []
        for t in transitions:
            state = t[1]
            all_possible_parents.append(self.get_parent_states(state))
        return all_possible_parents

    def get_number_parents_trajectory(self, trajectory: Trajectory):
        """
        simplified
        for each state in the trajectory, count the number of possible parents
        """
        transitions = trajectory.transitions
        all_number_parents = []
        for t in transitions:
            state = t[1]
            subtrees = state.subtrees
            all_number_parents.append(sum([tree.left_tree is not None for tree in subtrees]))
        return all_number_parents

    def state_to_trajectory(self, state: PhylogenticTreeState):
        """
        Given a terminal state, randomly select a possible trajectory
        """
        assert state.is_done, 'only supports terminal states'
        if self.compute_partial_reward:
            state.log_partial_reward = self.reward_fn.compute_fl_partial_reward(state)
        transitions = []
        while not state.is_initial:
            parents = self.get_parent_states(state)
            parent, a = random.sample(parents, 1)[0]
            transitions.append([parent, state, a, 0, False])
            state = parent
            if self.compute_partial_reward:
                state.log_partial_reward = self.reward_fn.compute_fl_partial_reward(state)

        transitions = transitions[::-1]
        transitions[-1][-1] = True
        trajectory = Trajectory(None)
        trajectory.actions = [x[2] for x in transitions]
        trajectory.transitions = transitions
        trajectory.current_state = transitions[-1][1]
        trajectory.done = True
        trajectory.reward = self.reward_fn(trajectory.current_state)
        return trajectory

    def state_to_all_trajectory(self, state: PhylogenticTreeState):
        """
        Given a terminal state, get all unique trajectories leading to it
        """
        assert state.is_done, 'only supports terminal states'
        all_transitions = [[[state, None, None, None, None]]]
        all_trajectories = []

        while len(all_transitions) > 0:
            transitions = all_transitions.pop(0)
            current_state = transitions[-1][0]
            if not current_state.is_initial:
                parents = self.get_parent_states(current_state)
                for parent, action in parents:
                    transitions_copy = transitions.copy()
                    transitions_copy.append([parent, current_state, action, 0, False])
                    all_transitions.append(transitions_copy)
            else:
                transitions = transitions[1:][::-1]
                transitions[-1][-1] = True
                trajectory = Trajectory(None)
                trajectory.actions = [x[2] for x in transitions]
                trajectory.transitions = transitions
                trajectory.current_state = transitions[-1][1]
                trajectory.done = True
                trajectory.reward = self.reward_fn(trajectory.current_state)
                all_trajectories.append(trajectory)

        return all_trajectories

    def trajectory_permute_leaves(self, trajectory: Trajectory):
        """
        for a terminal state, exchange the leaf species and arrive at a new terminal state which has the same
        (unlabelled) binary tree topology hence having the identical number of trajectories leading to it
        """
        all_action = []
        for state, next_state, a, reward, done in trajectory.transitions:
            all_action.append(a)

        initial_subtrees = [PhyloTree(None, None, [seq, [idx]]) for idx, seq in enumerate(self.seq_arrays)]
        random.shuffle(initial_subtrees)
        state = PhylogenticTreeState(initial_subtrees)
        trajectory = Trajectory(state)
        for a in all_action:
            next_state = self.apply_action(state, a)
            r = self.reward_fn(next_state)
            trajectory.update(next_state, a, r, next_state.is_done)
            state = next_state

        return trajectory

    def tree_to_trajectory(self, tree: PhyloTree):

        traj = self.state_to_trajectory(PhylogenticTreeState([tree]))
        return traj

    def apply_action(self, state, action):

        tree_pairs = self.tree_pairs_dict[state.num_trees]
        i, j = tree_pairs[action]
        new_trees = []
        for idx in range(len(state.subtrees)):
            if idx not in (i, j):
                new_trees.append(state.subtrees[idx])
        new_tree = PhyloTree(state.subtrees[i], state.subtrees[j])
        new_trees.append(new_tree)
        state = PhylogenticTreeState(new_trees)
        if self.compute_partial_reward:
            state.log_partial_reward = self.reward_fn.compute_fl_partial_reward(state)
        return state


def build_one_step_env(cfg, all_seqs):
    env = PhylogeneticTreeEnv(cfg, all_seqs)
    state2input = PhylogenticTreeState2Input(cfg, len(all_seqs))
    return env, state2input
