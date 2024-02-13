import torch
import random
import numpy as np
from typing import List, Union
import torch.nn.functional as F
from src.env.trajectory import Trajectory
import itertools
from src.utils.evolution_model import EvolutionModel
from src.env.edges_env import build_edge_env
from copy import deepcopy
import networkx as nx
from matplotlib import pyplot as plt

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
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'T': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.]
    },
    'RNA_WITH_GAP': {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        '-': [1., 1., 1., 1.],
        'N': [1., 1., 1., 1.]
    }
}


class PhyloTree(object):

    def __init__(self, evolution_model, at_root, left_tree_data=None, right_tree_data=None, root_seq_data=None):
        """
        BINARY TREE REPRESENTATION OF THE PHYLOGENETIC TREE
        """
        assert (left_tree_data is not None and right_tree_data is not None) or (root_seq_data is not None)

        self.left_tree_data = left_tree_data
        self.right_tree_data = right_tree_data
        self.evolution_model = evolution_model
        self.at_root = at_root
        if root_seq_data is not None:
            root_seq, seq_indices = root_seq_data
            self.root_seq = root_seq
            self.seq_indices = seq_indices
            self.total_mutations = 0
            self.seq_indices_str = str(self.seq_indices)
            self.root_rep_str = self.seq_indices_str
            self.log_score = 0
        else:
            left_idx = left_tree_data['tree'].seq_indices[0]
            right_idx = right_tree_data['tree'].seq_indices[0]
            if left_idx > right_idx:
                self.left_tree_data = right_tree_data
                self.right_tree_data = left_tree_data
            self.root_seq, self.log_score = self.build_felsenstein_representation(evolution_model)
            seq_indices = self.left_tree_data['tree'].seq_indices + self.right_tree_data['tree'].seq_indices
            self.seq_indices = sorted(seq_indices)
            self.seq_indices_str = str(self.seq_indices)
            self.root_rep_str = self.seq_indices_str + '_l_{:.3f}_r_{:.3f}'.format(
                float(self.left_tree_data['branch_length']),
                float(self.right_tree_data['branch_length'])
            )
        self.min_seq_index = min(self.seq_indices)
        self.tree_str = self.build_tree_str()

    @property
    def is_leaf(self):
        return self.left_tree_data is None and self.right_tree_data is None

    @property
    def is_internal(self):
        return self.left_tree_data is not None or self.right_tree_data is not None

    def __str__(self):
        try:
            return self.tree_str
        except:
            self.tree_str = self.build_tree_str()
            return self.tree_str

    def build_tree_str(self):
        if self.left_tree_data is None:
            return self.root_rep_str + '\n'

        l = self.left_tree_data['tree']
        r = self.right_tree_data['tree']
        left_str = str(l)
        right_str = str(r)
        left_str_parts = left_str.split('\n')[:-1]
        right_str_parts = right_str.split('\n')[:-1]

        data = [self.root_rep_str]
        data.append('├── ' + left_str_parts[0])
        for p in left_str_parts[1:]:
            data.append('│   ' + p)
        data.append('└── ' + right_str_parts[0])
        for p in right_str_parts[1:]:
            data.append('    ' + p)

        new_tree_str = '\n'.join(data) + '\n'
        return new_tree_str

    def build_felsenstein_representation(self, evolution_model):

        data = [
            [self.left_tree_data['tree'].root_seq, self.left_tree_data['branch_length']],
            [self.right_tree_data['tree'].root_seq, self.right_tree_data['branch_length']]
        ]
        p, log_score = evolution_model.compute_partial_prob(data, self.at_root)
        return p, log_score

    def __lt__(self, obj):
        return (obj.log_score - self.log_score) > 0.0001

    def __eq__(self, obj):
        return abs(self.log_score - obj.log_score) < 0.0001

    def postorder_traversal_internal(self):
        """
        perform post order traversal over the tree, return the internal nodes
        In principle, this function will not be used or called because binary rooted tree is not the final tree
        :return: list of internal nodes,  dictionary of edge lengths between the node and its direct parent
        """
        # have a dictionary recording pairwise distances
        tree_edge_lengths = {}

        # leaves not included in the traversal
        stack = [self]
        stack_rev_postorder = []
        while len(stack) > 0:
            node = stack.pop()
            node_str = node.seq_indices_str
            if node.is_internal:
                # only keep internal nodes
                stack_rev_postorder.append(node)

            if node.left_tree_data is not None:
                for t in [node.left_tree_data, node.right_tree_data]:
                    t_str = t['tree'].seq_indices_str
                    stack.append(t['tree'])
                    tree_edge_lengths[(t_str, node_str)] = t['branch_length']
                    tree_edge_lengths[(node_str, t_str)] = t['branch_length']

        traversed_list = stack_rev_postorder[::-1]
        return traversed_list, tree_edge_lengths

    def to_unrooted_tree(self):

        traversed_list, tree_edge_lengths = self.postorder_traversal_internal()  # this list only contains internal nodes
        nb_internal_nodes = len(traversed_list)

        nx_graph = nx.Graph()
        # ALL NODES HERE HAVE CHILDREN
        for idx, node in enumerate(traversed_list):

            left_tree = node.left_tree_data['tree']
            right_tree = node.right_tree_data['tree']

            # add leaves to the graph
            # removing root_seq attribute to avoid RAM accumulation
            if left_tree.is_leaf:
                nx_graph.add_node(left_tree.seq_indices_str, seq_indices=left_tree.seq_indices)

            if right_tree.is_leaf:
                nx_graph.add_node(right_tree.seq_indices_str, seq_indices=right_tree.seq_indices)

            if idx == nb_internal_nodes - 1:
                # for root node, do not add the root node in graph, directly add edge for its left right children
                nx_graph.add_edge(left_tree.seq_indices_str, right_tree.seq_indices_str)

                # add the distance between the left and right node to the edge length dict
                node_str = node.seq_indices_str
                l_str = left_tree.seq_indices_str
                r_str = right_tree.seq_indices_str
                total_edge_length = tree_edge_lengths[(node_str, l_str)] + tree_edge_lengths[(node_str, r_str)]
                tree_edge_lengths[(l_str, r_str)] = total_edge_length
                tree_edge_lengths[(r_str, l_str)] = total_edge_length
                # remove the old parent-child entry in the dict, because its no longer exist
                del tree_edge_lengths[(l_str, node_str)]
                del tree_edge_lengths[(node_str, l_str)]
                del tree_edge_lengths[(r_str, node_str)]
                del tree_edge_lengths[(node_str, r_str)]
            else:
                # add the internal node
                nx_graph.add_node(node.seq_indices_str, seq_indices=None)
                nx_graph.add_edge(left_tree.seq_indices_str, node.seq_indices_str)
                nx_graph.add_edge(right_tree.seq_indices_str, node.seq_indices_str)

        return UnrootedPhyloTree(nx_graph, tree_edge_lengths, self.log_score)


class UnrootedPhyloTree(object):

    def __init__(self, nx_graph, tree_edge_lengths, log_score):
        """

        :param nx_graph: nx graph of the current rooted tree removing the head root
        :param tree_edge_lengths: dictionary recording pairwise nodes distances
        :param log_score:  log likelihood / log posterior p depending on the config
        """
        # the last two nodes are the children of the original root node
        self.nx_graph = nx_graph
        self.tree_edge_lengths = tree_edge_lengths
        self.tree_edge_lengths_print = {k: round(tree_edge_lengths[k], 3) for k in tree_edge_lengths}
        self.log_score = log_score

        # to find a unique representation of the tree, since we have edge lengths here, we do represent the
        # tree by the set of splits because we have edge lengths here
        # for now what we do is following:
        # find the edge connect to sequence 0, take the middle point of that edge as the root, and build a rooted tree
        # the root tree rep string is the unrooted tree's representation
        self.leafsets_signature = self.build_tree_signature()

        # store tree edge lengths by edge idx
        self.tree_edge_lengths_by_edge_idx = {}
        for idx, edge in enumerate(self.nx_graph.edges):
            self.tree_edge_lengths_by_edge_idx[idx] = self.tree_edge_lengths[edge]

    def build_tree_signature(self):

        # get edge connect to node 0
        edge = list(self.nx_graph.edges("[0]"))[0]
        root_edge_length = self.tree_edge_lengths[edge]
        root_edge_left = root_edge_length * 0.5
        root_edge_right = root_edge_length * 0.5
        # edge idx to insert the root node, transforming the tree into a rooted tree
        # can be easily modified to insert nodes between two node indices
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_leaves = nb_nodes // 2 + 1

        # insert the node over the edge
        # need to copy the graph because we are building a rooted tree graph
        nx_graph = deepcopy(self.nx_graph)
        node_1_id, node_2_id = edge
        root_seq_indices = list(range(nb_leaves))
        root_id = str(root_seq_indices)
        nx_graph.add_node(root_id, seq_indices=None)
        nx_graph.remove_edge(node_1_id, node_2_id)
        nx_graph.add_edge(root_id, node_1_id)
        nx_graph.add_edge(root_id, node_2_id)
        for node_id in nx.dfs_postorder_nodes(nx_graph, source=root_id):

            current_node = nx_graph.nodes[node_id]
            seq_indices = current_node['seq_indices']

            if seq_indices is not None:
                tree_str = str(seq_indices) + '\n'
            else:
                neighbor_nodes = nx.neighbors(nx_graph, node_id)
                children_indices = [x for x in neighbor_nodes if 'tree_str' in nx_graph.nodes[x]]
                children_nodes = [(nei_id, nx_graph.nodes[nei_id]) for nei_id in children_indices]
                children_nodes = sorted(children_nodes, key=lambda x: x[1]['seq_indices'][0])
                seq_indices = sum([x[1]['seq_indices'] for x in children_nodes], [])
                seq_indices = sorted(seq_indices)

                left_key = (children_nodes[0][0], node_id)
                right_key = (children_nodes[1][0], node_id)
                if node_id != root_id:
                    left_length = self.tree_edge_lengths[left_key]
                    right_length = self.tree_edge_lengths[right_key]
                else:
                    left_length = root_edge_left
                    right_length = root_edge_right

                root_rep_str = str(seq_indices) + '_l_{:.3f}_r_{:.3f}'.format(
                    float(left_length),
                    float(right_length)
                )

                left_str = children_nodes[0][1]['tree_str']
                right_str = children_nodes[1][1]['tree_str']
                left_str_parts = left_str.split('\n')[:-1]
                right_str_parts = right_str.split('\n')[:-1]
                data = [root_rep_str]
                data.append('├── ' + left_str_parts[0])
                for p in left_str_parts[1:]:
                    data.append('│   ' + p)
                data.append('└── ' + right_str_parts[0])
                for p in right_str_parts[1:]:
                    data.append('    ' + p)
                tree_str = '\n'.join(data) + '\n'

            nx_graph.nodes[node_id]['tree_str'] = tree_str
            nx_graph.nodes[node_id]['seq_indices'] = seq_indices
        return nx_graph.nodes[root_id]['tree_str']

    def to_rooted_tree(self, edge_idx, new_edge_left, new_edge_right, env, max_edge_length=None,
                       rotate_seq_pos=False, perturbation_size=None):
        # edge idx to insert the root node, transforming the tree into a rooted tree
        # can be easily modified to insert nodes between two node indices
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_edges = nb_nodes - 1
        assert edge_idx < nb_edges, f'edge_idx exceeds {nb_edges} in the current graph'
        edge = list(self.nx_graph.edges)[edge_idx]
        tree = self.to_rooted_tree_by_edge(edge, new_edge_left, new_edge_right, env, max_edge_length,
                                           rotate_seq_pos=rotate_seq_pos, perturbation_size=perturbation_size)
        return tree

    def to_rooted_tree_by_edge(self, edge, root_edge_left, root_edge_right, env,
                               max_edge_length=None, rotate_seq_pos=False, perturbation_size=None):
        """

        :param edge: edge of the unrooted tree to be split into a rooted tree
        :param root_edge_left: left root edge length
        :param root_edge_right: right root edge length
        :param max_edge_length: max (left+ right) length allowed, if actual length longer, reduce it by proportion
        :return:
        """
        # if root level left edge length is not provided, set it to be 1/2 of total edge length
        if root_edge_left is None:
            edge_length = self.tree_edge_lengths[edge]
            root_edge_left = edge_length * 0.5
            root_edge_right = edge_length * 0.5

        # edge idx to insert the root node, transforming the tree into a rooted tree
        # can be easily modified to insert nodes between two node indices
        nb_nodes = len(self.nx_graph.nodes)  # leafs and internal nodes
        nb_leaves = nb_nodes // 2 + 1

        # insert the node over the edge
        # need to copy the graph because we are building a rooted tree graph
        nx_graph = deepcopy(self.nx_graph)
        node_1_id, node_2_id = edge
        root_id = str(list(range(nb_leaves)))
        nx_graph.add_node(root_id, seq_indices=None)  # root node is also an internal node
        nx_graph.remove_edge(node_1_id, node_2_id)
        nx_graph.add_edge(root_id, node_1_id)
        nx_graph.add_edge(root_id, node_2_id)
        if rotate_seq_pos:
            order = np.random.permutation(env.seq_pos_indices)
        # build the new rooted tree bottom up
        for node_id in nx.dfs_postorder_nodes(nx_graph, source=root_id):

            current_node = nx_graph.nodes[node_id]
            if 'seq_indices' in current_node:
                seq_indices = current_node['seq_indices']
            elif 'seq_index' in current_node:  # ufboost tree compatibility
                seq_indices = [current_node['seq_index']]
            else:  # ufboost tree compatibility
                seq_indices = None

            if seq_indices is not None:
                root_seq = env.seq_arrays[seq_indices[0]]
                if rotate_seq_pos:
                    root_seq = root_seq[order]
                # if the node is a leaf, build its phylo tree directly
                tree = PhyloTree(env.evolution_model, at_root=False, root_seq_data=[root_seq, seq_indices])
            else:
                # otherwise we are at an internal nodes, by the nature of posorder traversal, all its children
                # are already visited, get both children
                neighbor_nodes = nx.neighbors(nx_graph, node_id)
                children_indices = [x for x in neighbor_nodes if 'rooted_tree' in nx_graph.nodes[x]]
                child_nodes = [(nei_id, nx_graph.nodes[nei_id]['rooted_tree']) for nei_id in children_indices]
                # sorting is for the leafsets_signature
                child_nodes = sorted(child_nodes, key=lambda x: x[1].min_seq_index)

                # get parent-child edge length
                left_key = (child_nodes[0][0], node_id)
                right_key = (child_nodes[1][0], node_id)
                if node_id != root_id:
                    left_length = self.tree_edge_lengths[left_key]
                    right_length = self.tree_edge_lengths[right_key]
                else:
                    left_length = root_edge_left
                    right_length = root_edge_right

                # probably won't be used
                if max_edge_length is not None:
                    length = left_length + right_length
                    if length > max_edge_length:
                        left_length = max_edge_length * left_length / length
                        right_length = max_edge_length * right_length / length

                if perturbation_size is not None:
                    if node_id != root_id:
                        delta = np.random.uniform(-1 * perturbation_size, perturbation_size)
                        left_length = left_length + delta
                        delta = np.random.uniform(-1 * perturbation_size, perturbation_size)
                        right_length = right_length + delta
                    else:
                        delta = np.random.uniform(-1 * perturbation_size, perturbation_size)
                        left_length += delta / 2
                        right_length += delta / 2

                tree = PhyloTree(env.evolution_model,
                                 at_root=(node_id == root_id),
                                 left_tree_data={'tree': child_nodes[0][1], 'branch_length': left_length},
                                 right_tree_data={'tree': child_nodes[1][1], 'branch_length': right_length})

            # store the built tree at nx graph
            nx_graph.nodes[node_id]['rooted_tree'] = tree

        return nx_graph.nodes[root_id]['rooted_tree']

    def __lt__(self, obj):
        return (obj.log_score - self.log_score) > 0.0001

    def __eq__(self, obj):
        return abs(self.log_score - obj.log_score) < 0.0001

    def display(self):

        pos = nx.spring_layout(self.nx_graph)
        nx.draw(self.nx_graph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(
            self.nx_graph, pos,
            edge_labels=self.tree_edge_lengths_print
        )
        plt.show()


class PhylogeneticTreeState(object):

    def __init__(self, subtrees: List[Union[PhyloTree, UnrootedPhyloTree]]):

        self.subtrees = subtrees
        self.is_done = (len(self.subtrees) == 1)
        self.num_trees = len(self.subtrees)

        if type(subtrees[0]) == PhyloTree:
            trees = [x for x in self.subtrees if x.left_tree_data is not None]
            self.is_initial = len(trees) == 0
            self.last_state = False
        else:
            self.is_initial = False
            self.last_state = True

    def display(self):
        if self.last_state:
            self.subtrees[0].display()
        else:
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


class PhylogeneticTreeState2Input(object):

    def __init__(self, cfg, max_num_seqs):
        self.seq_length = cfg.GFN.MODEL.SEQ_LEN
        self.normalize = cfg.GFN.NORMALIZE_LIKELIHOOD
        self.likelihood_float16 = cfg.GFN.LIKELIHOOD_FLOAT16
        self.pairwise_masking_data = {}
        for n in range(2, max_num_seqs + 1):
            self.pairwise_masking_data[n] = self.build_pairs_masks_mapping(n)

    def state2input(self, state: PhylogeneticTreeState):
        """
        handles one state
        """
        seqs = np.array([x.root_seq for x in state.subtrees])
        if self.normalize:
            seqs = seqs / seqs.sum(-1, keepdims=True)
        dtype = np.float16 if self.likelihood_float16 else np.float32
        seqs = seqs.astype(dtype)
        seqs = seqs.reshape(len(seqs), -1)
        seqs = torch.tensor(seqs)
        return seqs

    def states2inputs(self, states: List[PhylogeneticTreeState]):
        """
        handles multiple states, each  with different number of subtrees or in different action modes
        """
        batch_input = []

        for state in states:
            input_state = self.state2input(state).unsqueeze(0)  # 1, nb_seq, char_num * seq_len
            batch_input.append(input_state)
        batch_nb_seq = np.array([input_state.shape[1] for input_state in batch_input])
        max_nb_seq = max(batch_nb_seq)

        mask_tensor, action_mapping_tensor, actions_mapping_reverse_tensor = self.pairwise_masking_data[max_nb_seq]
        mask_tensor = torch.tensor(mask_tensor[batch_nb_seq - 2])
        action_mapping_tensor = torch.tensor(action_mapping_tensor[batch_nb_seq - 2])
        actions_mapping_reverse_tensor = torch.tensor(actions_mapping_reverse_tensor[batch_nb_seq - 2])
        for i, input_state in enumerate(batch_input):
            batch_input[i] = F.pad(input_state, (0, 0, 0, max_nb_seq - batch_nb_seq[i]), "constant", 0.0)
        batch_input = torch.vstack(batch_input)
        batch_nb_seq = torch.tensor(batch_nb_seq).long()

        return {
            'batch_input': batch_input,
            'batch_nb_seq': batch_nb_seq,
            'pairwise_mask_tensor': mask_tensor,
            'pairwise_action_tensor': action_mapping_tensor,
            'pairwise_action_reverse_tensor': actions_mapping_reverse_tensor,
            'return_tree_reps': True
        }

    def build_pairs_masks_mapping(self, max_num_seqs):
        rows, cols = np.triu_indices(max_num_seqs, k=1)
        data = []
        for index, (i, j) in enumerate(zip(rows, cols)):
            data.append([index, i, j])

        pairwise_valid_tensor = []
        for i in range(2, max_num_seqs + 1):
            value = i - 1
            pairwise_valid_tensor.append([x[1] <= value and x[2] <= value for x in data])
        pairwise_valid_tensor = np.array(pairwise_valid_tensor)

        x_list, y_list = np.where(pairwise_valid_tensor)
        actions_mapping_tensor = np.ones_like(pairwise_valid_tensor, dtype=np.int64) * -1
        counter = 0
        current_x = 0
        actions_mapping_reverse_tensor = np.ones((len(pairwise_valid_tensor), len(rows)), dtype=np.int64) * -1

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

    def exponential_reward(self, state: PhylogeneticTreeState):
        """
        r = np.exp((C - total_mutations) / scale)
        NOTE: we changed the notation to make it similar to the earlier reward definition
        :param state:
        :return:
        """
        log_reward = (self.C + state.subtrees[0].log_score) / self.scale
        reward = np.exp(log_reward)
        state.log_reward = log_reward
        return {
            'reward': reward,
            'log_reward': log_reward
        }

    def __call__(self, state: PhylogeneticTreeState):
        """
        compute reward
        """
        if not state.is_done:
            state.log_reward = None
            return None
        return self.exponential_reward(state)


class PhylogeneticTreeEnv(object):

    def __init__(self, cfg, sequences):
        self.cfg = cfg
        self.sequences = sequences
        self.reward_fn = PhyloTreeReward(cfg.ENV.REWARD)
        self.chars_dict = CHARACTERS_MAPS[cfg.ENV.SEQUENCE_TYPE]
        self.seq_arrays = [self.seq2array(seq) for seq in self.sequences]
        self.evolution_model = EvolutionModel(cfg.ENV.EVOLUTION_MODEL)
        self.type = cfg.ENV.ENVIRONMENT_TYPE
        # store for each number of trees, what are the possible combination of pairs
        self.tree_pairs_dict = {}
        self.action_indices_dict = {}
        for n in range(2, len(self.sequences) + 1):
            tree_pairs = list(itertools.combinations(list(np.arange(n)), 2))
            self.tree_pairs_dict[n] = tree_pairs
            self.action_indices_dict[n] = {pair: idx for idx, pair in enumerate(tree_pairs)}
        self.parsimony_problem = False
        self.edge_env = build_edge_env(cfg)
        self.seq_pos_indices = np.arange(self.seq_arrays[0].shape[0])

    def seq2array(self, seq):
        seq = [self.chars_dict[x] for x in seq]
        data = np.array(seq)
        return data

    def sample(self, n, rotate_seq_pos=False):

        trajectories = []
        for _ in range(n):
            trajectories.append(self.generate_random_trajectory(rotate_seq_pos=rotate_seq_pos))
        return trajectories

    def generate_random_trajectory(self, rotate_seq_pos=False):

        state = self.get_initial_state(rotate_seq_pos)
        trajectory = Trajectory(state)
        while not state.is_done:
            n = len(state.subtrees)
            actions = list(range(int(n * (n - 1) / 2)))
            tree_action_pair = random.sample(actions, 1)[0]
            edge_action_pair = self.edge_env.generate_random_actions(at_root=(n == 2))
            a = {'tree_action': tree_action_pair, 'edge_action': edge_action_pair}
            next_state = self.apply_action(state, a)
            r = self.reward_fn(next_state)
            trajectory.update(next_state, a, r, next_state.is_done)
            state = next_state

        return trajectory

    def get_initial_state(self, rotate_seq_pos=False):

        if rotate_seq_pos:
            pos = np.random.permutation(self.seq_pos_indices)
            seq_arrays = [x[pos, :] for x in self.seq_arrays]
        else:
            seq_arrays = self.seq_arrays

        phylo_trees = [PhyloTree(self.evolution_model, at_root=False, root_seq_data=[seq, [idx]]) for idx, seq in
                       enumerate(seq_arrays)]
        state = PhylogeneticTreeState(phylo_trees)
        return state

    def transition(self, state: PhylogeneticTreeState, action):

        assert not state.is_done
        new_state = self.apply_action(state, action)
        done = new_state.is_done
        reward = self.reward_fn(new_state)

        return state, new_state, action, reward, done

    def determine_done(self, state: PhylogeneticTreeState):

        return state.is_done

    def get_parent_states(self, state: PhylogeneticTreeState):
        """
        Given state, return a list of possible parents
        """
        state_subtrees = state.subtrees
        possible_parents = []

        if len(state_subtrees) == 1:
            # note that although the last state stores a rooted tree, we are working on the unrooted scenario
            # we first convert the state tree to unrooted, there are 2n-3 rooted trees correspond to the same unrooted tree
            # the last transition forward would be 2n -3 rooted tree -> unrooted tree
            # and backward transition is unrooted tree -> 2n - 3 rooted tree split at the top
            unrooted_tree = state_subtrees[0]
            for edge_idx in range(2 * len(self.seq_arrays) - 3):
                rooted_tree = unrooted_tree.to_rooted_tree(edge_idx, None, None, self)
                trees = [rooted_tree.left_tree_data['tree'], rooted_tree.right_tree_data['tree']]
                tree_action = 0
                left_branch = rooted_tree.left_tree_data['branch_length']
                right_branch = rooted_tree.right_tree_data['branch_length']
                edge_action = self.edge_env.edges_2_actions(left_branch, right_branch, at_root=True)
                action = {'tree_action': tree_action, 'edge_action': edge_action}
                possible_parents.append([PhylogeneticTreeState(trees), action])
        else:
            for idx, subtree in enumerate(state_subtrees):
                if subtree.left_tree_data is not None:
                    trees = state_subtrees[:idx] + state_subtrees[idx + 1:] + [subtree.left_tree_data['tree'],
                                                                               subtree.right_tree_data['tree']]
                    # sorting is for the leafsets_signature
                    trees = sorted(trees, key=lambda x: x.min_seq_index)
                    tree_pos = {x.min_seq_index: idx for idx, x in enumerate(trees)}
                    left_signiture = subtree.left_tree_data['tree'].min_seq_index
                    right_signiture = subtree.right_tree_data['tree'].min_seq_index
                    action_pair = (tree_pos[left_signiture], tree_pos[right_signiture])
                    tree_action = self.action_indices_dict[len(trees)][action_pair]
                    left_branch = subtree.left_tree_data['branch_length']
                    right_branch = subtree.right_tree_data['branch_length']
                    edge_action = self.edge_env.edges_2_actions(left_branch, right_branch, at_root=False)
                    action = {'tree_action': tree_action, 'edge_action': edge_action}
                    possible_parents.append([PhylogeneticTreeState(trees), action])

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
            # for unrooted tree problem, the last state has 2n- 3 ways to go backward
            if len(subtrees) == 1:
                all_number_parents.append(2 * len(self.seq_arrays) - 3)
            else:
                all_number_parents.append(sum([tree.left_tree_data is not None for tree in subtrees]))
        return all_number_parents

    def sample_last_step_parent(self, state: PhylogeneticTreeState, rotate_seq_pos=False):

        unrooted_tree = state.subtrees[0]
        edge_id = random.sample(range(2 * len(self.seq_arrays) - 3), 1)[0]
        rooted_tree = unrooted_tree.to_rooted_tree(edge_id, None, None, self, rotate_seq_pos=rotate_seq_pos)
        trees = [rooted_tree.left_tree_data['tree'], rooted_tree.right_tree_data['tree']]
        tree_action = 0
        left_branch = rooted_tree.left_tree_data['branch_length']
        right_branch = rooted_tree.right_tree_data['branch_length']
        edge_action = self.edge_env.edges_2_actions(left_branch, right_branch, at_root=True)
        action = {'tree_action': tree_action, 'edge_action': edge_action}
        state = PhylogeneticTreeState(trees)
        return state, action

    def state_to_trajectory(self, state: PhylogeneticTreeState, rotate_seq_pos=False):
        """
        Given a terminal state, randomly select a possible trajectory
        """
        assert state.is_done, 'only supports terminal states'
        transitions = []

        # first step
        parent, a = self.sample_last_step_parent(state, rotate_seq_pos=rotate_seq_pos)
        transitions.append([parent, state, a, 0, False])
        state = parent

        # rest steps
        while not state.is_initial:
            parents = self.get_parent_states(state)
            parent, a = random.sample(parents, 1)[0]
            transitions.append([parent, state, a, 0, False])
            state = parent

        transitions = transitions[::-1]
        transitions[-1][-1] = True
        trajectory = Trajectory(None)
        trajectory.actions = [x[2] for x in transitions]
        trajectory.transitions = transitions
        trajectory.current_state = transitions[-1][1]
        trajectory.done = True
        trajectory.reward = self.reward_fn(trajectory.current_state)
        return trajectory

    def trajectory_permute_leaves(self, trajectory: Trajectory):
        """
        for a terminal state, exchange the leaf species and arrive at a new terminal state which has the same
        (unlabelled) binary tree topology hence having the identical number of trajectories leading to it
        """
        all_action = []
        for state, next_state, a, reward, done in trajectory.transitions:
            all_action.append(a)

        initial_subtrees = [PhyloTree(self.evolution_model, False, None, None, [seq, [idx]]) for idx, seq in
                            enumerate(self.seq_arrays)]
        random.shuffle(initial_subtrees)
        state = PhylogeneticTreeState(initial_subtrees)
        trajectory = Trajectory(state)
        for a in all_action:
            next_state = self.apply_action(state, a)
            r = self.reward_fn(next_state)
            trajectory.update(next_state, a, r, next_state.is_done)
            state = next_state

        return trajectory

    def tree_to_trajectory(self, tree: UnrootedPhyloTree, rotate_seq_pos=False):
        """
        tree here is unrooted
        :param tree:
        :return:
        """
        traj = self.state_to_trajectory(PhylogeneticTreeState([tree]), rotate_seq_pos=rotate_seq_pos)
        return traj

    def apply_action(self, state, action):

        tree_pair_action = action['tree_action']
        edge_pair_action = action['edge_action']
        l_length, r_length = self.edge_env.actions_2_edges(edge_pair_action, at_root=(state.num_trees == 2))
        tree_pairs = self.tree_pairs_dict[state.num_trees]
        i, j = tree_pairs[tree_pair_action]
        new_trees = []
        for idx in range(len(state.subtrees)):
            if idx not in (i, j):
                new_trees.append(state.subtrees[idx])

        left_tree_data = {
            'tree': state.subtrees[i],
            'branch_length': l_length
        }

        right_tree_data = {
            'tree': state.subtrees[j],
            'branch_length': r_length
        }

        at_root = len(state.subtrees) == 2
        new_tree = PhyloTree(self.evolution_model, at_root=at_root, left_tree_data=left_tree_data,
                             right_tree_data=right_tree_data)
        if at_root:
            unrooted_tree = new_tree.to_unrooted_tree()
            state = PhylogeneticTreeState([unrooted_tree])
        else:
            new_trees.append(new_tree)
            # sorting is for the leafsets_signature
            new_trees = sorted(new_trees, key=lambda x: x.min_seq_index)
            state = PhylogeneticTreeState(new_trees)
        return state

    def retrieve_tree_pairs(self, batch_nb_trees, batch_action):
        """
        Retrieve the pairs of trees to be joined
        :param batch_nb_trees: list of total number of trees per state
        :param batch_action:   list of actions to apply per state
        :return: list of pairs of trees
        """
        tree_pairs = []
        if type(batch_nb_trees) == torch.Tensor:
            batch_nb_trees = batch_nb_trees.cpu().numpy()
        if type(batch_action) == torch.Tensor:
            batch_action = batch_action.cpu().numpy()
        for num_trees, a in zip(batch_nb_trees, batch_action):
            tree_pair = self.tree_pairs_dict[num_trees][a]
            tree_pairs.append(tree_pair)
        return tree_pairs


def build_one_step_likelihood_env(cfg, all_seqs):
    env = PhylogeneticTreeEnv(cfg, all_seqs)
    state2input = PhylogeneticTreeState2Input(cfg, len(all_seqs))
    return env, state2input
