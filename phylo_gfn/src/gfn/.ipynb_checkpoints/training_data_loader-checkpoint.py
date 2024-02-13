import os
import torch
import pickle
import numpy as np
import time
from src.env.build import build_env
from torch.utils.data import Dataset, DataLoader
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.build import build_gfn
from src.utils.utils import dummy_collate_fn, linear_schedule
from heapq import heappush, heappushpop
import random
import multiprocessing as mp
import ctypes
from scipy.stats import truncnorm


class TrainingDataLoader(object):

    def __init__(self, cfg, generator_path, best_tree_path, generator_device, sequences):
        self.cfg = cfg
        self.generator_path = generator_path
        self.best_tree_path = best_tree_path
        self.generator_device = generator_device
        self.sequences = sequences
        num_workers = cfg.GFN.TRAINING_DATA_LOADER.NUM_WORKERS
        # four types of possible training examples
        batch_size = cfg.GFN.TRAINING_DATA_LOADER.BEST_STATE_BATCH_SIZE + \
                     cfg.GFN.TRAINING_DATA_LOADER.RANDOM_BATCH_SIZE + \
                     cfg.GFN.TRAINING_DATA_LOADER.GFN_BATCH_SIZE + \
                     cfg.GFN.TRAINING_DATA_LOADER.GFN_FIXED_SHAPE_BATCH_SIZE
        # the multiprocessing array is shared between dataloader process and the main process
        # for the purpose of sending loss values to the replay buffer
        if cfg.GFN.TRAINING_DATA_LOADER.BEST_STATE_BATCH_SIZE > 0:
            if num_workers > 0:
                self.shared_arrays = [mp.Array(ctypes.c_float, 10 * batch_size) for _ in range(num_workers)]
            else:
                self.shared_arrays = [mp.Array(ctypes.c_float, 10 * batch_size)]
        else:
            self.shared_arrays = None

    def build_data_loader(self, start_eps=None, end_eps=None,
                          start_scales_sampling_mu=None, end_scales_sampling_mu=None):
        loader_cfg = self.cfg.GFN.TRAINING_DATA_LOADER

        # reset the remaining counters from the last epoch
        if self.shared_arrays is not None:
            for shared_array in self.shared_arrays:
                shared_array[0] = 0.

        dataset = RolloutDataset(self.cfg, self.generator_path, self.generator_device, self.best_tree_path,
                                 self.sequences, self.shared_arrays, start_eps, end_eps,
                                 start_scales_sampling_mu, end_scales_sampling_mu)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=loader_cfg.NUM_WORKERS,
                                collate_fn=dummy_collate_fn, pin_memory=loader_cfg.PIN_MEMORY,
                                persistent_workers=loader_cfg.NUM_WORKERS > 0)
        # initialize data loader
        dataloader_iter = iter(dataloader)
        return dataloader_iter


class RolloutDataset(Dataset):

    def __init__(self, cfg, generator_path, generator_device, best_trees_path, sequences, shared_arrays=None,
                 start_eps=None, end_eps=None, start_scales_sampling_mu=None, end_scales_sampling_mu=None):
        self.cfg = cfg
        self.generator_path = generator_path
        self.generator_device = generator_device
        self.best_trees_path = best_trees_path
        self.sequences = sequences
        self.loss_type = cfg.GFN.LOSS_TYPE
        self.update_model_freq = cfg.GFN.TRAINING_DATA_LOADER.FREQ_UPDATE_MODEL_WEIGHTS
        self.num_workers = cfg.GFN.TRAINING_DATA_LOADER.NUM_WORKERS
        self.assets_built = False

        # gradient accumulation option
        splits_num = cfg.GFN.TRAINING_DATA_LOADER.MINI_BATCH_SPLITS
        self.steps_per_epoch = int(cfg.GFN.TRAINING_DATA_LOADER.STEPS_PER_EPOCH * splits_num)

        # random sampling options
        self.random_batch_size = int(cfg.GFN.TRAINING_DATA_LOADER.RANDOM_BATCH_SIZE / splits_num)

        # on-policy sampling options
        self.gfn_batch_size = int(cfg.GFN.TRAINING_DATA_LOADER.GFN_BATCH_SIZE / splits_num)
        self.rollout_random_prob = cfg.GFN.TRAINING_DATA_LOADER.RANDOM_ACTION_PROB

        # replay buffer sampling options
        self.best_state_batch_size = int(cfg.GFN.TRAINING_DATA_LOADER.BEST_STATE_BATCH_SIZE / splits_num)
        self.best_tree_buffer_size = cfg.GFN.TRAINING_DATA_LOADER.BEST_TREES_BUFFER_SIZE
        self.perturb_buffered_tree = cfg.GFN.TRAINING_DATA_LOADER.PERTURB_BUFFERED_TREE
        self.prev_iter_unrooted_trees = []
        self.shared_arrays = shared_arrays

        # condition scale options
        self.condition_on_scale = cfg.GFN.CONDITION_ON_SCALE
        self.scales_set = cfg.GFN.SCALES_SET
        self.start_scales_sampling_mu = start_scales_sampling_mu
        self.end_scales_sampling_mu = end_scales_sampling_mu
        self.scales_sampling_sigma = cfg.GFN.SCALES_SAMPLING_SIGMA
        self.steps = 0

        self.compute_partial_reward = (self.loss_type in ('FLDB', 'FLSUBTB'))

        # epsilon greedy exploration
        self.do_eps_annealing = self.cfg.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING
        self.non_eps_trajs = int(self.cfg.GFN.TRAINING_DATA_LOADER.EPS_ANNEALING_DATA.NON_EPS_TRAJS / splits_num)
        self.num_per_worker_iters = int(self.steps_per_epoch / self.num_workers) \
            if self.num_workers > 0 else self.steps_per_epoch
        self.start_eps = start_eps
        self.end_eps = end_eps

        # for sampling terminal states with fixed tree shapes
        self.gfn_fixed_shape_batch_size = int(cfg.GFN.TRAINING_DATA_LOADER.GFN_FIXED_SHAPE_BATCH_SIZE / splits_num)

    def build_assets(self):
        self.env, self.state2input = build_env(self.cfg, self.sequences)
        self.rollout_worker = RolloutWorker(self.cfg.GFN, self.env, self.state2input)
        self.generator = build_gfn(self.cfg, self.state2input, self.env, self.generator_device)
        self.__load_generator__()
        # replay buffer setup
        if self.best_state_batch_size > 0:
            self.replay_buffer = ReplayBuffer(
                self.env, self.best_trees_path, self.best_state_batch_size, self.best_tree_buffer_size,
                self.perturb_buffered_tree)
            if self.shared_arrays is not None:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:
                    # a dummy id
                    self.process_idx = 0
                else:
                    self.process_idx = worker_info.id
                self.shared_array = self.shared_arrays[self.process_idx]
            else:
                self.shared_array = None
                self.process_idx = None
        else:
            self.replay_buffer = None
            self.shared_array = None
            self.process_idx = None

        if self.gfn_fixed_shape_batch_size > 0:
            path = self.cfg.GFN.TRAINING_DATA_LOADER.FIXED_SHAPE_TREES_PATH
            self.fixed_shape_tree = pickle.load(open(path, 'rb'))

        self.assets_built = True

    def __load_generator__(self):
        loaded = False
        load_counter = 0
        while not loaded:
            try:
                self.generator.load(self.generator_path)
            except Exception as e:
                if load_counter < 5:
                    time.sleep(1)
                    load_counter += 1
                    pass
                else:
                    raise e
            loaded = True
        self.generator.eval()

    def __len__(self):
        return self.steps_per_epoch

    def sample_trajs_from_fixed_shape_trees(self):

        if self.gfn_fixed_shape_batch_size == 0:
            return []

        trajs = []
        # uniformly sample some tree shapes and generate random backward trajectories
        # the tree actions in these trajectories will be used to guide the GFN rollout samples
        unrooted_trees = random.choices(self.fixed_shape_tree, k=self.gfn_fixed_shape_batch_size)
        for unrooted_tree in unrooted_trees:
            traj = self.env.tree_to_trajectory(unrooted_tree)
            trajs.append(traj)
        return trajs

    def __getitem__(self, idx):

        if not self.assets_built:
            self.build_assets()

        if self.shared_array is not None and self.shared_array[0] > 0.:
            self.shared_array.acquire()
            counter = int(self.shared_array[0])
            prev_loss = self.shared_array[1:1 + counter]
            self.shared_array[0] = 0.
            self.shared_array.release()
            prev_trees = self.prev_iter_unrooted_trees[:counter]
            self.replay_buffer.save_best_trees(prev_trees, prev_loss)
            # clear all prev iter trees
            self.prev_iter_unrooted_trees = self.prev_iter_unrooted_trees[counter:]

        if self.steps > 0 and self.steps % self.update_model_freq == 0:
            self.__load_generator__()

        with torch.no_grad():

            if self.condition_on_scale:
                scales_sampling_mu = linear_schedule(self.start_scales_sampling_mu, self.end_scales_sampling_mu,
                                                     self.num_per_worker_iters, self.steps)
                truncnorm_a = (min(self.scales_set) - scales_sampling_mu) / self.scales_sampling_sigma
                truncnorm_b = (max(self.scales_set) - scales_sampling_mu) / self.scales_sampling_sigma
                scales = truncnorm.rvs(truncnorm_a, truncnorm_b,
                                       loc=scales_sampling_mu,
                                       scale=self.scales_sampling_sigma,
                                       size=self.gfn_batch_size + self.gfn_fixed_shape_batch_size).tolist()
                rollout_scales = [scales_sampling_mu] * (self.gfn_batch_size + self.gfn_fixed_shape_batch_size)
            else:
                scales = None
                rollout_scales = None
            if self.start_eps is not None:
                eps = linear_schedule(self.start_eps, self.end_eps, self.num_per_worker_iters, self.steps)
            else:
                eps = self.rollout_random_prob

            # overall, there are three types of trees that are sampled from the GFN model,
            # total: gfn_batch_size + gfn_fixed_shape_batch_size
            # (1) eps annealed, for exploration purpose
            # (2) essentially on-policy by enforcing a small rand_action_prob such as 0.01 or 0.001
            # (3) trajs sampled with fixed tree shape, important for the likelihood case
            fixed_shape_tree_trajs = [None] * self.gfn_batch_size
            if self.do_eps_annealing and self.non_eps_trajs > 0:
                list_rand_action_prob = [self.rollout_random_prob] * self.non_eps_trajs + \
                                        [eps] * (self.gfn_batch_size - self.non_eps_trajs)
            else:
                list_rand_action_prob = [eps] * self.gfn_batch_size
            if self.gfn_fixed_shape_batch_size > 0:
                fixed_shape_tree_trajs += self.sample_trajs_from_fixed_shape_trees()
                list_rand_action_prob += [eps] * self.gfn_fixed_shape_batch_size
            # we call gfn unroll once for computational efficiency
            gfn_trajs = self.rollout_worker.rollout(
                self.generator, self.gfn_batch_size + self.gfn_fixed_shape_batch_size,
                random_action_prob=list_rand_action_prob,
                scales=rollout_scales,
                fixed_shape_trajs=fixed_shape_tree_trajs)

            # random samples from the env
            random_trajs = self.env.sample(self.random_batch_size)
            trajs = gfn_trajs + random_trajs

            # samples from replay buffer
            if self.best_state_batch_size > 0:
                best_states_trajs = self.replay_buffer.sample_trajs_from_best_trees()
                trajs = trajs + best_states_trajs

            batch = self.combine_trajectories(trajs)

            if self.shared_array is not None:
                if self.env.parsimony_problem:
                    self.prev_iter_unrooted_trees.extend(
                        [traj.current_state.subtrees[0].to_unrooted_tree() for traj in trajs])
                else:
                    self.prev_iter_unrooted_trees.extend([traj.current_state.subtrees[0] for traj in trajs])
                batch['process_idx'] = self.process_idx

            # whether this is the last batch of the current process
            is_last_batch = idx >= (self.steps_per_epoch - self.num_workers)
            # if this is the last batch of the current process, return also all best states
            if is_last_batch and self.replay_buffer is not None:
                batch['best_seen_trees'] = self.replay_buffer.best_trees

            if self.condition_on_scale:
                batch_size = len(trajs)
                sampled_scales = scales + truncnorm.rvs(
                    truncnorm_a, truncnorm_b, loc=scales_sampling_mu, scale=self.scales_sampling_sigma,
                    size=batch_size - self.gfn_batch_size - self.gfn_fixed_shape_batch_size).tolist()
                # for updating the reward
                batch['scale'] = torch.tensor(np.array(sampled_scales, dtype=np.float32)).reshape(-1, 1)

            if self.compute_partial_reward:
                # todo, consider adding log partial rewards for likelihood problem,
                #  if we were to test temp-conditioned likelihood model
                log_partial_rewards = [[t[0].log_partial_reward for t in traj.transitions] for traj in trajs]
                log_partial_rewards = torch.tensor(np.array(log_partial_rewards))
                last_log_partial_rewards = [x.current_state.log_partial_reward for x in trajs]
                last_log_partial_rewards = torch.tensor(np.array(last_log_partial_rewards))
                log_partial_rewards = torch.cat([log_partial_rewards, last_log_partial_rewards.reshape(-1, 1)], dim=1)
                batch['log_partial_rewards'] = log_partial_rewards.float()
            batch['eps'] = eps
            if self.condition_on_scale:
                batch['scales_sampling_mu'] = scales_sampling_mu
            self.steps += 1
            return batch

    def combine_trajectories(self, list_trajs):

        if self.loss_type == 'TB':
            input_dict = self.combine_trajectories_tb(list_trajs)
        else:
            input_dict = self.combine_trajectories_db(list_trajs)

        # if one step model, also store the action indices for the pairwise logits table
        if 'pairwise_action_reverse_tensor' in input_dict:
            reverse_tensor = input_dict['pairwise_action_reverse_tensor']
            batch_action = input_dict['batch_action']
            batch_pairwise_action = reverse_tensor.gather(1, batch_action.unsqueeze(-1)).squeeze(-1)
            input_dict['batch_pairwise_action'] = batch_pairwise_action  # unpadded to padded actions
        return input_dict

    def combine_trajectories_tb(self, list_trajs):
        """
        preparing all needed tensor to accelerate loss computation
        NOTE: we no longer return list_trajs as it tend to cause memory leaks for pytorch dataloaders
        """
        batch_state, batch_traj_idx = [], []
        batch_action, batch_pb_log, batch_log_reward = [], [], []
        batch_edge_action = []
        for i, traj in enumerate(list_trajs):
            batch_traj_idx.extend([i] * len(traj.transitions))
            # for likelihood-based, the last state of a traj is an unrooted tree,
            # thus the traditional get_parent_state can incur large unnecessary computation
            parents_num = self.env.get_number_parents_trajectory(traj)
            pb_log_sum = np.log([1 / n for n in parents_num]).sum()
            batch_pb_log.append(pb_log_sum)
            batch_log_reward.append(traj.reward['log_reward'])
            for state, next_state, action, reward, done in traj.transitions:
                batch_state.append(state)
                if self.env.parsimony_problem:
                    batch_action.append(action)
                else:
                    batch_action.append(action['tree_action'])
                    batch_edge_action.append(action['edge_action'])

        input_dict = self.state2input.states2inputs(batch_state)
        input_dict['batch_pb_log'] = torch.tensor(np.array(batch_pb_log)).float()
        # NOTE: reward shaping may lead to an extreme range of reward
        input_dict['batch_log_reward'] = torch.tensor(batch_log_reward).float()
        input_dict['batch_action'] = torch.tensor(batch_action).long()
        input_dict['batch_traj_idx'] = torch.tensor(batch_traj_idx).long()
        input_dict['batch_size'] = len(list_trajs)

        if not self.env.parsimony_problem:
            batch_edge_action = np.array(batch_edge_action)
            if self.env.cfg.GFN.MODEL.EDGES_MODELING.DISTRIBUTION in ['CATEGORICAL', 'CATEGORICAL_INDEPENDENT']:
                input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).long()
            else:
                input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).float()

        return input_dict

    def combine_trajectories_db(self, list_trajs):
        """
        preparing all needed tensor to accelerate loss computation
        NOTE: we no longer return list_trajs as it tend to cause memory leaks for pytorch dataloaders
        """
        batch_state, batch_traj_idx = [], []
        batch_action, batch_pb_log, batch_log_reward = [], [], []
        batch_edge_action = []
        for i, traj in enumerate(list_trajs):
            batch_traj_idx.extend([i] * len(traj.transitions))
            # for likelihood-based, the last state of a traj is an unrooted tree,
            # thus the traditional get_parent_state can incur large unnecessary computation
            parents_num = self.env.get_number_parents_trajectory(traj)
            pb_log = np.log([1 / n for n in parents_num])  # note here is different to the TB objective
            batch_pb_log.append(pb_log)
            batch_log_reward.append(traj.reward['log_reward'])
            for state, next_state, action, reward, done in traj.transitions:
                batch_state.append(state)
                if self.env.parsimony_problem:
                    batch_action.append(action)
                else:
                    batch_action.append(action['tree_action'])
                    batch_edge_action.append(action['edge_action'])

        input_dict = self.state2input.states2inputs(batch_state)
        input_dict['batch_pb_log'] = torch.tensor(np.array(batch_pb_log)).float()
        # NOTE: reward shaping may lead to an extreme range of reward
        input_dict['batch_log_reward'] = torch.tensor(batch_log_reward).float()
        input_dict['batch_action'] = torch.tensor(batch_action).long()
        input_dict['batch_traj_idx'] = torch.tensor(batch_traj_idx).long()
        input_dict['batch_size'] = len(list_trajs)

        if not self.env.parsimony_problem:
            batch_edge_action = np.array(batch_edge_action)
            if self.env.cfg.GFN.MODEL.EDGES_MODELING.DISTRIBUTION in ['CATEGORICAL', 'CATEGORICAL_INDEPENDENT']:
                input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).long()
            else:
                input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).float()

        return input_dict


class ReplayBuffer:

    def __init__(self, env, best_trees_path, best_state_batch_size, best_tree_buffer_size, perturb_buffered_tree):
        super().__init__()
        self.env = env
        self.nb_seqs = len(self.env.sequences)
        self.best_trees_path = best_trees_path
        self.best_state_batch_size = best_state_batch_size
        self.best_tree_buffer_size = best_tree_buffer_size
        self.perturb_buffered_tree = perturb_buffered_tree
        self.initialize_best_trees()

    def initialize_best_trees(self):
        self.best_trees = []
        self.tree_to_loss = {}
        if os.path.isfile(self.best_trees_path):
            best_trees = pickle.load(open(self.best_trees_path, 'rb'))
            if len(best_trees) > self.best_tree_buffer_size:
                unrooted_trees = random.sample(best_trees, self.best_tree_buffer_size)
            else:
                unrooted_trees = best_trees
        else:
            trajs = self.env.sample(1000)
            if self.env.parsimony_problem:
                rooted_trees = sorted([x.current_state.subtrees[0] for x in trajs],
                                      key=lambda x: x.total_mutations)[:self.best_tree_buffer_size]
                unrooted_trees = [tree.to_unrooted_tree() for tree in rooted_trees]
            else:
                unrooted_trees = sorted([x.current_state.subtrees[0] for x in trajs],
                                        key=lambda x: - x.log_score)[:self.best_tree_buffer_size]
        for tree in unrooted_trees:
            heappush(self.best_trees, tree)
            self.tree_to_loss[tree.leafsets_signature] = (1e10, 0)

    def sample_trajs_from_best_trees(self):

        current_buffer_size = len(self.best_trees)
        if current_buffer_size == 0:
            return []

        replace = False if self.best_state_batch_size <= current_buffer_size else True
        all_tree_loss = [self.tree_to_loss[tree.leafsets_signature][0] for tree in self.best_trees]
        prob = np.array(all_tree_loss) / np.sum(all_tree_loss)
        unrooted_trees = np.random.choice(self.best_trees, self.best_state_batch_size, replace=replace, p=prob)

        trajs = []
        if self.env.parsimony_problem:
            # tree shape perturbation is only implemented for the parsimony case
            branch_swapping_algo = np.random.choice(
                [0, 1, 2, 3], size=self.best_state_batch_size, p=[0.1, 0.1, 0.1, 0.7])
            root_insert_edge_idx = np.random.randint(0, 2 * self.nb_seqs - 3, self.best_state_batch_size)
            for i, unrooted_tree in enumerate(unrooted_trees):
                if self.perturb_buffered_tree:
                    if branch_swapping_algo[i] == 0:
                        unrooted_tree = unrooted_tree.nearest_neighbour_interchange()
                    elif branch_swapping_algo[i] == 1:
                        unrooted_tree = unrooted_tree.subtree_pruning_regrafting()
                    elif branch_swapping_algo[i] == 2:
                        unrooted_tree = unrooted_tree.bisection_reconnection()
                traj = self.env.tree_to_trajectory(unrooted_tree.to_rooted_tree(root_insert_edge_idx[i]))
                trajs.append(traj)
        else:
            for unrooted_tree in unrooted_trees:
                # the likelihood env terminal state is an unrooted tree, opposite to the parsimony env where the
                # terminal state is still a rooted tree which needs to be manually converted to an unrooted version
                traj = self.env.tree_to_trajectory(unrooted_tree)
                trajs.append(traj)
        return trajs

    def save_best_trees(self, batch_unrooted_trees, batch_loss):

        for idx, (tree, loss) in enumerate(zip(batch_unrooted_trees, batch_loss)):
            leafsets_signature = tree.leafsets_signature
            if leafsets_signature not in self.tree_to_loss:
                self.tree_to_loss[leafsets_signature] = (loss, 1)  # to maintain a running average of the loss
                if len(self.best_trees) >= self.best_tree_buffer_size:
                    dropped_tree = heappushpop(self.best_trees, tree)
                    del self.tree_to_loss[dropped_tree.leafsets_signature]
                else:
                    heappush(self.best_trees, tree)
            else:
                avg_loss, counter = self.tree_to_loss[leafsets_signature]
                new_avg_loss = (avg_loss * counter + loss) / (counter + 1)
                counter += 1
                self.tree_to_loss[leafsets_signature] = (new_avg_loss, counter)
        # print(list(self.tree_to_loss.values()))
