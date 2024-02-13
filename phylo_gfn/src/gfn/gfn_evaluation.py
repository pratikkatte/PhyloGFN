import gc
import torch
import numpy as np
from tqdm import tqdm
import random
from scipy.stats import pearsonr
from collections import defaultdict
from src.utils.utils import dummy_collate_fn, chunks, process_trajectories_tb
from torch.utils.data import Dataset, DataLoader
from scipy.special import logsumexp


class EvalDataLoader:

    def __init__(self, env, state2input, num_workers, batch_size, states, trajectories_num=100, scale=None):
        self.env = env
        self.state2input = state2input
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.states = states
        self.states_num = len(self.states)
        self.trajectories_num = trajectories_num
        self.scale = scale
        self.nb_rounds = int(np.ceil(trajectories_num / batch_size))  # you need this for DS datasets

    def __iter__(self):
        states = self.states
        dataset = EvalDataset(states, self.env, self.state2input, self.trajectories_num, self.batch_size,
                              self.scale)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers,
                                collate_fn=dummy_collate_fn, pin_memory=True, persistent_workers=self.num_workers > 0)

        for b in dataloader:
            yield b

        del b, dataset, dataloader
        gc.collect()


class EvalDataset(Dataset):

    def __init__(self, states, env, state2input, trajectories_num, batch_size, scale=None):
        self.states = states
        self.env = env
        self.state2input = state2input
        self.trajectories_num = trajectories_num
        self.batch_size = batch_size
        self.generate_all_trajectories = self.trajectories_num == -1
        self.nb_rounds = int(np.ceil(trajectories_num / batch_size))
        self.scale = scale

    def __len__(self):
        return len(self.states) * self.nb_rounds

    def __getitem__(self, idx):
        state_idx = idx % len(self.states)
        state = self.states[state_idx]

        # generate trajectories
        if self.generate_all_trajectories:
            list_trajs = self.env.state_to_all_trajectory(state)
        else:
            list_trajs = []
            for _ in range(self.batch_size):
                traj = self.env.state_to_trajectory(state)
                list_trajs.append(traj)

        input_dict = self.combine_trajectories_tb(list_trajs)
        input_dict['scale'] = torch.tensor(np.array([self.scale] * self.batch_size, dtype=np.float32)).reshape(-1, 1)

        log_reward = list_trajs[0].reward['log_reward']
        # getting rid of the constant C in the reward definition
        log_reward -= self.env.reward_fn.C / self.env.reward_fn.scale
        if self.scale is not None:
            log_reward /= self.scale

        input_dict.update({
            'state_idx': state_idx,
            'log_reward': log_reward,
        })
        return input_dict

    def combine_trajectories_tb(self, list_trajs):
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
                if not self.env.parsimony_problem:
                    batch_action.append(action['tree_action'])
                    batch_edge_action.append(action['edge_action'])
                else:
                    batch_action.append(action)

        input_dict = self.state2input.states2inputs(batch_state)
        input_dict['batch_pb_log'] = torch.tensor(batch_pb_log).float()
        input_dict['batch_log_reward'] = torch.tensor(batch_log_reward).float()
        input_dict['batch_action'] = torch.tensor(batch_action).long()

        if not self.env.parsimony_problem:
            if self.env.cfg.GFN.MODEL.EDGES_MODELING.DISTRIBUTION in ['CATEGORICAL', 'CATEGORICAL_INDEPENDENT']:
                input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).long()
            else:
                input_dict['batch_edge_action'] = torch.tensor(batch_edge_action).float()

        input_dict['batch_traj_idx'] = torch.tensor(batch_traj_idx).long()
        input_dict['batch_size'] = len(list_trajs)

        if 'pairwise_action_reverse_tensor' in input_dict:
            reverse_tensor = input_dict['pairwise_action_reverse_tensor']
            batch_action = input_dict['batch_action']
            batch_pairwise_action = reverse_tensor.gather(1, batch_action.unsqueeze(-1)).squeeze(-1)
            input_dict['batch_pairwise_action'] = batch_pairwise_action

        return input_dict


class GFNEvaluator(object):

    def __init__(self, evaluation_cfg, rollout_worker, generator, states=None, verbose=True, scales_set=None):
        self.env = rollout_worker.env
        self.rollout_worker = rollout_worker
        self.generator = generator
        self.evaluation_cfg = evaluation_cfg
        self.verbose = verbose
        if states is not None:
            self.states = states
        else:
            self.states = self.generate_initial_states()
        self.scales_set = scales_set
        self.condition_on_scale = (scales_set is not None)

    def evaluate_gfn_samples(self, scale=None):

        if self.condition_on_scale:
            scales = [scale] * self.evaluation_cfg.MUTATIONS_TRAJS
        else:
            scales = None
        trajs = self.rollout_worker.rollout(
            self.generator, self.evaluation_cfg.MUTATIONS_TRAJS, random_action_prob=0.0, scales=scales)
        states = [x.current_state for x in trajs]
        if self.env.parsimony_problem:
            mutations = [x.subtrees[0].total_mutations for x in states]
            mut_mean, mut_std = np.mean(mutations), np.std(mutations)
            mut_min, mut_max = np.min(mutations), np.max(mutations)
            return {
                'trajs': trajs,
                'states': states,
                'mutations': mutations,
                'mut_mean': mut_mean,
                'mut_std': mut_std,
                'mut_min': mut_min,
                'mut_max': mut_max
            }
        else:
            log_scores = [x.subtrees[0].log_score for x in states]
            log_scores_mean, log_scores_std = np.mean(log_scores), np.std(log_scores)
            log_scores_min, log_scores_max = np.min(log_scores), np.max(log_scores)
            return {
                'trajs': trajs,
                'states': states,
                'log_scores': log_scores,
                'log_scores_mean': log_scores_mean,
                'log_scores_std': log_scores_std,
                'log_scores_min': log_scores_min,
                'log_scores_max': log_scores_max
            }

    def estimate_gfn_states_sampling_probs(self, states, scale):

        generator = self.generator.eval()
        prob_estimation_method = self.evaluation_cfg.PROB_ESTIMATION_METHOD
        assert prob_estimation_method in ['IMPORTANCE_SAMPLING', 'BIASED_RANDOM_SAMPLING']

        eval_loader = EvalDataLoader(
            self.env, generator.state2input, self.evaluation_cfg.NUM_WORKERS, self.evaluation_cfg.BATCH_SIZE,
            states, self.evaluation_cfg.TRAJECTORIES_PER_STATES, scale)
        states_log_rewards, state_to_logp, state_to_log_reward = [], defaultdict(list), {}
        if self.verbose:
            bar = tqdm(total=len(states) * eval_loader.nb_rounds * eval_loader.batch_size, position=0, leave=True,
                       desc='Total trajectories pearsonr evaluation progress')
        else:
            bar = None

        with torch.no_grad():
            for input_dict in eval_loader:
                generator_ret_dict = self.generator(input_dict, compute_edge=True)
                log_pf = self.generator.compute_log_pf(generator_ret_dict, input_dict)
                log_pb = input_dict['batch_pb_log'].to(log_pf.device)

                if prob_estimation_method == 'IMPORTANCE_SAMPLING':
                    # estimate state probability with importance sampling
                    # NOTE: this estimates  log (P * # trajs)
                    state_gfn_logp = torch.logsumexp(log_pf - log_pb, dim=0)
                else:
                    state_gfn_logp = torch.logsumexp(log_pf, dim=0)

                state_idx = input_dict['state_idx']
                state_to_log_reward[state_idx] = input_dict['log_reward']
                state_to_logp[state_idx].append(state_gfn_logp)

                if bar is not None:
                    bar.update(input_dict['batch_size'])

        states_log_rewards, states_gfn_logp = [], []
        for state_idx in state_to_log_reward.keys():
            states_log_rewards.append(state_to_log_reward[state_idx])
            states_gfn_logp.append(torch.logsumexp(torch.stack(state_to_logp[state_idx]), dim=0).item() -
                                   np.log(eval_loader.nb_rounds * eval_loader.batch_size))

        return states_gfn_logp, states_log_rewards

    def evaluate_gfn_quality(self, scale=None):
        states_gfn_logp, states_log_rewards = self.estimate_gfn_states_sampling_probs(self.states, scale)
        eval_ret = {
            'log_prob_reward': [states_gfn_logp, states_log_rewards],
            'log_pearsonr': pearsonr(states_gfn_logp, states_log_rewards)[0],
        }
        return eval_ret

    def evaluate_marginal_likelihood_z(self, trajs=None, num_trajs=128, scale=None):
        """
        evaluate marginal likelihoood with estimated Z
        :return:
        """
        if trajs is None:
            if self.condition_on_scale:
                scales = [scale] * num_trajs
            else:
                scales = None
            trajs = self.rollout_worker.rollout(
                self.generator, num_trajs, random_action_prob=0.0, scales=scales)

        trajs_batches = chunks(trajs, self.evaluation_cfg.BATCH_SIZE)
        log_pfs, log_pbs, log_rs = [], [], []
        bin_size = self.env.edge_env.categorical_bin_size
        bar = None
        if self.verbose:
            bar = tqdm(total=len(trajs), leave=True, position=0, desc='Total trajectories MLL estimation progress')
        for batch in trajs_batches:
            trees = [x.current_state.subtrees[0] for x in batch]
            trees = [t.to_rooted_tree(0, None, None, self.env, perturbation_size=bin_size / 2) for t in trees]
            log_r = torch.tensor([x.log_score for x in trees])
            with torch.no_grad():
                input_batch = process_trajectories_tb(self.env, self.generator.state2input, batch)
                input_batch['scale'] = torch.tensor(np.array([scale] * self.evaluation_cfg.BATCH_SIZE,
                                                             dtype=np.float32)).reshape(-1, 1)
                ret = self.generator(input_batch, compute_edge=True)
                log_pf = self.generator.compute_log_pf(ret, input_batch)
                log_pb = input_batch['batch_pb_log'].to(log_pf.device)
                log_r = log_r.to(log_pf.device)
                log_pfs.append(log_pf)
                log_pbs.append(log_pb)
                log_rs.append(log_r)
            if bar is not None:
                bar.update(len(batch))
        if bar is not None:
            bar.close()
        log_pfs = torch.cat(log_pfs, dim=0)
        log_pbs = torch.cat(log_pbs, dim=0)
        log_rs = torch.cat(log_rs, dim=0)
        if scale is not None:
            log_rs /= scale
        num_trees = len(self.env.seq_arrays)
        discrete_factor = (num_trees * 2 - 3) * np.log(bin_size)
        tree_factor = - np.sum(np.log(np.arange(3, 2 * num_trees - 3, 2)))
        marginal_likelihood = torch.logsumexp(log_rs + log_pbs - log_pfs, dim=0) - np.log(
            len(trajs)) + discrete_factor + tree_factor
        eval_ret = {'marginal_likelihood': marginal_likelihood.item()}
        return eval_ret

    def evaluate_marginal_likelihood_iwae(self, gfn_states, scale=None):
        if gfn_states is None:
            if self.condition_on_scale:
                scales = [scale] * self.evaluation_cfg.MUTATIONS_TRAJS
            else:
                scales = None
            trajs = self.rollout_worker.rollout(
                self.generator, self.evaluation_cfg.STATES_NUM, random_action_prob=0.0, scales=scales)
            gfn_states = [traj.current_state for traj in trajs]

        states_gfn_logp, states_log_rewards = self.estimate_gfn_states_sampling_probs(gfn_states, scale)
        num_trees = len(self.env.seq_arrays)
        bin_size = self.env.edge_env.categorical_bin_size
        discrete_factor = (num_trees * 2 - 3) * np.log(bin_size)
        tree_factor = - np.sum(np.log(np.arange(3, 2 * num_trees - 3, 2)))
        bounds = states_log_rewards + discrete_factor + tree_factor - states_gfn_logp
        marginal_likelihood = logsumexp(bounds) - np.log(len(gfn_states))
        eval_ret = {'marginal_likelihood': marginal_likelihood}

        return eval_ret

    # TODO REWORK
    def update_states_set(self, states):
        """
        update the states set with best seen states and some sampled states
        :param states
        :return:
        """

        distinct_scores_dict = {}
        distinct_states = {}
        for state in states:

            state_key = str(state.subtrees[0])
            if state_key not in distinct_states:
                if self.env.parsimony_problem:
                    score = state.subtrees[0].total_mutations
                else:
                    score = round(float(state.subtrees[0].log_score), 2)
                if score in distinct_scores_dict:
                    distinct_scores_dict[score].append(state)
                else:
                    distinct_scores_dict[score] = [state]
                distinct_states[state_key] = 1
            else:
                distinct_states[state_key] += 1

        distinct_mutations_set = list(distinct_scores_dict.keys())
        scores_set = random.choices(distinct_mutations_set, k=self.evaluation_cfg.STATES_NUM)
        new_states = []
        for m in scores_set:
            new_states.append(random.choices(distinct_scores_dict[m], k=1)[0])

        self.states = new_states

    def generate_initial_states(self):

        assert self.evaluation_cfg.STATES_GENERATION_METHOD in ['RANDOM', 'UNIQUE_MUTATIONS', 'UNIFORM_BINS']
        state_generation_method = self.evaluation_cfg.STATES_GENERATION_METHOD
        if not self.env.parsimony_problem:
            state_generation_method = 'RANDOM'

        if state_generation_method == 'RANDOM':
            states = [x.current_state for x in self.env.sample(self.evaluation_cfg.STATES_NUM)]
        elif state_generation_method == 'UNIQUE_MUTATIONS':
            states = self.generate_states_unique_mutations()
        else:
            states = self.generate_states_by_bins()
        return states

    def generate_states_unique_mutations(self):

        reference_trajectory = None
        if self.evaluation_cfg.SAME_TREE_STRUCTURE:
            reference_trajectory = self.env.sample(1)[0]

        all_states = []
        # to obtain a diverse sample of total mutations (aka rewards)
        mutations_counter = defaultdict(lambda: 0)

        nb_steps = 0
        while len(all_states) < self.evaluation_cfg.STATES_NUM:
            nb_steps += 1
            if self.evaluation_cfg.SAME_TREE_STRUCTURE:
                # constrain all sampled trees to have the same underlying topology
                traj = self.env.trajectory_permute_leaves(reference_trajectory)
            else:
                traj = self.env.sample(1)[0]
            traj_mutations = traj.current_state.subtrees[0].total_mutations
            if mutations_counter[traj_mutations] >= self.evaluation_cfg.MAX_DUPLICATE_MUTATIONS:
                if nb_steps <= 100000:
                    continue
                else:
                    raise ValueError(
                        'Sampling step exceeds 100000, cannot satisfy \'max_duplicate_mutations\' requirements')
            mutations_counter[traj_mutations] += 1
            all_states.append(traj.current_state)
        return all_states

    def generate_states_by_bins(self):
        """
        :return:
        """
        reference_trajectory = None
        bin_num = self.evaluation_cfg.BINS_NUM
        if self.evaluation_cfg.SAME_TREE_STRUCTURE:
            reference_trajectory = self.env.sample(1)[0]

        all_states = []
        trajs = self.env.sample(1000)
        mutations = [traj.current_state.subtrees[0].total_mutations for traj in trajs]
        min_mutations, max_mutations = min(mutations), max(mutations)
        bin_size = (max_mutations - min_mutations) / bin_num

        mutations_bins = {
            x: min(bin_num - 1, int((x - min_mutations) / bin_size)) for x in range(min_mutations, max_mutations + 1)
        }
        bin_counts = {x: 0 for x in range(bin_num)}
        states_per_bin = self.evaluation_cfg.STATES_NUM / bin_num
        while len(all_states) < self.evaluation_cfg.STATES_NUM:
            if self.evaluation_cfg.SAME_TREE_STRUCTURE:
                traj = self.env.trajectory_permute_leaves(reference_trajectory)
            else:
                traj = self.env.sample(1)[0]
            state = traj.current_state
            mutations = state.subtrees[0].total_mutations
            if mutations in mutations_bins:
                bin_idx = mutations_bins[mutations]
            elif mutations < min_mutations:
                bin_idx = 0
            else:
                bin_idx = bin_num - 1
            if bin_counts[bin_idx] < states_per_bin:
                all_states.append(state)
                bin_counts[bin_idx] += 1
        return all_states
