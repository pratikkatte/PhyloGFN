import numpy as np
import torch
from torch.distributions import Categorical
from numpy.distutils.misc_util import is_sequence
from typing import Optional, List, Dict

from src.env.trajectory import Trajectory
from src.gfn.base import GeneratorBase


class RolloutWorker:

    def __init__(self, gfn_cfg, env, state2input):
        self.env = env
        self.cfg = gfn_cfg
        self.state2input = state2input
        self.scales_set = gfn_cfg.SCALES_SET
        self.env_type = env.type
        self.parsimony_problem = env.parsimony_problem

    def sample_trees_selection_action(self,
                                      generator_ret_dict: Dict,
                                      input_dict: Dict,
                                      list_random_action_prob: List[float]):
        """
        :param generator_ret_dict:  model output
        :param input_dict:  model input
        :param random_action_prob:
        :return:
        """
        logits = generator_ret_dict['logits']
        if self.env_type == 'TWO_STEPS_BINARY_TREE':
            mask = generator_ret_dict['mask']
            logits = logits.masked_fill(mask, float('-inf'))
        else:
            pairwise_mask = input_dict['pairwise_mask_tensor']
            logits = logits.masked_fill(pairwise_mask, float('-inf'))

        cat = Categorical(logits=logits)
        actions = cat.sample()
        if self.env_type == 'ONE_STEP_BINARY_TREE':
            action_mapping_tensor = input_dict['pairwise_action_tensor']
            actions = action_mapping_tensor[torch.arange(len(actions)), actions]

        # apply random tree actions; random edge actions is handled inside the edge_model separately
        for i in range(actions.shape[0]):
            if np.random.uniform(0, 1) < list_random_action_prob[i]:
                if self.env_type == 'ONE_STEP_BINARY_TREE':
                    pairwise_mask = input_dict['pairwise_mask_tensor']
                    nb_valid_actions = torch.sum(~pairwise_mask[i]).item()
                else:
                    mask = generator_ret_dict['mask']
                    nb_valid_actions = torch.sum(~mask[i]).item()
                rand_action = np.random.randint(0, nb_valid_actions)
                actions[i] = rand_action
        actions = actions.cpu().numpy()
        return actions

    def rollout(self,
                generator: GeneratorBase,
                episodes: int,
                random_action_prob: float | List[float] = 0.0,
                scales: Optional[List[float]] = None,
                fixed_shape_trajs: Optional[List[Trajectory | None]] = None):
        # scales and fixed_shape_trajs are always assumed to be lists, unless they are None
        if is_sequence(random_action_prob):
            list_random_action_prob = random_action_prob
        else:
            list_random_action_prob = [random_action_prob] * episodes

        # initialize # of trajectories
        trajectories = []
        for _ in range(episodes):
            trajectories.append(Trajectory(self.env.get_initial_state()))

        rollout_done = False
        step = 0
        while not rollout_done:
            print(step)
            current_states = [x.current_state for x in trajectories]
            with torch.no_grad():
                input_dict = self.state2input.states2inputs(current_states)
                if generator.condition_on_scale:
                    input_dict['scale'] = torch.tensor(np.array(scales, dtype=np.float32)).reshape(-1, 1)
                    input_dict['batch_size'] = episodes
                ret = generator(input_dict)
                tree_actions = self.sample_trees_selection_action(ret, input_dict, list_random_action_prob)

                # fix tree shapes for some of the trajs
                if fixed_shape_trajs is not None:
                    for i in range(episodes):
                        if fixed_shape_trajs[i] is not None:
                            if self.parsimony_problem:
                                trees_action = fixed_shape_trajs[i].actions[step]
                            else:
                                trees_action = fixed_shape_trajs[i].actions[step]['tree_action']
                            tree_actions[i] = trees_action

            if self.parsimony_problem:
                actions = tree_actions
            else:
                # get the additional edge actions
                edge_ret = generator.calculate_edge_data(
                    ret['summary_reps'], ret['trees_reps'], sample=True,
                    input_dict={'list_random_action_prob': np.array(list_random_action_prob),
                                'batch_nb_seq': input_dict['batch_nb_seq'],
                                'batch_action': tree_actions}
                )
                edge_actions = edge_ret['edge_actions'].cpu().numpy()
                actions = [{'tree_action': t_a, 'edge_action': e_a} for t_a, e_a in
                           list(zip(tree_actions, edge_actions))]

            for idx, a in enumerate(actions):
                trajectory = trajectories[idx]
                current_state = trajectory.current_state
                current_state, next_state, a, reward, done = self.env.transition(current_state, a)
                trajectory.update(next_state, a, reward, done)
                rollout_done = done
            step += 1

        return trajectories
