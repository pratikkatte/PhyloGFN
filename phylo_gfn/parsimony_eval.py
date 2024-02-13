"""
Usage:
    parsimony_eval.py <model_path> <sequences_path> [--scale=<scale_val>] [--quiet]

Options:
    <sequences_path>                        sequence file path
    <model_path>                            directory of an earlier experiment to resume
    --quiet                                 do not show progress information during training or evaluation
    --scale=<scale_val>                     temperature, if the model being loaded is temperature-conditioned [default: None]
    -h --help                               Show this screen
"""

import os
import pickle
from docopt import docopt
from src.configs.defaults import get_cfg_defaults
from src.env.build import build_env
from src.gfn.gfn_evaluation import GFNEvaluator
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.build import build_gfn
from src.utils.utils import load_sequences, load_seqs_cfg_data
from src.env.binary_tree_env_one_step import PhylogenticTreeState

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
from src.utils.utils import chunks, compute_trajectory_prob
from src.env.binary_tree_env import PhyloTree


def build_tree(tree_data, sequences_data):
    if len(tree_data) == 1:
        tree = PhyloTree(None, None, sequences_data[tree_data[0]])
        return tree
    left_tree = build_tree(tree_data[0], sequences_data)
    right_tree = build_tree(tree_data[1], sequences_data)
    tree = PhyloTree(left_tree, right_tree)
    return tree


def fix_tree_data(tree_data):
    if len(tree_data) == 3:
        return fix_tree_data([[tree_data[0], tree_data[1]], tree_data[2]])
    l = tree_data[0]
    r = tree_data[1]
    if type(l) == int:
        l = [l - 1]
    else:
        l = fix_tree_data(l)

    if type(r) == int:
        r = [r - 1]
    else:
        r = fix_tree_data(r)

    return [l, r]


def probability_estimation_is(env, generator, state, num_trajs=100):
    trajs = []
    trajs_pb = []

    for _ in range(num_trajs):
        traj = env.state_to_trajectory(state)
        trajs.append(traj)
        trajs_pb.append(np.prod(1 / np.array(env.get_number_parents_trajectory(traj))))

    log_pfs = []
    for batch in chunks(trajs, 10):
        log_pfs.append(compute_trajectory_prob(generator, batch, True))

    log_pfs = torch.cat(log_pfs)
    log_pbs = torch.log(torch.tensor(np.array(trajs_pb))).to(log_pfs)

    state_gfn_logp = torch.logsumexp(log_pfs - log_pbs, dim=0)

    return state_gfn_logp - np.log(num_trajs)


def load_paup_best_trees(tree_file, env, all_seqs):
    lines = open(tree_file).readlines()
    lines = [x for x in lines if x[:4] == 'tree']
    unrooted_tree_tuples = [eval(x.split('[&U] ')[1].split(';')[0]) for x in lines]
    unrooted_trees = []
    for tree in unrooted_tree_tuples:
        tree_shape = fix_tree_data(tree)
        sequences_data = [[env.seq2array(seq), [idx]] for idx, seq in enumerate(all_seqs)]
        best_tree = build_tree(tree_shape, sequences_data)
        best_tree_unrooted = best_tree.to_unrooted_tree()
        unrooted_trees.append(best_tree_unrooted)

    return unrooted_trees


if __name__ == '__main__':

    # parse args
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    verbose = not arguments['--quiet']
    scale = eval(arguments['--scale'])

    # load sequences
    sequences_path = arguments['<sequences_path>']
    all_seqs = load_sequences(sequences_path)

    resume_model_path = arguments['<model_path>']
    cfg_path = os.path.join(resume_model_path, 'config.yaml')

    # load cfg
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = load_seqs_cfg_data(all_seqs, cfg)

    # setup devices
    torch.multiprocessing.set_start_method('spawn')  # to allow dataloader workers to access cuda devices

    # check scale setup
    if cfg.GFN.CONDITION_ON_SCALE and cfg.ENV.REWARD.SCALE != 1.:
        print(f'Warning. conditioning on scale but the base scale is not equal to 1 ({cfg.ENV.REWARD.SCALE})')

    # build up environment and GFN
    env, state2input = build_env(cfg, all_seqs)
    rollout_worker = RolloutWorker(cfg.GFN, env, state2input)
    generator = build_gfn(cfg, state2input, env, torch.device('cuda:0'))

    cfg_eval = cfg.GFN.MODEL.EVALUATION

    checkpoints_folder = os.path.join(resume_model_path, 'checkpoints')
    checkpoints_paths = [os.path.join(checkpoints_folder, file) for file in os.listdir(checkpoints_folder) if
                         file.endswith('.pt')]
    checkpoints_paths = sorted(checkpoints_paths)
    latest_checkpoint_path = checkpoints_paths[-1]
    print(f'loading checkpoint: {latest_checkpoint_path}')
    generator.load(latest_checkpoint_path)

    if cfg.GFN.CONDITION_ON_SCALE:
        scales_set = cfg.GFN.SCALES_SET
        assert scales_set is not None, 'must specify "SCALES_SET" in config when "CONDITION_ON_SCALE" is on'
        assert scale is not None, ('must specify a "scale" when the model is temperature-conditioned; '
                                   'this is for calculating the model sampling log-prob under that scale')
        scale = float(scale)
    else:
        scales_set = None
        scale = None

    cfg_eval.BATCH_SIZE = 8
    gfn_evaluator = GFNEvaluator(cfg_eval, rollout_worker, generator, states='none', verbose=verbose,
                                 scales_set=scales_set)

    # sampling 10,000 trees from the generator
    gfn_sample_result = {'states': [], 'mutations': [], 'unrooted_trees': []}
    for _ in range(10):
        gfn_sample_result_ = gfn_evaluator.evaluate_gfn_samples(scale)
        gfn_sample_result['states'].extend(gfn_sample_result_['states'])
        gfn_sample_result['mutations'].extend(gfn_sample_result_['mutations'])
        gfn_sample_result['unrooted_trees'].extend([
            state.subtrees[0].to_unrooted_tree() for state in gfn_sample_result_['states']])

    _, unique_indices = np.unique([tree.leafsets_signature for tree in gfn_sample_result['unrooted_trees']],
                                  return_index=True)
    unique_trees = np.array(gfn_sample_result['unrooted_trees'])[unique_indices]
    unique_mutations = np.array(gfn_sample_result['mutations'])[unique_indices]

    dataset_id = sequences_path.split('.pickle')[0][-1]
    paup_result_path = 'dataset/benchmark_datasets/paup_results/ds{}.tre'.format(dataset_id)
    if os.path.exists(paup_result_path):
        print('loading paup results:', paup_result_path)
        best_unrooted_trees = load_paup_best_trees(paup_result_path, env, all_seqs)
    else:
        print('path does not exist:', paup_result_path)
        optimal_mutation = np.sort(np.unique(unique_mutations))[0]
        best_unrooted_trees = unique_trees[unique_mutations == optimal_mutation][:5]

    # optimal trees identified by the paup*
    results = []
    for best_unrooted_tree in np.array(best_unrooted_trees):
        data = {
            'unrooted_tree': best_unrooted_tree,
            'rooted_tree_results': []
        }
        all_state = []
        for idx in range(2 * len(all_seqs) - 3):
            t = best_unrooted_tree.to_rooted_tree(idx)
            # NOTE, one step env
            state = PhylogenticTreeState([t])
            all_state.append(state)

        # NOTE, update the evaluator object with rooted versions of the target unrooted tree
        gfn_evaluator.states = all_state

        ret = gfn_evaluator.evaluate_gfn_quality(scale=scale)
        all_logp = ret['log_prob_reward'][0]
        for idx in range(2 * len(all_seqs) - 3):
            data['rooted_tree_results'].append({
                'rooted_tree': all_state[idx].subtrees[0],
                'logp': all_logp[idx]  # averaging is already done
            })
        results.append(data)

    pickle.dump(results, open(os.path.join(resume_model_path, 'best_trees_results.pkl'), 'wb'))

    plt.figure(figsize=(10, 5))
    plt.boxplot([[y['logp'] for y in x['rooted_tree_results']] for x in results])
    plt.xticks(list(range(1, len(results) + 1)),
               [x['rooted_tree_results'][0]['rooted_tree'].total_mutations for x in results])
    plt.savefig(os.path.join(resume_model_path, 'best_trees_results.png'), dpi=350)

    # identifying suboptimal trees
    # index 0 would be the optimal trees
    suboptimal_mutations = np.sort(np.unique(unique_mutations))[1:5]
    all_suboptimal_trees = []
    for mut in suboptimal_mutations:
        suboptimal_trees = unique_trees[unique_mutations == mut][:5]
        all_suboptimal_trees.extend(suboptimal_trees)

    # computing logp for the suboptimal trees
    results_suboptimal = []
    for unrooted_tree in all_suboptimal_trees:
        data = {
            'unrooted_tree': unrooted_tree,
            'rooted_tree_results': []
        }
        all_state = []
        for idx in range(2 * len(all_seqs) - 3):
            t = unrooted_tree.to_rooted_tree(idx)
            # NOTE, one step env
            state = PhylogenticTreeState([t])
            all_state.append(state)

        # NOTE, update the evaluator object with rooted versions of the target unrooted tree
        gfn_evaluator.states = all_state

        ret = gfn_evaluator.evaluate_gfn_quality(scale=scale)
        all_logp = ret['log_prob_reward'][0]
        for idx in range(2 * len(all_seqs) - 3):
            data['rooted_tree_results'].append({
                'rooted_tree': all_state[idx].subtrees[0],
                'logp': all_logp[idx]
            })
        results_suboptimal.append(data)
        pickle.dump([gfn_sample_result, results_suboptimal],
                    open(os.path.join(resume_model_path, 'suboptimal_trees_results.pkl'), 'wb'))

        plt.figure(figsize=(15, 7))
        all_ret = results + results_suboptimal
        plt.boxplot([[y['logp'] for y in x['rooted_tree_results']] for x in all_ret])
        plt.xticks(list(range(1, len(all_ret) + 1)),
                   [x['rooted_tree_results'][0]['rooted_tree'].total_mutations for x in all_ret])
        plt.savefig(os.path.join(resume_model_path, 'suboptimal_trees_results.png'), dpi=350)
