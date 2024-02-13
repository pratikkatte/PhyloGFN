"""
Run GFN training on sequences generation problem

Usage:
    run.py <cfg_path> <sequences_path> <output_path> [--nb_device=<device_num>] [--acc_gen_sampling] [--quiet]
        [--random_sampling_batch_size=<rs_bs>]
    run.py resume <model_path> <sequences_path> <output_path> [--nb_device=<device_num>] [--acc_gen_sampling] [--quiet]
        [--random_sampling_batch_size=<rs_bs>]

Options:
    <cfg_path>                              config path
    <sequences_path>                        sequence file path
    <output_path>                           output folder
    <model_path>                            directory of an earlier experiment to resume
    --nb_device=<device_num>                specify the number of cuda devices available for training [default: 1]
    --quiet                                 do not show progress information during training or evaluation
    --acc_gen_sampling                      accelerate generator sampling in the dataloader by placing its workers onto GPU device
    --random_sampling_batch_size=<rs_bs>    option to use a smaller batch size for random sampling [default: 10000]
    -h --help                               Show this screen
"""

import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import datetime
import pickle
from docopt import docopt
import shutil
import gzip
import time
from datetime import timedelta
from src.utils.logging import get_logger
from src.configs.defaults import get_cfg_defaults
from src.env.build import build_env
import src.env.binary_tree_env
import src.env.binary_tree_env_one_step
import src.env.binary_tree_env_one_step_likelihood
from src.gfn.gfn_evaluation import GFNEvaluator
from src.gfn.rollout_worker_phylo import RolloutWorker
from src.gfn.training_data_loader import TrainingDataLoader
from src.gfn.build import build_gfn
import src.utils.plot_utils as plot_utils
from src.utils.utils import load_sequences, linear_schedule, load_seqs_cfg_data
from src.utils.resources_setup import get_list_of_devices
from heapq import heappushpop, heappush
from collections import Counter


start + (end - start) * t / T
def calculate_epoch_linear_schedule(start_val, end_val, T, epoch):
    start_val_epoch = linear_schedule(start_val, end_val, T, epoch)
    end_val_epoch = linear_schedule(start_val, end_val, T, epoch + 1)
    return start_val_epoch, end_val_epoch

def train_gfn(cfg, dataloader, generator, logger, gfn_evaluator, output_path, verbose=False,
              epoch_to_start=0, random_sampling_batch_size=10000):
    training_cfg = cfg.GFN.TRAINING_DATA_LOADER
    epochs_num = training_cfg.EPOCHS_NUM
    steps_per_epoch = training_cfg.STEPS_PER_EPOCH
    batch_size = training_cfg.GFN_BATCH_SIZE + training_cfg.GFN_FIXED_SHAPE_BATCH_SIZE + \
                 training_cfg.BEST_STATE_BATCH_SIZE + training_cfg.RANDOM_BATCH_SIZE
    total_trees_per_epoch = steps_per_epoch * batch_size
    mini_num_splits = training_cfg.MINI_BATCH_SPLITS
    evaluation_freq = cfg.GFN.MODEL.EVALUATION.EVALUATION_FREQ
    dataloader_generator_update_freq = training_cfg.FREQ_UPDATE_MODEL_WEIGHTS
    generator_path = os.path.join(output_path, 'dataloader_generator.pt')
    best_trees_path = os.path.join(output_path, 'best_trees.pt')
    shared_arrays = dataloader.shared_arrays
    train_with_amp = cfg.AMP
    scaler_schedule_epoch_interval = max(30, epochs_num // 5)
    target_net_update_freq = cfg.GFN.MODEL.TARGET_NET_UPDATE_FREQ * mini_num_splits
    do_eps_annealing = training_cfg.EPS_ANNEALING
    eps_annealing_cfg = training_cfg.EPS_ANNEALING_DATA

    total_random_trajectories = gfn_evaluator.evaluation_cfg.MUTATIONS_TRAJS
    gfn_evaluator.evaluation_cfg.MUTATIONS_TRAJS = random_sampling_batch_size
    num_sampling_rounds = total_random_trajectories // random_sampling_batch_size + \
                          int(total_random_trajectories % random_sampling_batch_size > 0)

    condition_on_scale = cfg.GFN.CONDITION_ON_SCALE
    if condition_on_scale:
        full_scale_set = cfg.GFN.SCALES_SET  # the full scale set
        full_scale_set_tensor = torch.tensor(full_scale_set, dtype=torch.float32).reshape(-1, 1). \
            to(generator.all_device[0])
    else:
        full_scale_set = None
        full_scale_set_tensor = None

    for epoch_id in range(epoch_to_start, epochs_num):
        bar = None
        if verbose:
            bar = tqdm(total=total_trees_per_epoch, leave=True, position=0, desc=f'Epoch: {epoch_id + 1}')

        # train
        generator.train()
        mini_batch_counter = 0
        synced_best_trees, synced_best_trees_dict = None, None

        # build the data loader for the current epoch
        if do_eps_annealing:
            if not eps_annealing_cfg.RESTART:
                start_eps_epoch, end_eps_epoch = calculate_epoch_linear_schedule(
                    4, 2, 200, epoch_id)
            else:
                start_eps_epoch, end_eps_epoch = calculate_epoch_linear_schedule(
                    eps_annealing_cfg.START_EPS, eps_annealing_cfg.END_EPS, scaler_schedule_epoch_interval,
                    epoch_id % scaler_schedule_epoch_interval)
        else:
            start_eps_epoch, end_eps_epoch = None, None
        if condition_on_scale:
            start_scales_sampling_mu, end_scales_sampling_mu = calculate_epoch_linear_schedule(
                max(full_scale_set), min(full_scale_set), epochs_num, epoch_id
            )
        else:
            start_scales_sampling_mu, end_scales_sampling_mu = None, None
            
        dataloader_iter = dataloader.build_data_loader(
            start_eps_epoch, end_eps_epoch, start_scales_sampling_mu, end_scales_sampling_mu)
        start_time = time.time()

        # iterate through batches
        for t, batch in enumerate(dataloader_iter):
            if 'best_seen_trees' in batch:
                synced_best_trees, synced_best_trees_dict = \
                    synchronize_replay_buffer(synced_best_trees, synced_best_trees_dict, batch['best_seen_trees'],
                                              10 * cfg.GFN.TRAINING_DATA_LOADER.BEST_TREES_BUFFER_SIZE)

            # select the appropriate shared array and distribute it to the worker process
            if shared_arrays is None:
                process_shared_array = None
            else:
                process_shared_array = shared_arrays[batch['process_idx']]
            if train_with_amp:
                generator.accumulate_loss_amp(batch, mini_num_splits, process_shared_array)
            else:
                generator.accumulate_loss(batch, mini_num_splits, process_shared_array)

            mini_batch_counter += 1
            if mini_batch_counter % mini_num_splits == 0:
                if train_with_amp:
                    ret = generator.update_model_amp()
                else:
                    ret = generator.update_model()
                status_str = 'Epoch {}, '.format(epoch_id + 1)
                
                for key, value in ret.items():
                    logger.add_scalar(key, value)
                    status_str += f'{key}: {value:.4f}, '
                    if key == 'loss':
                        plot_utils.plot('log_loss', np.log(value))
                    else:
                        plot_utils.plot(key, value)

                if bar is not None:
                    bar.update(batch_size)
                    bar.set_description(status_str)
                plot_utils.tick(index=0)  # index=0 is for the iterations
                mini_batch_counter = 0

            if cfg.GFN.MODEL.USE_TARGET_NET and t % target_net_update_freq == target_net_update_freq - 1:
                generator.update_target_net()

            # evaluate the learnt partition across temperatures
            if t % 50 == 0:
                with torch.no_grad():
                    log_z = generator.log_Z(full_scale_set_tensor)
                    if condition_on_scale:
                        log_z = log_z.reshape(-1)
                        for idx, scale in enumerate(full_scale_set):
                            logger.add_scalar('log_partition_{}'.format(scale), log_z[idx])
                            plot_utils.plot('log_partition_{}'.format(scale), log_z[idx])
                    else:
                        if cfg.GFN.LOSS_TYPE in ('FLDB', 'FLSUBTB'):
                            log_z = log_z[0]
                        logger.add_scalar('log_partition', log_z)
                        plot_utils.plot('log_partition', log_z)
                logger.add_scalar('eps', batch['eps'])
                plot_utils.plot('eps', batch['eps'])
                if 'scales_sampling_mu' in batch:
                    logger.add_scalar('scales_sampling_mu', batch['scales_sampling_mu'])
                    plot_utils.plot('scales_sampling_mu', batch['scales_sampling_mu'])

            # save the generator weights frequently to update the generator in dataloader
            if t % dataloader_generator_update_freq == 0:
                generator.save(generator_path)

        # deal with edge case when the last minibatch is not handled
        if generator.loss != 0:
            if train_with_amp:
                generator.update_model_amp()
            else:
                generator.update_model()

        del dataloader_iter
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        elapsed = time.time() - start_time
        print(f'epoch: {epoch_id + 1}', str(timedelta(seconds=elapsed)))

        # consolidate best trees from all worker processes
        if synced_best_trees is not None:
            pickle.dump(synced_best_trees, open(best_trees_path, 'wb'))

        # evaluation
        if epoch_id % evaluation_freq == evaluation_freq - 1:
            generator.eval()
            # pearson correlation evaluation
            eval_scale = min(full_scale_set) if condition_on_scale else None
            evaluation_result = gfn_evaluator.evaluate_gfn_quality(eval_scale)
            # random samples from the GFN
            # we split gfn_sample_result to several rounds because in large dataset the GPU memory may not be enough
            if env.parsimony_problem:
                gfn_sample_result = {'states': [], 'mutations': []}
                for _ in range(num_sampling_rounds):
                    gfn_sample_result_ = gfn_evaluator.evaluate_gfn_samples(eval_scale)
                    gfn_sample_result['states'].extend(gfn_sample_result_['states'])
                    gfn_sample_result['mutations'].extend(gfn_sample_result_['mutations'])
                gfn_sample_result.update({
                    'mut_mean': np.mean(gfn_sample_result['mutations']),
                    'mut_std': np.std(gfn_sample_result['mutations']),
                    'mut_min': np.min(gfn_sample_result['mutations']),
                    'mut_max': np.max(gfn_sample_result['mutations']),
                })
                evaluation_result['gfn_sample_result'] = gfn_sample_result

                print('sampled reward range: [%d —> %d]' % (gfn_sample_result['mut_min'], gfn_sample_result['mut_max']))
                leafsets_signature = [state.subtrees[0].leafsets_signature for state in gfn_sample_result['states']]
                _, idx = np.unique(leafsets_signature, return_index=True)
                print('number of unique trees by mutations',
                      sorted(Counter(np.array(gfn_sample_result['mutations'])[idx]).items()))
            else:
                gfn_sample_result = {'states': [], 'log_scores': [], 'trajs': []}
                for _ in range(num_sampling_rounds):
                    gfn_sample_result_ = gfn_evaluator.evaluate_gfn_samples(eval_scale)
                    gfn_sample_result['states'].extend(gfn_sample_result_['states'])
                    gfn_sample_result['log_scores'].extend(gfn_sample_result_['log_scores'])
                    gfn_sample_result['trajs'].extend(gfn_sample_result_['trajs'])
                gfn_sample_result.update({
                    'log_scores_mean': np.mean(gfn_sample_result['log_scores']),
                    'log_scores_std': np.std(gfn_sample_result['log_scores']),
                    'log_scores_min': np.min(gfn_sample_result['log_scores']),
                    'log_scores_max': np.max(gfn_sample_result['log_scores']),
                })
                evaluation_result['gfn_sample_result'] = gfn_sample_result

                print('sampled reward range: [%d —> %d]' % (
                    gfn_sample_result['log_scores_min'], gfn_sample_result['log_scores_max']))
                leafsets_signature = [state.subtrees[0].leafsets_signature for state in gfn_sample_result['states']]
                _, idx = np.unique(leafsets_signature, return_index=True)
                print('number of unique trees by log_scores',
                      sorted(Counter(np.array(gfn_sample_result['log_scores'])[idx]).items()))

                # # for likelihood problem, also estimate the marginal likelihood
                # gfn_states = np.random.choice(gfn_sample_result['states'], cfg.GFN.MODEL.EVALUATION.STATES_NUM,
                #                               replace=False)
                # evaluation_result.update(gfn_evaluator.evaluate_gfn_marginal_likelihood(gfn_states, eval_scale))
                evaluation_result.update(gfn_evaluator.evaluate_marginal_likelihood_z(
                    gfn_sample_result['trajs'], scale=eval_scale))

            # visualization of evaluation results
            plot_evaluation_results(logger, plot_utils, evaluation_result, output_path, epoch_id)

            # update states set
            states = gfn_evaluator.states + gfn_sample_result['states']
            gfn_evaluator.update_states_set(states)
            pickle.dump(gfn_evaluator.states, open(os.path.join(cfg.OUTPUT_PATH, 'eval_states.pkl'), 'wb'))

            if not condition_on_scale:
                # decide if we should reduce the scale to increase the probability of sampling trees with high reward
                log_z = generator.log_Z()
                if env.parsimony_problem:
                    log_reward = (cfg.ENV.REWARD.C - gfn_sample_result['mut_min']) / cfg.ENV.REWARD.SCALE
                else:
                    log_reward = (cfg.ENV.REWARD.C + gfn_sample_result['log_scores_max']) / cfg.ENV.REWARD.SCALE
                if log_reward > log_z:
                    # it means the model has not seen enough high value states
                    print('Warning: log partition %.3f less than the maximal log reward %.3f' % (log_z, log_reward))
                optimal_tree_prob = np.exp(log_reward - log_z)
                if epoch_id > 0 and epoch_id % scaler_schedule_epoch_interval == scaler_schedule_epoch_interval - 1:
                    # the pearson corr condition is no longer observed: evaluation_result['log_pearsonr'] > 0.9
                    # for the parsimony problem, the scale (aka statistical temperature) is annealed based on optimal_tree_prob
                    # for the likelihood problem, the scale will always be brought down to 1
                    if (env.parsimony_problem and optimal_tree_prob < 1e-4 and cfg.ENV.REWARD.SCALE > 0.25) or \
                            (not env.parsimony_problem and cfg.ENV.REWARD.SCALE > 1.):
                        cfg.ENV.REWARD.SCALE /= 2.
                        print('condition is met:', optimal_tree_prob)
                        print('updating cfg.ENV.REWARD.SCALE to:', cfg.ENV.REWARD.SCALE)
                        # update reward_fn
                        if env.parsimony_problem:
                            if cfg.ENV.ENVIRONMENT_TYPE == 'TWO_STEPS_BINARY_TREE':
                                env.reward_fn = src.env.binary_tree_env.PhyloTreeReward(cfg.ENV.REWARD)
                                env.fl_c = cfg.ENV.REWARD.C / cfg.ENV.REWARD.SCALE / (2 * len(env.sequences) - 4)
                            else:
                                env.reward_fn = src.env.binary_tree_env_one_step.PhyloTreeReward(cfg.ENV.REWARD)
                                env.fl_c = cfg.ENV.REWARD.C / cfg.ENV.REWARD.SCALE / (len(env.sequences) - 1)
                        else:
                            env.reward_fn = src.env.binary_tree_env_one_step_likelihood.PhyloTreeReward(cfg.ENV.REWARD)
                            # todo, fl_c for likelihood model

            print('#' * 100)

        cfg.dump(stream=open(os.path.join(cfg.OUTPUT_PATH, 'config.yaml'), 'w'))
        plot_utils.tick(index=1)  # index=1 is for the epochs
        plot_utils.flush()

        # save current model
        save_path = os.path.join(output_path, 'checkpoints', "checkpoint_%06d.pt" % (epoch_id,))
        generator.save(save_path)
        logger.save()
        plot_utils.save()
        if bar is not None:
            bar.close()


def synchronize_replay_buffer(synced_best_trees, synced_best_trees_dict, worker_best_trees, max_buffer_size):
    # consolidate best states from all worker processes
    if synced_best_trees is None:
        synced_best_trees = []
        synced_best_trees_dict = {}
    for tree in worker_best_trees:
        leafsets_signature = tree.leafsets_signature
        if leafsets_signature not in synced_best_trees_dict:
            synced_best_trees_dict[leafsets_signature] = None
            if len(synced_best_trees) >= max_buffer_size:
                dropped_tree = heappushpop(synced_best_trees, tree)
                del synced_best_trees_dict[dropped_tree.leafsets_signature]
            else:
                heappush(synced_best_trees, tree)
    return synced_best_trees, synced_best_trees_dict


def plot_evaluation_results(logger, plot_utils, evaluation_result, output_path, epoch_id):
    logger.add_scalar('gfn_quality_log_pearsonr', evaluation_result['log_pearsonr'])
    plot_utils.plot('gfn_quality_log_pearsonr', evaluation_result['log_pearsonr'], index=1)
    if 'marginal_likelihood' in evaluation_result:
        logger.add_scalar('marginal_likelihood', evaluation_result['marginal_likelihood'])
        plot_utils.plot('gfn_quality_marginal_likelihood', evaluation_result['marginal_likelihood'], index=1)
        print('marginal likelihood:', evaluation_result['marginal_likelihood'])

    # scatter plots for model probabilities and rewards
    eval_scatter_path = os.path.join(output_path, f'eval_scatter_{epoch_id:06d}')
    if not os.path.exists(eval_scatter_path):
        os.makedirs(eval_scatter_path)
    plot_utils.plot_scatter(evaluation_result['log_prob_reward'][0], evaluation_result['log_prob_reward'][1],
                            'model logp', 'log reward', os.path.join(eval_scatter_path, 'eval_log_prob_reward.png'))
    pickle.dump(evaluation_result['log_prob_reward'],
                open(os.path.join(eval_scatter_path, 'eval_log_prob_reward.pkl'), 'wb'))

    gfn_sample_result = evaluation_result['gfn_sample_result']
    if env.parsimony_problem:
        logger.add_scalar('gfn_sampled_mutations_mean', gfn_sample_result['mut_mean'])
        logger.add_scalar('gfn_sampled_mutations_std', gfn_sample_result['mut_std'])
        logger.add_scalar('gfn_sampled_mutations_min', gfn_sample_result['mut_min'])
        logger.add_scalar('gfn_sampled_mutations_max', gfn_sample_result['mut_max'])
        plot_utils.plot('gfn_sampled_mutations_mean', gfn_sample_result['mut_mean'], index=1)
        plot_utils.plot('gfn_sampled_mutations_mean_std', (gfn_sample_result['mut_mean'],
                                                           gfn_sample_result['mut_std']), index=1)
        plot_utils.plot('gfn_sampled_mutations_mean_min_max', (gfn_sample_result['mut_mean'],
                                                               gfn_sample_result['mut_min'],
                                                               gfn_sample_result['mut_max']), index=1)
    else:
        logger.add_scalar('gfn_sampled_log_scores_mean', gfn_sample_result['log_scores_mean'])
        logger.add_scalar('gfn_sampled_log_scores_std', gfn_sample_result['log_scores_std'])
        logger.add_scalar('gfn_sampled_log_scores_min', gfn_sample_result['log_scores_min'])
        logger.add_scalar('gfn_sampled_log_scores_max', gfn_sample_result['log_scores_max'])
        plot_utils.plot('gfn_sampled_log_scores_mean', gfn_sample_result['log_scores_mean'], index=1)
        plot_utils.plot('gfn_sampled_log_scores_mean_std', (gfn_sample_result['log_scores_mean'],
                                                            gfn_sample_result['log_scores_std']), index=1)
        plot_utils.plot('gfn_sampled_log_scores_mean_min_max', (gfn_sample_result['log_scores_mean'],
                                                                gfn_sample_result['log_scores_min'],
                                                                gfn_sample_result['log_scores_max']), index=1)
    # some pytorch version seems to have "No loop matching the specified signature
    # and casting was found for ufunc greater" error.
    # note pytorch 2.0.1 has the issue of sampling zero probability entries in Categorical/multinomial
    # Disabling the following line on my side just temporarily
    # logger.draw_histogram('sampled states mutations', np.array(samples_mutation_result['mutations']), epoch_id)


if __name__ == '__main__':

    # parse args
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    verbose = not arguments['--quiet']
    nb_device = int(arguments['--nb_device'])

    # load sequences
    sequences_path = arguments['<sequences_path>']
    all_seqs = load_sequences(sequences_path)

    if arguments['resume']:
        resume_model_path = arguments['<model_path>']
        cfg_path = os.path.join(resume_model_path, 'config.yaml')
    else:
        resume_model_path = None
        cfg_path = arguments['<cfg_path>']
    output_path = arguments['<output_path>']

    # load cfg
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = load_seqs_cfg_data(all_seqs, cfg)

    # setup devices
    enable_ags = arguments['--acc_gen_sampling']
    dataloader_device, generator_devices = get_list_of_devices(nb_device, enable_ags)
    torch.multiprocessing.set_start_method('spawn')  # to allow dataloader workers to access cuda devices

    # check scale setup
    if cfg.GFN.CONDITION_ON_SCALE and cfg.ENV.REWARD.SCALE != 1.:
        print(f'Warning. conditioning on scale but the base scale is not equal to 1 ({cfg.ENV.REWARD.SCALE})')

    # automatically add formatted datetime to output_path
    assert output_path != ''
    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_ = output_path.split(os.path.sep)
    output_path_[-1] = cur_time + '_' + output_path_[-1]
    output_path = os.path.sep.join(output_path_)
    cfg.OUTPUT_PATH = output_path

    # create folders
    checkpoints_path = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    # backup major files for future reference
    backup_dir = os.path.join(output_path, 'backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # copy the full src directory instead of few files
    shutil.copy(__file__, backup_dir)
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    shutil.copytree(os.path.join(current_dir_path, 'src'), os.path.join(backup_dir, 'src'))

    if arguments['--quiet']:
        # directly write logs to the experiment directory
        outfile = open(os.path.join(output_path, '%s.out' % (output_path.split('/')[-1])), "w")
        sys.stdout = outfile
        sys.stderr = outfile

    cfg.dump(stream=open(os.path.join(output_path, 'config.yaml'), 'w'))

    # build logger
    logger = get_logger(cfg)

    # build up environment and GFN
    env, state2input = build_env(cfg, all_seqs)
    rollout_worker = RolloutWorker(cfg.GFN, env, state2input)
    generator = build_gfn(cfg, state2input, env, generator_devices)

    cfg_eval = cfg.GFN.MODEL.EVALUATION
    eval_states = None
    if resume_model_path is not None:
        # load weight to generator
        checkpoints_folder = os.path.join(resume_model_path, 'checkpoints')
        checkpoints_paths = [os.path.join(checkpoints_folder, file) for file in os.listdir(checkpoints_folder) if
                             file.endswith('.pt')]
        checkpoints_paths = sorted(checkpoints_paths)
        latest_checkpoint_path = checkpoints_paths[-1]
        if verbose:
            print(f'loading checkpoint: {latest_checkpoint_path}')
        generator.load(latest_checkpoint_path)
        epoch_to_start = int(latest_checkpoint_path.split('_')[-1].split('.')[0]) + 1
        if cfg.GFN.MODEL.USE_TARGET_NET:
            generator.update_target_net()

        # load plotting related data
        plot_utils.load(os.path.join(resume_model_path, 'plot_utils_save.pkl'))

        # load logger data
        logger.data = pickle.load(gzip.open(os.path.join(resume_model_path, 'logs')))

        # load evaluation states
        eval_states = pickle.load(open(os.path.join(resume_model_path, 'eval_states.pkl'), 'rb'))

        # best states data
        if os.path.exists(os.path.join(resume_model_path, 'best_trees.pt')):
            shutil.copy(os.path.join(resume_model_path, 'best_trees.pt'), os.path.join(output_path, 'best_trees.pt'))
    else:
        epoch_to_start = 0
        # initialize plotting setups
        plot_utils._enlarge_ticker(1)
        plot_utils.set_xlabel_for_tick(1, 'epochs')
    plot_utils.set_output_dir(cfg.OUTPUT_PATH)
    plot_utils.suppress_stdout()

    # if cfg.GFN.LOSS_TYPE == 'FLSUBTB' and cfg.ENV == 'TWO_STEPS_BINARY_TREE':
    #     raise ValueError('two step env is not compatible with FLSUBTB')

    # build dataloader
    generator_path = os.path.join(output_path, 'dataloader_generator.pt')
    best_trees_path = os.path.join(output_path, 'best_trees.pt')
    generator.save(generator_path)
    dataloader = TrainingDataLoader(cfg, generator_path, best_trees_path, dataloader_device, all_seqs)

    if cfg.GFN.CONDITION_ON_SCALE:
        scales_set = cfg.GFN.SCALES_SET
        assert scales_set is not None, 'must specify "SCALES_SET" in config when "CONDITION_ON_SCALE" is on'
    else:
        scales_set = None

    gfn_evaluator = GFNEvaluator(cfg_eval, rollout_worker, generator, states=eval_states, verbose=verbose,
                                 scales_set=scales_set)

    # train
    train_gfn(cfg, dataloader, generator, logger, gfn_evaluator, output_path, verbose, epoch_to_start,
              int(arguments['--random_sampling_batch_size']))
