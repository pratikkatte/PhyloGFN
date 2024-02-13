from src.env.binary_tree_env import build_two_steps_env
from src.env.binary_tree_env_one_step import build_one_step_env
from src.env.binary_tree_env_one_step_likelihood import build_one_step_likelihood_env


def build_env(cfg, all_seqs):
    assert cfg.ENV.ENVIRONMENT_TYPE in ['TWO_STEPS_BINARY_TREE', 'ONE_STEP_BINARY_TREE']
    if cfg.ENV.ENVIRONMENT_TYPE == 'TWO_STEPS_BINARY_TREE':
        env, state2input = build_two_steps_env(cfg, all_seqs)
    else:
        if cfg.PARSIMONY_PROBLEM:
            env, state2input = build_one_step_env(cfg, all_seqs)
        else:
            env, state2input = build_one_step_likelihood_env(cfg, all_seqs)
    return env, state2input
