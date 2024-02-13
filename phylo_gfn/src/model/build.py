from src.model.phylo_model_two_steps import PhyloTreeModelTwoSteps
from src.model.phylo_model_one_step import PhyloTreeModelOneStep


def build_model(gfn_cfg, env_type):
    if env_type == 'TWO_STEPS_BINARY_TREE':
        generator = PhyloTreeModelTwoSteps(gfn_cfg)
    else:
        generator = PhyloTreeModelOneStep(gfn_cfg)
    return generator
