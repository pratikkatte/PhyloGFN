from src.gfn.tb_gfn_phylo import TBGFlowNetGenerator
from src.gfn.flsubtb_gfn_phylo import FLSubTBGFlowNetGenerator


def build_gfn(cfg, state2input, env, generator_devices):
    assert cfg.GFN.LOSS_TYPE in ['TB', 'DB', 'FLDB', 'SUBTB', 'FLSUBTB']
    if cfg.GFN.LOSS_TYPE == 'TB':
        generator = TBGFlowNetGenerator(cfg.GFN, state2input, env, generator_devices)
    elif cfg.GFN.LOSS_TYPE == 'FLSUBTB':
        generator = FLSubTBGFlowNetGenerator(cfg.GFN, state2input, env, generator_devices)
    else:
        raise NotImplemented('coming soon to this branch!')
    return generator
