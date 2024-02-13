from src.model.edges_model.categorical.categorical import EdgesModelCategorical


def build_edge_model(gfn_cfg):
    edges_cfg = gfn_cfg.MODEL.EDGES_MODELING
    dist = edges_cfg.DISTRIBUTION
    assert dist in ['CATEGORICAL']
    edge_model = EdgesModelCategorical(edges_cfg.CATEGORICAL)
    return edge_model
