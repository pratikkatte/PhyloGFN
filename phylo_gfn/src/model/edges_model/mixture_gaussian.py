import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, MixtureSameFamily
from src.model.mlp import MLP


class MixtureGaussianModel(nn.Module):

    def __init__(self, mixture_cfg):
        super().__init__()
        mlp_cfg = mixture_cfg.MLP
        self.model = MLP(mlp_cfg)
        self.nb_components = mixture_cfg.NB_COMPONENTS
        self.soft_plus = nn.Softplus()

    def forward(self, input):
        logits = self.model(input)
        mixture_logits = logits[:, :self.nb_components]
        mean = logits[:, self.nb_components: 2 * self.nb_components]
        var_logits = logits[:, 2 * self.nb_components: 3 * self.nb_components]
        var = self.soft_plus(var_logits) + 0.001
        dist = MixtureSameFamily(
            Categorical(logits=mixture_logits),
            Normal(mean, var),
        )
        ret = {
            'logits': logits,
            'dist': dist
        }
        return ret