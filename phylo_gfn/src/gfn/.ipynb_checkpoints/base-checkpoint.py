import torch
import torch.nn as nn


class GeneratorBase(nn.Module):
    def __init__(self, cfg, state2input, env):
        super().__init__()
        self.cfg = cfg
        self.state2input = state2input
        self.env = env

    def train_step(self, *args):
        raise NotImplementedError()

    def forward(self, *args):
        raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)
