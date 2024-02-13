import torch
import torch.nn as nn
from src.model.mlp import MLP
import torch.nn.functional as F
from src.model.weight_init import trunc_normal_
from src.model.transformer import TransformerEncoder


class PhyloTreeModelOneStep(nn.Module):

    def __init__(self, gfn_cfg):
        super().__init__()

        transformer_cfg = gfn_cfg.MODEL.TRANSFORMER
        self.compute_state_flow = (gfn_cfg.LOSS_TYPE != 'TB')
        self.concatenate_summary_token = transformer_cfg.LOGITS_HEAD.CONCATENATE_SUMMARY_TOKEN
        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE
        self.encoder = TransformerEncoder(transformer_cfg)

        self.seq_emb = MLP(transformer_cfg.SEQ_EMB)
        embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        if self.condition_on_scale:
            self.scale_module = MLP(gfn_cfg.MODEL.SCALE_MLP)
        else:
            self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)
            trunc_normal_(self.summary_token, std=0.1)

        self.logits_head = MLP(transformer_cfg.LOGITS_HEAD)
        if self.compute_state_flow:
            self.flow_head = MLP(transformer_cfg.FLOW_HEAD)

    def get_head_token(self, scale):
        if self.condition_on_scale:
            token = self.scale_module(scale)
        else:
            token = self.summary_token
        return token

    def model_params(self):
        return list(self.parameters())

    def forward(self, **kwargs):
        """
        :param batch_input: input tensors of shape [batch_size, nb_seq, seq_len], each sample in the batch is a state
        :param batch_intermediate_flag: boolean to tell if a state is intermediate
        :param batch_nb_seq: list of actual sequence length for each sample the batch
        """
        batch_input = kwargs['batch_input']
        batch_nb_seq = kwargs['batch_nb_seq']
        scale = kwargs.get('scale')
        return_tree_reps = kwargs.get('return_tree_reps', False)

        batch_size, max_nb_seq, _ = batch_input.shape

        # batch_size, max_nb_seq, emb_size
        x = self.seq_emb(batch_input)

        # add summary token
        summary_token = self.get_head_token(scale)
        if self.condition_on_scale:
            traj_length = batch_size // kwargs['batch_size']
            summary_token = summary_token.unsqueeze(1).expand(-1, traj_length, -1).reshape(batch_size, 1, -1)
        else:
            summary_token = summary_token.expand(batch_size, -1, -1)
        x = torch.cat((summary_token, x), dim=1)

        # padding mask
        batch_padding_mask = torch.ones((batch_size, max_nb_seq)).to(x).cumsum(dim=1) > batch_nb_seq[:, None]
        batch_padding_mask = batch_padding_mask.bool()
        batch_padding_mask = F.pad(batch_padding_mask, (1, 0), "constant", False)

        x = self.encoder(x, batch_padding_mask)
        summary_token = x[:, :1]
        trees_reps = x[:, 1:]

        # add all pairs of embeddings
        #  x[i, j]  + x[i, k] = C[i, j, k]
        #  B x N x E  =>     B x N x N x E
        tmp = (trees_reps[:, :, None, :] + trees_reps[:, None, :, :])
        # get all distinct pairs
        row, col = torch.triu_indices(max_nb_seq, max_nb_seq, offset=1)
        x_pairs = tmp[:, row, col]

        if self.concatenate_summary_token:
            _, num_trees, _ = x_pairs.shape
            s = summary_token.expand(-1, num_trees, -1)
            x_pairs = torch.cat([x_pairs, s], dim=2)

        logits = self.logits_head(x_pairs).squeeze(-1)
        if self.compute_state_flow:
            log_state_flow = self.flow_head(summary_token).reshape(-1)
            ret = {
                'logits': logits,
                'log_flow': log_state_flow,
                'mask': batch_padding_mask  # batch_padding_mask is useless for one step model
            }
        else:
            ret = {
                'logits': logits,
                'mask': batch_padding_mask
            }
        if return_tree_reps:
            ret['summary_reps'] = summary_token[:, 0]
            ret['trees_reps'] = trees_reps
        return ret