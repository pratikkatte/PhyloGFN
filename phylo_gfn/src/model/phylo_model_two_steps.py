import torch
from torch import nn
from src.model.mlp import MLP
from src.model.transformer import TransformerEncoder
from torch.nn import functional as F
from src.model.weight_init import trunc_normal_


class PhyloTreeModelTwoSteps(nn.Module):

    def __init__(self, gfn_cfg):
        super().__init__()

        transformer_cfg = gfn_cfg.MODEL.TRANSFORMER
        self.compute_state_flow = (gfn_cfg.LOSS_TYPE != 'TB')
        self.concatenate_summary_token = transformer_cfg.PART1_HEAD.CONCATENATE_SUMMARY_TOKEN
        self.concatenate_candidate_tree = transformer_cfg.PART2_HEAD.CONCATENATE_CANDIDATE_TREE
        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE
        self.use_tree_type_embedding = transformer_cfg.USE_TREE_TYPE_EMBEDDING

        self.seq_emb = MLP(transformer_cfg.SEQ_EMB)
        embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        if self.condition_on_scale:
            self.scale_module = MLP(gfn_cfg.MODEL.SCALE_MLP)
        else:
            self.summary_token = nn.Parameter(torch.zeros(1, 1, embedding_size), requires_grad=True)
            trunc_normal_(self.summary_token, std=0.1)

        if self.use_tree_type_embedding:
            self.tree_type_embeddings = nn.Parameter(torch.zeros(2, embedding_size), requires_grad=True)
            trunc_normal_(self.tree_type_embeddings, std=0.1)

        self.shared_encoder = transformer_cfg.SHARED_ENCODER
        if self.shared_encoder:
            self.encoder = TransformerEncoder(transformer_cfg)
        else:
            self.part1_encoder = TransformerEncoder(transformer_cfg)
            self.part2_encoder = TransformerEncoder(transformer_cfg)

        self.part1_logits_head = MLP(transformer_cfg.PART1_HEAD)
        self.part2_logits_head = MLP(transformer_cfg.PART2_HEAD)
        if self.compute_state_flow:
            self.part1_flow_head = MLP(transformer_cfg.PART1_HEAD,
                                       input_size=transformer_cfg.PART1_HEAD.INPUT_SIZE // 2)
            self.part2_flow_head = MLP(transformer_cfg.PART2_HEAD,
                                       input_size=transformer_cfg.PART2_HEAD.INPUT_SIZE // 2)

    def get_head_token(self, scale):
        if self.condition_on_scale:
            token = self.scale_module(scale)
        else:
            token = self.summary_token
        return token

    def forward_part1(self, x, key_padding_mask=None):

        if self.shared_encoder:
            encoder = self.encoder
        else:
            encoder = self.part1_encoder

        x = encoder(x, key_padding_mask)
        summary_token = x[:, :1]
        x = x[:, 1:]
        flow_head_input = summary_token

        if self.concatenate_summary_token:
            _, num_trees, _ = x.shape
            summary_token = summary_token.expand(-1, num_trees, -1)
            x = torch.cat([x, summary_token], dim=2)

        logits = self.part1_logits_head(x).squeeze(-1)

        if self.compute_state_flow:
            log_state_flow = self.part1_flow_head(flow_head_input)
        else:
            log_state_flow = None
        return logits, log_state_flow

    def forward_part2(self, x, key_padding_mask=None):

        if self.shared_encoder:
            encoder = self.encoder
        else:
            encoder = self.part1_encoder

        x = encoder(x, key_padding_mask)
        flow_head_input = x[:, :1]
        candidate_token = x[:, 1:2]
        x = x[:, 2:]

        if self.concatenate_candidate_tree:
            _, num_trees, _ = x.shape
            candidate_token = candidate_token.expand(-1, num_trees, -1)
            x = torch.cat([candidate_token, x], dim=2)

        logits = self.part2_logits_head(x).squeeze(-1)
        if self.compute_state_flow:
            log_state_flow = self.part2_flow_head(flow_head_input)
        else:
            log_state_flow = None

        return logits, log_state_flow

    def model_params(self):
        return list(self.parameters())

    def forward(self, **kwargs):
        """
        :param batch_input: input tensors of shape [batch_size, nb_seq, seq_len], each sample in the batch is a state
        :param batch_intermediate_flag: boolean to tell if a state is intermediate
        :param batch_nb_seq: list of actual sequence length for each sample in the batch
        """
        batch_input = kwargs.get('batch_input')
        batch_intermediate_flag = kwargs.get('batch_intermediate_flag')
        batch_nb_seq = kwargs.get('batch_nb_seq')
        scale = kwargs.get('scale')

        batch_size, max_nb_seq, _ = batch_input.shape

        # batch_size, max_nb_seq, emb_size
        x = self.seq_emb(batch_input)

        # add tree type embedding
        if self.use_tree_type_embedding:
            x += self.tree_type_embeddings[0]
            if torch.any(batch_intermediate_flag):
                x[batch_intermediate_flag, 0] = x[batch_intermediate_flag, 0] - self.tree_type_embeddings[0] + \
                                                self.tree_type_embeddings[1]

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

        # in two-step unconditional models, the summary token is float32
        # therefore the concatenated x on line 138 is float32 too,
        # but the output logits and/or state_flow can be float16
        dtype = x.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
        all_logits_output = torch.zeros(batch_size, max_nb_seq, dtype=dtype).to(x.device)
        if self.compute_state_flow:
            all_log_flow_outputs = torch.zeros(batch_size, dtype=dtype).to(x.device)
        else:
            all_log_flow_outputs = None

        if torch.any(batch_intermediate_flag):
            # in intermediate state, the number of action is the number of sequences/subtrees minus 1
            # the minus one is for the candidate tree
            current_padding_mask = batch_padding_mask[batch_intermediate_flag]
            current_padding_mask = F.pad(current_padding_mask, (1, 0), "constant", False)
            part2_logits, part2_log_flows = self.forward_part2(x[batch_intermediate_flag], current_padding_mask)
            all_logits_output[batch_intermediate_flag, :max_nb_seq - 1] = part2_logits
            if self.compute_state_flow:
                all_log_flow_outputs[batch_intermediate_flag] = part2_log_flows.reshape(-1)

            # update the padding mask, because the candidate subtree has been removed from the logits
            batch_padding_mask[batch_intermediate_flag, batch_nb_seq[batch_intermediate_flag] - 1] = True

        if torch.any(~batch_intermediate_flag):
            # there may be a summary node
            current_padding_mask = batch_padding_mask[~batch_intermediate_flag]
            current_padding_mask = F.pad(current_padding_mask, (1, 0), "constant", False)
            part1_logits, part1_log_flows = self.forward_part1(x[~batch_intermediate_flag], current_padding_mask)
            all_logits_output[~batch_intermediate_flag] = part1_logits
            if self.compute_state_flow:
                all_log_flow_outputs[~batch_intermediate_flag] = part1_log_flows.reshape(-1)

        # logits in the shape of [batch_size, max_nb_seq]
        # note: logits contain paddings
        if self.compute_state_flow:
            ret = {
                'logits': all_logits_output,
                'log_flow': all_log_flow_outputs,
                'mask': batch_padding_mask
            }
        else:
            ret = {
                'logits': all_logits_output,
                'mask': batch_padding_mask
            }
        return ret
