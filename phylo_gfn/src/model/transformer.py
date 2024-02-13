import torch.nn as nn
from src.model.weight_init import trunc_normal_


class SAMlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., with_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=with_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=with_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SAMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, with_bias=True)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, transformer_cfg):
        super().__init__()

        embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        self.blocks = nn.ModuleList([
            Block(
                dim=embedding_size, num_heads=transformer_cfg.NUM_HEADS,
                mlp_ratio=transformer_cfg.MLP_RATIO,
                drop=transformer_cfg.DROP_RATE, attn_drop=transformer_cfg.ATTN_DROP_RATE)
            for i in range(transformer_cfg.DEPTH)])

        self.norm = nn.LayerNorm(embedding_size, elementwise_affine=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, key_padding_mask=None):

        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        x = self.norm(x)
        return x
