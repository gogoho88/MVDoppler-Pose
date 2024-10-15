"""
-refer to https://github.com/autonomousvision/transfuser/blob/main/transfuser/model.py
"""
import math
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, n_channel=3, patch_size=16, emb_size=512):
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(n_channel, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    def forward(self, x):
        x = self.projection(x)
        return x


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            # nn.Tanh(), # changed from GELU
            nn.ReLU(True),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    T_anchor_dim, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop):
        """
            Args:
                n_embd: # expected features from input/output
                n_head: # heads in multiheadattention
        """
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.T_anchor_dim = T_anchor_dim


        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.randn(1, 2*T_anchor_dim, n_embd))
        
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, radar1_tensor, radar2_tensor):
        """
        Args:
            radar1_tensor (tensor): B, C, T (Batch, Channel, Time x Freq.)
            radar2_tensor (tensor): B, C, T (Batch, Channel, Time x Freq.)
        """
        B, C, T = radar1_tensor.shape
        
        # forward the image model for token embeddings
        radar1_tensor = radar1_tensor.view(B, self.seq_len, -1, T)
        radar2_tensor = radar2_tensor.view(B, self.seq_len, -1, T)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([radar1_tensor, radar2_tensor], dim=1).permute(0,1,3,2).contiguous()
        token_embeddings = token_embeddings.view(B, -1, self.n_embd) # (B, 2 * T, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings) # (B, an * T, C)
        
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        x = x.view(B, 2*self.seq_len, self.T_anchor_dim, self.n_embd)
        x = x.permute(0,1,3,2).contiguous() # same as token_embeddings

        radar1_tensor_out = x[:, 0, :, :].contiguous()
        radar2_tensor_out = x[:, 1, :, :].contiguous()
        
        return radar1_tensor_out, radar2_tensor_out