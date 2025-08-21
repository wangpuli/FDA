# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from condition.vanilla_iTransfomer import ConditionModel

# modulate x: add noises to samples
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
# hidden size: embedding time step into the same dimension as noisy latents
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # covert to frequency domain
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# Condition Vector Embeddings (latent dynamics of spikes)
class LatentDynamicEmbedder(nn.Module):
    """
    Embeds spike signal windows into latent dynamics
    """
    def __init__(self, model_config):
        super().__init__()
        self.transformer_model = ConditionModel(model_config)
    
    def forward(self, x):
        _, cond_vec = self.transformer_model(
            x_enc=x,
            x_mark_enc=None
        )
        return cond_vec

#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # set attention and normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # feedforward networks (hidden dim of mlp: mlp_hidden_dim)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            # no_act
            nn.SiLU(),
            nn.Linear(2*hidden_size, 6* hidden_size, bias=True),
        )

    def forward(self, x, c):
        # c: conditional vectors (AdaLN)
        _, _, _, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # feedforward networks (with residual connections)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    hidden_size: dimension of hidden features
    """
    # def __init__(self, hidden_size, patch_size, out_channels):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # output layers
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            # no_scale
            nn.SiLU(),
            nn.Linear(hidden_size*2, hidden_size*2, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # no_scale
        shift, scale = shift*10, scale*10
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone. (single time step)
    """
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        window_size=5,
        hidden_size=1152,
        out_dim=2,
        depth=5,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=True,
        model_config=None,
        invert_flag=True,
        trans_flag=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # time step embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)

        # conditional latent features (c)
        if model_config is None:
            raise ValueError("Model configuration must be provided.")
        model_config.d_model = hidden_size
        self.dynamic_embedder = LatentDynamicEmbedder(model_config)

        self.spike_embedder = nn.Linear(int(in_channels*self.window_size), hidden_size, bias=True)

        # SiTBlock Layers
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # output layers
        self.final_layer = FinalLayer(hidden_size, hidden_size)
        # decoding layer for expected neural manifold
        self.linear_encoder = nn.Linear(out_dim, hidden_size, bias=True)
        # reconstruction loss
        self.reconstruction_decoder = nn.Linear(hidden_size, in_channels, bias=True)
        self.invert_flag = invert_flag
        self.trans_flag = trans_flag
        self.cond_vec = None
        self.out_dim = out_dim

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_cond_vec(self):
        return self.cond_vec
    
    def get_cond_latent(self, y):
        if self.invert_flag:
            y = y.transpose(-2, -1)
        _, y = self.dynamic_embedder.transformer_model(y, x_mark_enc=None)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        return y

    def forward(self, x, t, y):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        t = self.t_embedder(t)                   # (N, D)
        # latent dynamic representation
        if self.invert_flag:
            y = y.transpose(-2, -1)
        _, y = self.dynamic_embedder.transformer_model(y, x_mark_enc=None) 
        
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        if len(t.shape) == 1:
            t = t.unsqueeze(0)

        self.cond_vec = y.clone()
        
        # conditional vectors
        c = torch.cat((t, y), dim=1)                            # (N, D)
        # no_act, no_scale
        for block in self.blocks:
            # data + conditional vectors
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        return x
    
    # designed for sampling (classifier-free guidance)
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        x: noisy latent features
        y: conditional latent dynamics
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # combined = x
        model_out = self.forward(combined, t, y)

        eps = model_out[:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        return eps

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb