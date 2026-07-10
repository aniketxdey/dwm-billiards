from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        latent_h: int,
        latent_w: int,
        context_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: float,
        dropout: float,
        action_dim: int,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.context_len = int(context_len)
        self.latent_dim = self.latent_channels * self.latent_h * self.latent_w

        self.context_proj = nn.Linear(self.latent_dim, d_model)
        self.target_proj = nn.Linear(self.latent_dim, d_model)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.t_embed = nn.Embedding(diffusion_steps, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.context_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        ff_dim = int(d_model * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.latent_dim)

    def forward(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        noisy_target: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        # context: [B, L, C, H, W]
        b, l, c, h, w = context.shape
        assert l == self.context_len
        assert c == self.latent_channels and h == self.latent_h and w == self.latent_w

        ctx_flat = context.reshape(b, l, -1)
        tgt_flat = noisy_target.reshape(b, -1)

        ctx_tok = self.context_proj(ctx_flat)
        tgt_tok = self.target_proj(tgt_flat).unsqueeze(1)
        cond_tok = self.action_proj(action).unsqueeze(1) + self.t_embed(t_idx).unsqueeze(1)
        tgt_tok = tgt_tok + cond_tok

        x = torch.cat([ctx_tok, tgt_tok], dim=1)
        x = x + self.pos_emb[:, : x.shape[1], :]
        x = self.transformer(x)
        x = self.norm(x)

        pred = self.out_proj(x[:, -1, :])
        pred = pred.reshape(b, self.latent_channels, self.latent_h, self.latent_w)
        return pred
