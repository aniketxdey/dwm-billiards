from __future__ import annotations

import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, latent_channels: int = 4) -> None:
        super().__init__()

        b = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, b, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, b, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, b * 2, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
        )

        self.to_mu = nn.Conv2d(b * 4, latent_channels, 1)
        self.to_logvar = nn.Conv2d(b * 4, latent_channels, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, b * 4, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(b, b, 4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(b, in_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
