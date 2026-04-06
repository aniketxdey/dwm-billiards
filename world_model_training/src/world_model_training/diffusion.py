from __future__ import annotations

import torch


class GaussianDiffusion:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        if timesteps < 2:
            raise ValueError("timesteps must be >= 2")

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.timesteps = int(timesteps)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    @staticmethod
    def _extract(coeff: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = coeff[t]
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        c1 = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        c2 = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return c1 * x0 + c2 * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        c1 = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        c2 = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        return (x_t - c2 * eps) / torch.clamp(c1, min=1e-8)

    def training_loss(
        self,
        model,
        context: torch.Tensor,
        action: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(target)
        noisy = self.q_sample(target, t, noise)
        pred_noise = model(context=context, action=action, noisy_target=noisy, t_idx=t)
        loss = torch.mean((pred_noise - noise) ** 2)
        return loss, pred_noise, noise
