from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


class LPIPSWrapper:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.model = None
        if enabled:
            try:
                import lpips  # type: ignore

                self.model = lpips.LPIPS(net="vgg")
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
            except Exception:
                self.enabled = False
                self.model = None

    def to(self, device: torch.device) -> "LPIPSWrapper":
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.model is None:
            return torch.zeros((), device=x.device)
        return self.model(x, y).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def beta_schedule(processed_frames: int, cfg_loss: Dict[str, Any]) -> float:
    start = float(cfg_loss["kl_beta_start"])
    end = float(cfg_loss["kl_beta_end"])
    warmup = int(cfg_loss["kl_warmup_frames"])
    if warmup <= 0:
        return end
    ratio = min(1.0, processed_frames / float(warmup))
    return start + ratio * (end - start)


def compute_vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    processed_frames: int,
    cfg_loss: Dict[str, Any],
    lpips_fn: LPIPSWrapper,
) -> dict[str, torch.Tensor | float]:
    l1 = F.l1_loss(recon, x)
    lp = lpips_fn(recon, x)
    kl = kl_divergence(mu, logvar)

    beta = beta_schedule(processed_frames, cfg_loss)

    total = (
        float(cfg_loss["recon_l1_weight"]) * l1
        + float(cfg_loss["lpips_weight"]) * lp
        + float(cfg_loss["kl_weight"]) * beta * kl
    )

    return {
        "total": total,
        "l1": l1.detach(),
        "lpips": lp.detach(),
        "kl": kl.detach(),
        "beta": beta,
    }
