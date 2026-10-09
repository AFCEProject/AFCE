"""Frozen DINOv3 patch-token extractor. Used only to build training targets.

Inference never loads this module. Spatial patch tokens are kept (no 4×4 pool).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from effect_vla.constants import DINO_IMAGE_SIZE


class FrozenDINOv3(nn.Module):
    """Returns (B, N, D) patch tokens. CLS and register tokens are dropped."""

    def __init__(
        self,
        ckpt_path: str | Path,
        image_size: int = DINO_IMAGE_SIZE,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.ckpt_path = Path(ckpt_path)
        self.image_size = int(image_size)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        self.processor = AutoImageProcessor.from_pretrained(self.ckpt_path)
        self.backbone = AutoModel.from_pretrained(self.ckpt_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.to(device=self.device, dtype=self.dtype)
        self.hidden = int(self.backbone.config.hidden_size)
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 16))
        self.n_reg = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)

    @torch.no_grad()
    def forward_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B,3,H,W) uint8 or float. Returns (B, N, D) float32 on CPU device of backbone."""
        x = images
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW, got {tuple(x.shape)}")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:
            x = x.float() / 255.0
        else:
            x = x.float()
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = ((x - mean_t) / std_t).to(device=self.device, dtype=self.dtype)

        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state
        patches = tokens[:, 1 + self.n_reg :, :]
        return patches.float()

    @torch.no_grad()
    def encode_numpy(self, imgs: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """imgs (N,H,W,3) uint8 → (N, P, D) float32."""
        imgs = np.asarray(imgs)
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(f"Expected NHWC RGB, got {imgs.shape}")
        outs = []
        for i in range(0, len(imgs), batch_size):
            chunk = imgs[i : i + batch_size]
            t = torch.from_numpy(np.ascontiguousarray(chunk)).permute(0, 3, 1, 2)
            outs.append(self.forward_images(t).cpu().numpy().astype(np.float32))
        if not outs:
            side = self.image_size // self.patch_size
            return np.zeros((0, side * side, self.hidden), np.float32)
        return np.concatenate(outs, 0)

    def patch_grid(self) -> int:
        return self.image_size // self.patch_size
