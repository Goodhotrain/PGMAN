"""Decomposed multimodal alignment used by PGMAN.

DecAlign projects visual, text, and audio representations into a shared space
and optimizes the three pairwise alignment objectives independently.  Keeping
the objectives separate makes it easy to inspect or reweight each modality
pair without changing the feature encoders.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class DecAlign(nn.Module):
    """Decomposed visual-text-audio contrastive alignment.

    Args:
        visual_dim: Dimension of the visual representation.
        text_dim: Dimension of the text representation.
        audio_dim: Dimension of the audio representation.
        projection_dim: Size of the shared alignment space.
        temperature: Temperature used by the pairwise InfoNCE objectives.
        pair_weights: Optional weights for ``visual_text``, ``visual_audio``,
            and ``text_audio``. The weights are normalized to sum to one.
    """

    PAIRS = {
        "visual_text": ("visual", "text"),
        "visual_audio": ("visual", "audio"),
        "text_audio": ("text", "audio"),
    }

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        audio_dim: int,
        projection_dim: int = 768,
        temperature: float = 0.1,
        pair_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = float(temperature)
        self.projections = nn.ModuleDict(
            {
                "visual": nn.Linear(visual_dim, projection_dim),
                "text": nn.Linear(text_dim, projection_dim),
                "audio": nn.Linear(audio_dim, projection_dim),
            }
        )

        weights = dict.fromkeys(self.PAIRS, 1.0)
        if pair_weights is not None:
            unknown = set(pair_weights) - set(self.PAIRS)
            if unknown:
                raise ValueError(f"unknown modality pairs: {sorted(unknown)}")
            weights.update(pair_weights)
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("pair weights must be non-negative")
        weight_sum = sum(weights.values())
        if weight_sum == 0:
            raise ValueError("at least one pair weight must be positive")
        self.pair_weights = {
            name: weight / weight_sum for name, weight in weights.items()
        }

    def encode(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        audio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project and L2-normalize one representation per modality."""
        features = {"visual": visual, "text": text, "audio": audio}
        batch_sizes = {name: value.shape[0] for name, value in features.items()}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(f"modalities have different batch sizes: {batch_sizes}")
        return {
            name: F.normalize(self.projections[name](value), dim=-1)
            for name, value in features.items()
        }

    def _pair_loss(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        logits = left @ right.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        audio: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aligned = self.encode(visual, text, audio)
        losses = {
            name: self._pair_loss(aligned[left], aligned[right])
            for name, (left, right) in self.PAIRS.items()
        }
        total = sum(self.pair_weights[name] * loss for name, loss in losses.items())
        if return_components:
            return total, losses
        return total
