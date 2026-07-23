"""Malicious client integration for the Neurotoxin training attack."""
from __future__ import annotations

from typing import Any, Dict

from core.client.malicious_client import MaliciousClient


class NeurotoxinClient(MaliciousClient):
    """Prepare the previous-global-update mask before malicious local training."""

    def train(self) -> Dict[str, Any]:
        prepare = getattr(self.attack_profile, "prepare_gradient_mask", None)
        if callable(prepare):
            prepare(self.model, getattr(self, "prev_global_delta", None))
        return super().train()
