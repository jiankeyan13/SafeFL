"""Clean and backdoor model evaluation for SafeFL v2."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


class Evaluator:
    """Stateless evaluator for global clean accuracy (CDA) and backdoor ASR."""

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        criterion: Optional[nn.Module],
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate clean accuracy and optional average loss over ``dataloader``."""
        was_training = model.training
        model.eval()
        model.to(device)

        correct = 0
        total = 0
        total_loss = 0.0

        for inputs, targets in dataloader:
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
            if targets.device != device:
                targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            bsz = int(targets.size(0))
            if criterion is not None:
                total_loss += float(criterion(outputs, targets).item()) * bsz

            preds = outputs.argmax(dim=1) if outputs.ndim > 1 else outputs
            correct += int(preds.eq(targets).sum().item())
            total += bsz

        if was_training:
            model.train()

        results: Dict[str, float] = {
            "accuracy": (correct / total) if total > 0 else 0.0,
        }
        if criterion is not None:
            results["loss"] = (total_loss / total) if total > 0 else 0.0
        return results

    @torch.no_grad()
    def evaluate_backdoor(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        criterion: Optional[nn.Module],
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate targeted ASR and backdoor accuracy in a single forward pass.

        Expects each batch from ``dataloader`` to be ``(inputs, target_labels, original_labels)``:
          - ``asr``: fraction of non-target samples classified as ``target_labels``.
          - ``backdoor_acc``: fraction of triggered samples still classified as ``original_labels``.
        """
        was_training = model.training
        model.eval()
        model.to(device)

        asr_correct = 0
        asr_total = 0
        backdoor_correct = 0
        total = 0
        total_loss = 0.0

        for inputs, target_labels, original_labels in dataloader:
            if inputs.device != device:
                inputs = inputs.to(device, non_blocking=True)
            if target_labels.device != device:
                target_labels = target_labels.to(device, non_blocking=True)
            if original_labels.device != device:
                original_labels = original_labels.to(device, non_blocking=True)

            outputs = model(inputs)
            bsz = int(original_labels.size(0))
            if criterion is not None:
                total_loss += float(criterion(outputs, original_labels).item()) * bsz

            preds = outputs.argmax(dim=1)
            non_target_mask = original_labels.ne(target_labels)
            asr_total += int(non_target_mask.sum().item())
            asr_correct += int((preds.eq(target_labels) & non_target_mask).sum().item())
            backdoor_correct += int(preds.eq(original_labels).sum().item())
            total += bsz

        if was_training:
            model.train()

        out: Dict[str, float] = {
            "asr": (asr_correct / asr_total) if asr_total > 0 else 0.0,
            "backdoor_acc": (backdoor_correct / total) if total > 0 else 0.0,
        }
        if criterion is not None and total > 0:
            out["backdoor_loss"] = total_loss / total
        return out
