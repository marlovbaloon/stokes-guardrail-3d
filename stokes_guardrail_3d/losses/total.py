"""
Total Multi-Task Objective Loss:
    L_total = L_classification + lambda1 * L_stokes + lambda2 * L_align
"""
import torch
import torch.nn as nn
from .stokes import StokesGuardrailLoss
from .alignment import AnatomicalAlignmentLoss


class TotalObjectiveLoss(nn.Module):
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1, tau: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.stokes_loss = StokesGuardrailLoss(tau=tau)
        self.align_loss = AnatomicalAlignmentLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        vector_field: torch.Tensor, 
        input_image: torch.Tensor
    ) -> torch.Tensor:
        l_cls = self.ce_loss(logits, targets)
        l_stokes = self.stokes_loss(vector_field)
        l_align = self.align_loss(vector_field, input_image)

        return l_cls + (self.lambda1 * l_stokes) + (self.lambda2 * l_align)
