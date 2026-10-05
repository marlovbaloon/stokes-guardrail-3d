"""
Stokes' Guardrail Loss Function.
Enforces geometric integrity using magnitude of 3D curl:
    L_stokes = (1/V) * sum( max(0, ||nabla x F|| - tau) )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.curl import DiscreteCurl3D


class StokesGuardrailLoss(nn.Module):
    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        self.curl_op = DiscreteCurl3D()

    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_field (torch.Tensor): Latent vector field M(H) of shape (B, 3, D, H, W)
        """
        curl_field = self.curl_op(vector_field)
        curl_magnitude = torch.norm(curl_field, dim=1, keepdim=True)
        penalty = F.relu(curl_magnitude - self.tau)
        return torch.mean(penalty)
