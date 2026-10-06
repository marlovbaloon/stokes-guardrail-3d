"""
Stokes' Guardrail Loss Function.
Enforces geometric integrity using magnitude of 3D curl:
    L_stokes = (1/V) * sum( max(0, ||nabla x F|| - tau) )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.curl import DiscreteCurl3D
from ..ops.divergence import DiscreteDivergence3D


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

class HelmholtzGuardrailLoss(nn.Module):
    """Combines Curl Loss (Stokes) and Divergence Loss (Gauss)

    L_helmholtz = L_curl + gamma * L_div
    """

    def __init__(
        self, tau_curl: float = 0.5, tau_div: float = 0.5, gamma: float = 1.0
    ):
        super().__init__()
        self.tau_curl = tau_curl
        self.tau_div = tau_div
        self.gamma = gamma

        self.curl_op = DiscreteCurl3D()
        self.div_op = DiscreteDivergence3D()

    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        # Curl Penalty (Rotational Turbulence)
        curl_field = self.curl_op(vector_field)
        curl_magnitude = torch.norm(curl_field, dim=1, keepdim=True)
        l_curl = torch.mean(F.relu(curl_magnitude - self.tau_curl))

        # Divergence Penalty (Compressional / Expansional Anomaly)
        div_field = self.div_op(vector_field)
        l_div = torch.mean(F.relu(torch.abs(div_field) - self.tau_div))

        return l_curl + (self.gamma * l_div)
