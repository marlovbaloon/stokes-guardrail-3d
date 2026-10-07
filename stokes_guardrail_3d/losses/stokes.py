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
    """Combines Curl Loss (Stokes) and Divergence Loss (Gauss) with Voxel-wise Mapping support."""

    def __init__(
        self, 
        tau_curl: float = 0.5, 
        tau_div: float = 0.5, 
        gamma: float = 1.0,
        return_map: bool = False
    ):
        super().__init__()
        self.tau_curl = tau_curl
        self.tau_div = tau_div
        self.gamma = gamma
        self.return_map = return_map

        self.curl_op = DiscreteCurl3D()
        self.div_op = DiscreteDivergence3D()

    def forward(self, vector_field: torch.Tensor, roi_mask: torch.Tensor = None):
        # 1. Compute Voxel-wise Fields
        curl_field = self.curl_op(vector_field)
        curl_map = F.relu(torch.norm(curl_field, dim=1, keepdim=True) - self.tau_curl)

        div_field = self.div_op(vector_field)
        div_map = F.relu(torch.abs(div_field) - self.tau_div)

        # 2. Regional Reduction via ROI Masking
        if roi_mask is not None:
            roi_vol = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + 1e-8
            l_curl = torch.sum(curl_map * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            l_div = torch.sum(div_map * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            
            l_curl = torch.mean(l_curl)
            l_div = torch.mean(l_div)
        else:
            l_curl = torch.mean(curl_map)
            l_div = torch.mean(div_map)

        total_loss = l_curl + (self.gamma * l_div)

        if self.return_map:
            return {
                "loss": total_loss,
                "curl_map": curl_map,    # (B, 1, D, H, W) Turbulence Map
                "div_map": div_map       # (B, 1, D, H, W) Atrophy/Shrinkage Map
            }

        return total_loss
