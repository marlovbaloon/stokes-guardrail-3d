"""
Volume Mass Conservation Loss.
Penalizes mass flux imbalances and local density violations:
    L_mass = ||Net_Flux||^2 + gamma * mean(max(0, |nabla . F| - tau))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union
from ..ops.mass_flux import DiscreteVolumeFlux3D, FusedVolumeFlux3DIR


class VolumeMassConservationLoss(nn.Module):
    def __init__(self, tau: float = 0.0, gamma: float = 1.0, return_map: bool = False, eps: float = 1e-8):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.return_map = return_map
        self.eps = eps
        self.flux_op = DiscreteVolumeFlux3D()

    def forward(
        self, 
        vector_field: torch.Tensor, 
        roi_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.flux_op(vector_field)
        div_density = out["divergence_density"]
        net_flux = out["net_volume_flux"]

        # Global Flux Conservation Penalty
        l_global_flux = torch.mean(net_flux ** 2)

        # Local Density Excess Penalty
        local_penalty = F.relu(torch.abs(div_density) - self.tau)

        if roi_mask is not None:
            roi_vol = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + self.eps
            l_local = torch.sum(local_penalty * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            l_local = torch.mean(l_local)
        else:
            l_local = torch.mean(local_penalty)

        total_loss = l_global_flux + (self.gamma * l_local)

        if self.return_map:
            return {
                "loss": total_loss,
                "mass_leakage_map": local_penalty,  # Voxel-wise mass anomaly map
                "net_flux": net_flux
            }

        return total_loss


class FusedVolumeMassConservationIR(nn.Module):
    """Fused IR Node for Mass Conservation Loss Evaluation."""
    def __init__(self, tau: float = 0.0, gamma: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.eps = eps
        self.fused_flux_op = FusedVolumeFlux3DIR()

    def forward(
        self, 
        vector_field: torch.Tensor, 
        roi_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        out = self.fused_flux_op(vector_field)
        div_density = out["divergence_density"]
        net_flux = out["net_volume_flux"]

        l_global_flux = torch.mean(net_flux ** 2)
        local_penalty = F.relu(torch.abs(div_density) - self.tau)

        if roi_mask is not None:
            roi_vol = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + self.eps
            l_local = torch.sum(local_penalty * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            l_local = torch.mean(l_local)
        else:
            l_local = torch.mean(local_penalty)

        total_loss = l_global_flux + (self.gamma * l_local)

        return {
            "loss": total_loss,
            "mass_leakage_map": local_penalty,
            "net_flux": net_flux
        }
