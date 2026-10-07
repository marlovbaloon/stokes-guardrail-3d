"""
Gauss-Bonnet Topology Preserving Loss.
Enforces target Euler Characteristic chi_target (e.g., sphere topology chi = 2):
    L_gb = (euler_characteristic - target_chi)^2 + gamma * mean(max(0, K - tau))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union
from ..ops.gauss_bonnet import DiscreteEulerCharacteristic3D


class GaussBonnetTopologyLoss(nn.Module):
    def __init__(
        self, 
        target_chi: float = 2.0, 
        tau: float = 0.1, 
        gamma: float = 0.1, 
        return_map: bool = False,
        eps: float = 1e-8
    ):
        super().__init__()
        self.target_chi = target_chi
        self.tau = tau
        self.gamma = gamma
        self.return_map = return_map
        self.eps = eps
        self.gb_op = DiscreteEulerCharacteristic3D()

    def forward(
        self, 
        feature_map: torch.Tensor, 
        roi_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.gb_op(feature_map)
        k_map = out["gaussian_curvature_map"]
        euler_char = out["euler_characteristic"]

        # Global Topological Defect Loss
        l_chi = torch.mean((euler_char - self.target_chi) ** 2)

        # Local Curvature Spike Penalty
        local_penalty = F.relu(k_map - self.tau)

        if roi_mask is not None:
            roi_vol = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + self.eps
            l_local = torch.sum(local_penalty * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            l_local = torch.mean(l_local)
        else:
            l_local = torch.mean(local_penalty)

        total_loss = l_chi + (self.gamma * l_local)

        if self.return_map:
            return {
                "loss": total_loss,
                "topological_defect_map": local_penalty,  # Voxel-wise anomaly heatmap
                "euler_characteristic": euler_char
            }

        return total_loss
