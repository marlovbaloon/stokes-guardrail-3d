"""
Ricci Curvature Regularization Loss & Fused Optimization Pass.
Penalizes high Ricci curvature singularities on 3D latent manifolds:
    L_ricci = (1/V) * sum( max(0, ||R_ij|| - tau) )
where R_ij is approximated via Discrete Metric Laplacian on Grid Domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class RicciCurvatureLoss(nn.Module):
    """
    Computes Ricci Curvature Penalty over 3D Latent Manifolds.
    Enforces Smooth Riemannian Geometry during training to prevent 
    topological hallucinations and artifact spikes in medical image analysis.
    """
    def __init__(self, tau: float = 0.1, return_map: bool = False, eps: float = 1e-8):
        super().__init__()
        self.tau = tau
        self.return_map = return_map
        self.eps = eps

        # 3D Discrete Laplacian Kernel for metric second-order derivatives
        # Used to approximate Ricci Tensor components: R_ij ≈ -1/2 * Δ(g_ij)
        laplacian_kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        self.register_buffer("laplacian_kernel", laplacian_kernel)

    def forward(
        self, 
        vector_field: torch.Tensor, 
        roi_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            vector_field (torch.Tensor): Latent feature tensor of shape (B, C, D, H, W)
            roi_mask (torch.Tensor, optional): Binary mask of shape (B, 1, D, H, W)
        Returns:
            torch.Tensor | Dict[str, torch.Tensor]: Scalar loss or Map dictionary
        """
        b, c, d, h, w = vector_field.shape

        # 1. Compute Metric Laplacian across C channels via Grouped Conv3D
        # Reshape to treat each channel as a scalar field component of the metric
        v_reshaped = vector_field.view(b * c, 1, d, h, w)
        laplacian = F.conv3d(v_reshaped, self.laplacian_kernel, padding=1)
        laplacian = laplacian.view(b, c, d, h, w)

        # 2. Approximate Ricci Curvature Norm: ||R_ij|| ≈ 0.5 * ||Δ(g)||_2
        ricci_norm = 0.5 * torch.norm(laplacian, dim=1, keepdim=True)

        # 3. Apply ReLU thresholding for curvature penalty
        ricci_map = F.relu(ricci_norm - self.tau)

        # 4. Regional Masking & Loss Reduction
        if roi_mask is not None:
            roi_vol = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + self.eps
            loss = torch.sum(ricci_map * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            loss = torch.mean(loss)
        else:
            loss = torch.mean(ricci_map)

        if self.return_map:
            return {
                "loss": loss,
                "ricci_map": ricci_map,  # Voxel-wise Curvature Penalty Heatmap (B, 1, D, H, W)
                "ricci_norm": ricci_norm
            }

        return loss


class FusedRicciCurvatureIR(nn.Module):
    """
    Fused High-Performance Pass for Ricci Curvature Computation.
    Fuses channel-wise 3D Discrete Laplacians into a single grouped 3D convolution kernel 
    to maximize L2 Cache hit rates and reduce GPU memory latency.
    """
    def __init__(self, channels: int, tau: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.tau = tau
        self.eps = eps

        # Pack C channels into a single grouped convolution filter: (C, 1, 3, 3, 3)
        base_laplacian = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        fused_kernel = base_laplacian.repeat(channels, 1, 1, 1, 1)
        self.register_buffer("fused_kernel", fused_kernel)

    def forward(
        self, 
        vector_field: torch.Tensor, 
        roi_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        # Single Grouped Conv3D execution pass over all channels at once
        laplacian = F.conv3d(vector_field, self.fused_kernel, padding=1, groups=self.channels)
        
        # Calculate Ricci curvature magnitude
        ricci_norm = 0.5 * torch.norm(laplacian, dim=1, keepdim=True)
        ricci_map = F.relu(ricci_norm - self.tau)

        if roi_mask is not None:
            roi_vol = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + self.eps
            loss = torch.sum(ricci_map * roi_mask, dim=(1, 2, 3, 4), keepdim=True) / roi_vol
            loss = torch.mean(loss)
        else:
            loss = torch.mean(ricci_map)

        return {
            "loss": loss,
            "ricci_map": ricci_map,
            "ricci_norm": ricci_norm
        }


def optimize_ricci_loss_pass(loss_module: RicciCurvatureLoss, in_channels: int) -> FusedRicciCurvatureIR:
    """
    Hot-swaps standard RicciCurvatureLoss with FusedRicciCurvatureIR node.
    """
    fused_module = FusedRicciCurvatureIR(
        channels=in_channels,
        tau=loss_module.tau,
        eps=loss_module.eps
    )
    return fused_module
