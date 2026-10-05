"""
3D Discrete Curl Operator via Central Difference Scheme.
Mathematical formulation:
    nabla x F = (dFz/dy - dFy/dz)i + (dFx/dz - dFz/dx)j + (dFy/dx - dFx/dy)k
Deterministic computation via torch.nn.functional.conv3d.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteCurl3D(nn.Module):
    def __init__(self, delta_spatial: float = 1.0):
        super().__init__()
        self.delta_spatial = delta_spatial
        
        # Setup 3D Sobel / Central Difference Kernel
        kernel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / (32.0 * delta_spatial)

        kernel_y = kernel_x.transpose(1, 2)
        kernel_z = kernel_x.transpose(0, 2)

        spatial_kernel = torch.stack([kernel_x, kernel_y, kernel_z], dim=0).unsqueeze(1)
        self.register_buffer("spatial_kernel", spatial_kernel)

    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_field (torch.Tensor): Shape (B, 3, D, H, W) -> (Fx, Fy, Fz)
        Returns:
            torch.Tensor: Curl vector field of shape (B, 3, D, H, W)
        """
        fx = vector_field[:, 0:1, ...]
        fy = vector_field[:, 1:2, ...]
        fz = vector_field[:, 2:3, ...]

        deriv_fx = F.conv3d(fx, self.spatial_kernel, padding=1)
        deriv_fy = F.conv3d(fy, self.spatial_kernel, padding=1)
        deriv_fz = F.conv3d(fz, self.spatial_kernel, padding=1)

        curl_x = deriv_fz[:, 1:2, ...] - deriv_fy[:, 2:3, ...]
        curl_y = deriv_fx[:, 2:3, ...] - deriv_fz[:, 0:1, ...]
        curl_z = deriv_fy[:, 0:1, ...] - deriv_fx[:, 1:2, ...]

        return torch.cat([curl_x, curl_y, curl_z], dim=1)
