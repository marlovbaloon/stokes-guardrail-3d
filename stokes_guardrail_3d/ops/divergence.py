import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteDivergence3D(nn.Module):
    """Computes 3D Discrete Divergence (nabla . F) via Central Difference Scheme.

    Input shape:  (B, 3, D, H, W) representing vector field (Fx, Fy, Fz)
    Output shape: (B, 1, D, H, W) scalar divergence field
    """

    def __init__(self, delta_spatial: float = 1.0):
        super().__init__()
        self.delta_spatial = delta_spatial

        # Setup 3D Central Difference Kernel (Sobel-like along x-axis)
        kernel_x = torch.tensor(
            [
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            ],
            dtype=torch.float32,
        ) / (32.0 * delta_spatial)

        kernel_y = kernel_x.transpose(1, 2)
        kernel_z = kernel_x.transpose(0, 2)

        # Register kernels as buffers
        self.register_buffer("kernel_x", kernel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_y", kernel_y.unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_z", kernel_z.unsqueeze(0).unsqueeze(0))

    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        if vector_field.dim() != 5 or vector_field.shape[1] != 3:
            raise ValueError(
                f"Expected tensor of shape (B, 3, D, H, W), got {tuple(vector_field.shape)}"
            )

        fx = vector_field[:, 0:1, ...]
        fy = vector_field[:, 1:2, ...]
        fz = vector_field[:, 2:3, ...]

        dfx_dx = F.conv3d(fx, self.kernel_x, padding=1)
        dfy_dy = F.conv3d(fy, self.kernel_y, padding=1)
        dfz_dz = F.conv3d(fz, self.kernel_z, padding=1)

        divergence = dfx_dx + dfy_dy + dfz_dz
        return divergence
