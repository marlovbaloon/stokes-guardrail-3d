"""
Anatomical Directional Alignment Loss.
Constrains vector field F with anatomical gradient field G = grad(I):
    L_align = 1 - (1/V) * sum( ((F . G) / (||F||*||G|| + eps))^2 )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.curl import DiscreteCurl3D


class AnatomicalAlignmentLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.grad_op = DiscreteCurl3D()  # Shared spatial kernel operator

    def forward(self, vector_field: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vector_field (torch.Tensor): Feature vector field (B, 3, D, H, W)
            input_image (torch.Tensor): Raw 3D Brain MRI image (B, 1, D, H, W)
        """
        grad_x = F.conv3d(input_image, self.grad_op.spatial_kernel[0:1], padding=1)
        grad_y = F.conv3d(input_image, self.grad_op.spatial_kernel[1:2], padding=1)
        grad_z = F.conv3d(input_image, self.grad_op.spatial_kernel[2:3], padding=1)
        anatomical_grad = torch.cat([grad_x, grad_y, grad_z], dim=1)

        dot_product = torch.sum(vector_field * anatomical_grad, dim=1, keepdim=True)
        norm_v = torch.norm(vector_field, dim=1, keepdim=True)
        norm_g = torch.norm(anatomical_grad, dim=1, keepdim=True)

        cosine_sim = dot_product / (norm_v * norm_g + self.eps)
        return 1.0 - torch.mean(cosine_sim ** 2)
