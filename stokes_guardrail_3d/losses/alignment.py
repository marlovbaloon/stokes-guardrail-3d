"""
Anatomical Directional Alignment Loss & Biomarker Extractor.
Calculates voxel-wise directional disalignment and supports optional ROI masking 
(e.g., Hippocampus, Cortex) for regional Alzheimer's biomarker quantification.

 Constrains vector field F with anatomical gradient field G = grad(I):
    L_align = 1 - (1/V) * sum( ((F . G) / (||F||*||G|| + eps))^2 ) 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.curl import DiscreteCurl3D


class AnatomicalAlignmentLoss(nn.Module):
    def __init__(self, eps: float = 1e-8, return_map: bool = False):
        super().__init__()
        self.eps = eps
        self.return_map = return_map
        self.grad_op = DiscreteCurl3D()  # Shared spatial kernel operator

    def forward(
        self, 
        vector_field: torch.Tensor, 
        input_image: torch.Tensor, 
        roi_mask: torch.Tensor = None
    ):
        """
        Args:
            vector_field (torch.Tensor): Feature vector field (B, 3, D, H, W)
            input_image (torch.Tensor): Raw 3D Brain MRI image (B, 1, D, H, W)
            roi_mask (torch.Tensor, optional): Binary 3D mask for target ROI (B, 1, D, H, W)
                                               e.g., Hippocampus or Cortex segmentation.
        Returns:
            torch.Tensor | dict: Scalar Loss or Dict containing Voxel-wise Map and ROI Score.
        """
        # Calculate Anatomical Image Gradients grad(I)
        grad_x = F.conv3d(input_image, self.grad_op.spatial_kernel[0:1], padding=1)
        grad_y = F.conv3d(input_image, self.grad_op.spatial_kernel[1:2], padding=1)
        grad_z = F.conv3d(input_image, self.grad_op.spatial_kernel[2:3], padding=1)
        anatomical_grad = torch.cat([grad_x, grad_y, grad_z], dim=1)

        # Spatial Vector Operations
        dot_product = torch.sum(vector_field * anatomical_grad, dim=1, keepdim=True)
        norm_v = torch.norm(vector_field, dim=1, keepdim=True)
        norm_g = torch.norm(anatomical_grad, dim=1, keepdim=True)

        cosine_sim = dot_product / (norm_v * norm_g + self.eps)
        
        # 1. Voxel-wise Disalignment Heatmap Map: (B, 1, D, H, W)
        # Represents local anatomical directional deviation per voxel
        alignment_map = 1.0 - (cosine_sim ** 2)

        # 2. Regional Reduction / Masking
        if roi_mask is not None:
            # Masking out non-ROI regions
            masked_map = alignment_map * roi_mask
            roi_volume = torch.sum(roi_mask, dim=(1, 2, 3, 4), keepdim=True) + self.eps
            loss = torch.sum(masked_map, dim=(1, 2, 3, 4), keepdim=True) / roi_volume
            loss = torch.mean(loss)  # Batch mean
        else:
            loss = torch.mean(alignment_map)

        # 3. Output Control
        if self.return_map:
            return {
                "loss": loss,
                "alignment_map": alignment_map,  # Voxel-wise 3D Heatmap
                "roi_biomarker_score": loss if roi_mask is not None else None
            }
            
        return loss
