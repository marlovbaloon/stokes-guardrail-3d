import torch
import torch.nn as nn
from monai.networks.nets import UNet
from stokes_guardrail_3d.ops.projector import LatentToVectorField3D


class StokesMONAIUNet3D(nn.Module):
    """Wraps MONAI's 3D UNet to extract latent features and project them

    to a 3D vector field for Stokes' Guardrail regularization.
    """

    def __init__(
        def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        latent_channels: int = 64,
    ):
        super().__init__()
        # 1. MONAI 3D UNet Backbone
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
        )

        # 2. Stokes Projector (C channels -> 3D Vector Field)
        self.stokes_projector = LatentToVectorField3D(
            in_channels=latent_channels, bound_mode="tanh"
        )

    def forward(self, x: torch.Tensor):
        # Extract features using MONAI UNet encoder
        latent = self.backbone.model[0](x)  # Encoder Output

        # Segmentation Logits
        logits = self.backbone(x)

        # Map Latent Feature Map to 3D Vector Field F (B, 3, D, H, W)
        vector_field = self.stokes_projector(latent)

        return logits, vector_field