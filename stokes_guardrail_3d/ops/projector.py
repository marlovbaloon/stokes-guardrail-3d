import torch
import torch.nn as nn


class LatentToVectorField3D(nn.Module):
    """Projects arbitrary high-dimensional latent channels (C) down to a 3D vector field (3).

    Input shape:  (B, C, D, H, W)
    Output shape: (B, 3, D, H, W) representing (Fx, Fy, Fz)
    """

    def __init__(self, in_channels: int, bound_mode: str = "tanh"):
        super().__init__()
        if in_channels < 1:
            raise ValueError(
                f"in_channels must be greater than 0, got {in_channels}"
            )

        self.in_channels = in_channels
        self.bound_mode = bound_mode.lower()

        # Pointwise 1x1x1 convolution for efficient spatial linear projection
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D tensor (B, C, D, H, W), got shape {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )

        field = self.proj(x)

        if self.bound_mode == "tanh":
            field = torch.tanh(field)
        elif self.bound_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported bound_mode: {self.bound_mode}")

        return field
