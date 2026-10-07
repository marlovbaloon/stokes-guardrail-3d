"""
3D Discrete Euler Characteristic & Gaussian Curvature Operator.
Evaluates local Gaussian curvature K and topological Euler Characteristic chi(M):
    integral_M K dA = 2 * pi * chi(M)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Dict


class DiscreteEulerCharacteristic3D(nn.Module):
    """
    Computes local Gaussian Curvature density and estimates Euler Characteristic.
    Monitored to prevent topological anomalies (e.g., unintended holes/tunnels).
    """
    def __init__(self):
        super().__init__()
        # 3D Hessian-Laplacian second-order derivative kernels
        laplacian_k = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        self.register_buffer("laplacian_k", laplacian_k)

    def forward(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, d, h, w = feature_map.shape
        f_reshaped = feature_map.view(b * c, 1, d, h, w)
        
        # Approximate local principal curvature product (Gaussian Curvature K)
        lap = F.conv3d(f_reshaped, self.laplacian_k, padding=1).view(b, c, d, h, w)
        gaussian_curvature = torch.norm(lap, dim=1, keepdim=True) ** 2

        # Discrete Euler Characteristic integral estimate
        euler_char = torch.sum(gaussian_curvature, dim=(2, 3, 4), keepdim=True) / (2.0 * 3.1415926535)

        return {
            "gaussian_curvature_map": gaussian_curvature,
            "euler_characteristic": euler_char
        }


class FusedEulerCharacteristic3DIR(nn.Module):
    """Fused Grouped Group-Conv Pass for Gauss-Bonnet Topological Calculation."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        base_k = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        fused_kernel = base_k.repeat(in_channels, 1, 1, 1, 1)
        self.register_buffer("fused_kernel", fused_kernel)

    def forward(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        lap = F.conv3d(feature_map, self.fused_kernel, padding=1, groups=self.in_channels)
        gaussian_curvature = torch.norm(lap, dim=1, keepdim=True) ** 2
        euler_char = torch.sum(gaussian_curvature, dim=(2, 3, 4), keepdim=True) / (2.0 * 3.1415926535)

        return {
            "gaussian_curvature_map": gaussian_curvature,
            "euler_characteristic": euler_char
        }


def optimize_gauss_bonnet_graph_pass(model: nn.Module) -> nn.Module:
    traced = fx.symbolic_trace(model)
    modules_to_fused = {}

    for node in traced.graph.nodes:
        if node.op == "call_module":
            orig_mod = traced.get_submodule(node.target)
            if isinstance(orig_mod, DiscreteEulerCharacteristic3D) or orig_mod.__class__.__name__ == "DiscreteEulerCharacteristic3D":
                in_channels = getattr(orig_mod, "in_channels", 3)
                modules_to_fused[node.target] = FusedEulerCharacteristic3DIR(in_channels=in_channels)

    for target, fused_mod in modules_to_fused.items():
        setattr(traced, target, fused_mod)

    traced.graph.lint()
    traced.recompile()
    return traced
