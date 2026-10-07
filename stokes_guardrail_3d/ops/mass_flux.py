"""
3D Discrete Mass Flux & Conservation Operator.
Computes net surface flux via Discrete Divergence Theorem:
    Flux = integral_V (nabla . F) dV = integral_d V (F . n) dS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Dict
from .divergence import DiscreteDivergence3D


class DiscreteVolumeFlux3D(nn.Module):
    """
    Computes local and global mass divergence/flux across 3D latent volumes.
    Identifies unnatural mass creation (sources) or mass disappearance (sinks).
    """
    def __init__(self, delta_spatial: float = 1.0):
        super().__init__()
        self.div_op = DiscreteDivergence3D(delta_spatial=delta_spatial)

    def forward(self, vector_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Local divergence density map: (B, 1, D, H, W)
        div_density = self.div_op(vector_field)
        
        # Total net volume flux via spatial integration (sum over spatial dims)
        net_flux = torch.sum(div_density, dim=(2, 3, 4), keepdim=True)
        
        return {
            "divergence_density": div_density,
            "net_volume_flux": net_flux
        }


class FusedVolumeFlux3DIR(nn.Module):
    """
    Fused IR Node for Mass Flux Calculation.
    Executes a single 3D group convolution kernel for divergence derivation.
    """
    def __init__(self, delta_spatial: float = 1.0):
        super().__init__()
        base_k = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / (32.0 * delta_spatial)

        fused_kernel = torch.zeros((3, 1, 3, 3, 3), dtype=torch.float32)
        fused_kernel[0, 0] = base_k
        fused_kernel[1, 0] = base_k.transpose(1, 2)
        fused_kernel[2, 0] = base_k.transpose(0, 2)

        self.register_buffer("fused_kernel", fused_kernel)

    def forward(self, vector_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        grads = F.conv3d(vector_field, self.fused_kernel, padding=1, groups=3)
        div_density = grads[:, 0:1, ...] + grads[:, 1:2, ...] + grads[:, 2:3, ...]
        net_flux = torch.sum(div_density, dim=(2, 3, 4), keepdim=True)

        return {
            "divergence_density": div_density,
            "net_volume_flux": net_flux
        }


def optimize_flux_graph_pass(model: nn.Module) -> nn.Module:
    traced = fx.symbolic_trace(model)
    modules_to_fused = {}

    for node in traced.graph.nodes:
        if node.op == "call_module":
            orig_mod = traced.get_submodule(node.target)
            if isinstance(orig_mod, DiscreteVolumeFlux3D) or orig_mod.__class__.__name__ == "DiscreteVolumeFlux3D":
                delta = getattr(orig_mod.div_op, "delta_spatial", 1.0)
                modules_to_fused[node.target] = FusedVolumeFlux3DIR(delta_spatial=delta)

    for target, fused_mod in modules_to_fused.items():
        setattr(traced, target, fused_mod)

    traced.graph.lint()
    traced.recompile()
    return traced
