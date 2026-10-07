"""
Stokes' Guardrail 3D - IR Optimization Pass.
Fuses Latent Projection, 3D Discrete Curl, and 3D Discrete Divergence 
into a single, high-performance execution subgraph using torch.fx.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from typing import Dict, Any

from .projector import LatentToVectorField3D


class FusedStokesGuardrailIR(nn.Module):
    """
    Fused and optimized deployment block. 
    Replaces separate convolution layers with a single highly-parallelized 
    3D group convolution kernel to minimize GPU global memory roundtrips.
    """
    def __init__(self, in_channels: int, delta_spatial: float = 1.0, bound_mode: str = "tanh"):
        super().__init__()
        self.in_channels = in_channels
        self.bound_mode = bound_mode.lower()
        self.delta_spatial = delta_spatial

        # 1. 1x1x1 Projection Weights (from LatentToVectorField3D)
        self.proj = nn.Conv3d(in_channels, 3, kernel_size=1, bias=True)

        # 2. Build Fused Sobel Kernel
        # Base Sobel-like central difference component along x-axis
        base_k = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / (32.0 * delta_spatial)

        kernel_x = base_k.clone()
        kernel_y = base_k.transpose(1, 2)
        kernel_z = base_k.transpose(0, 2)

        # Pack into a single 3-channel group-conv kernel
        # Out_channels=9 (3 directional derivatives per 3 vector components)
        # Shape: (9, 1, 3, 3, 3) for efficient grouped convolution
        fused_kernel = torch.zeros((9, 1, 3, 3, 3), dtype=torch.float32)
        
        # Component X derivatives (dx, dy, dz)
        fused_kernel[0, 0] = kernel_x
        fused_kernel[1, 0] = kernel_y
        fused_kernel[2, 0] = kernel_z
        
        # Component Y derivatives (dx, dy, dz)
        fused_kernel[3, 0] = kernel_x
        fused_kernel[4, 0] = kernel_y
        fused_kernel[5, 0] = kernel_z
        
        # Component Z derivatives (dx, dy, dz)
        fused_kernel[6, 0] = kernel_x
        fused_kernel[7, 0] = kernel_y
        fused_kernel[8, 0] = kernel_z

        self.register_buffer("fused_kernel", fused_kernel)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Step 1: Pointwise Linear Projection (C -> 3)
        field = self.proj(x)
        if self.bound_mode == "tanh":
            field = torch.tanh(field)

        # Step 2: Fused 3D Spatial Derivation Pass (Only 1 Conv3D Call instead of 7!)
        # Using groups=3 maps 3 input channels to 3 private operational slots each
        grads = F.conv3d(field, self.fused_kernel, padding=1, groups=3)

        # Slice out the pre-computed derivatives
        dfx_dx, dfx_dy, dfx_dz = grads[:, 0:1, ...], grads[:, 1:2, ...], grads[:, 2:3, ...]
        dfy_dx, dfy_dy, dfy_dz = grads[:, 3:4, ...], grads[:, 4:5, ...], grads[:, 5:6, ...]
        dfz_dx, dfz_dy, dfz_dz = grads[:, 6:7, ...], grads[:, 7:8, ...], grads[:, 8:9, ...]

        # Step 3: Vector Math Reconstruction
        curl_x = dfz_dy - dfy_dz
        curl_y = dfx_dz - dfz_dx
        curl_z = dfy_dx - dfx_dy
        curl = torch.cat([curl_x, curl_y, curl_z], dim=1)

        divergence = dfx_dx + dfy_dy + dfz_dz

        return {"vector_field": field, "curl": curl, "divergence": divergence}


def optimize_stokes_graph_pass(model: nn.Module) -> nn.Module:
    """
    Symbolic Graph Transformation Pass.
    Traces the network topology, detects separate Stokes components, 
    and hot-swaps them into a single FusedStokesGuardrailIR node.
    """
    # Symbolic tracing of the target runtime graph
    traced = fx.symbolic_trace(model)
    modules_to_fused = {}

    # Scan and match subgraph patterns for substitution
    for node in traced.graph.nodes:
        if node.op == "call_module":
            orig_mod = traced.get_submodule(node.target)
            
            # Robust Type Checking: Directly check class instance or class name
            is_target_module = isinstance(orig_mod, LatentToVectorField3D) or (
                orig_mod.__class__.__name__ == "LatentToVectorField3D"
            )
            
            if is_target_module:
                # Construct the optimal IR replacement
                fused_block = FusedStokesGuardrailIR(
                    in_channels=orig_mod.in_channels,
                    bound_mode=orig_mod.bound_mode
                )
                
                # Copy projection weights if available
                if hasattr(orig_mod, "proj"):
                    fused_block.proj.load_state_dict(orig_mod.proj.state_dict())

                modules_to_fused[node.target] = fused_block

    # Dynamic hot-swapping inside the computational graph
    for target, fused_mod in modules_to_fused.items():
        setattr(traced, target, fused_mod)
        
    traced.graph.lint()
    traced.recompile()
    return traced
