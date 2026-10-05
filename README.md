# Stokes' Guardrail 3D

Physics-Informed & Differential Geometry Feature Guardrails for 3D Medical Deep Learning in PyTorch.

---

## Executive Summary

Deep learning models applied to 3D medical imaging—such as Brain Magnetic Resonance Imaging (MRI)—frequently suffer from spatial inconsistency, high-frequency noise artifacts, and latent feature hallucination. Standard neural network architectures treat deep latent feature maps purely as statistical representations, lacking inherent geometric constraints. Consequently, models can easily overfit to scanner-specific noise or exploit spurious non-anatomical correlations (shortcut learning).

**Stokes' Guardrail 3D** addresses this fundamental limitation by introducing a differential-geometric framework directly into the neural network's latent space. By formulating high-dimensional feature maps as a 3D vector field $\mathbf{F}$, we enforce mathematical consistency through **Stokes' Theorem**:

$$\oint_{\partial \Sigma} \mathbf{F} \cdot d\mathbf{r} = \iint_{\Sigma} (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

By constraining the spatial curl ($\nabla \times \mathbf{F}$) and regularizing directional alignment relative to raw anatomical image gradients ($\nabla I$), this library provides a deterministic, differentiable guardrail that penalizes local turbulence while preserving real tissue boundaries.

---

## Primary Objectives

1. **Latent Feature Regularization:** Enforce smooth, mathematically consistent feature fields across 3D volumetric representations without sacrificing structural detail.
2. **Deterministic & Differentiable Operations:** Provide CPU/GPU-accelerated discrete spatial operators (3D discrete curl and directional gradients) built natively on PyTorch primitives (`torch.nn.functional.conv3d`).
3. **Anatomical Boundary Preservation:** Align feature vector fields with physical anatomical intensity gradients to ensure regularizations do not oversmooth pathological lesions or brain tissue boundaries.
4. **Model Invariance & Generalization:** Improve out-of-distribution robustness across heterogeneous MRI scanner domains (e.g., GE, Siemens, Philips) by penalizing non-physical feature fluctuations.

---

## Current Project Status & Limitations

> **WARNING: Work in Progress (WIP) / Early Proof of Concept (PoC)**
> This package is currently in early active research and development. The mathematical foundations and tensor operators are functional and pass synthetic unit tests, but the repository is **not yet production-ready or drop-in usable for arbitrary 3D architectures**.

### Known Technical Bottlenecks

1. **Absence of Latent Feature Projection Layers:**
* The current `DiscreteCurl3D` operator expects an explicit 3D vector field tensor of shape `(B, 3, D, H, W)`.
* Standard 3D encoders (e.g., 3D UNet, Swin UNETR, ResNet3D) output latent channels of arbitrary depth $C$ (e.g., 64, 128, 256).
* *In Development:* A generalized `LatentToVectorField3D` module to map $C$-channel feature maps into a bounded 3-dimensional directional field $(\mathcal{F}_x, \mathcal{F}_y, \mathcal{F}_z)$.


2. **Incomplete Helmholtz Decomposition (Divergence Unconstrained):**
* According to Helmholtz Decomposition, any smooth 3D vector field can be resolved into a curl-free component and a divergence-free component:

$$\mathbf{F} = -\nabla \phi + \nabla \times \mathbf{A}$$


* The current loss functions penalize local rotational energy (Curl: $\nabla \times \mathbf{F}$), but do not yet constrain compressional/expansional anomalies (Divergence: $\nabla \cdot \mathbf{F}$).
* *In Development:* `DiscreteDivergence3D` to constrain source-sink feature anomalies.


3. **Pending Validation on Real Clinical Datasets:**
* Current test suites validate tensor shapes, scalar loss outputs, and backpropagation gradients using synthetic normal distributions.
* Empirically validated benchmark results on clinical datasets (e.g., ADNI, BraTS, IXI) are actively being conducted.



---

## Architecture Overview

```text
stokes_guardrail_3d/
├── pyproject.toml               # Package build configurations
├── README.md                    # Project documentation
├── stokes_guardrail_3d/
│   ├── __init__.py
│   ├── ops/
│   │   ├── __init__.py
│   │   └── curl.py              # 3D Discrete Curl Operator (Central Difference)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── stokes.py            # L_stokes (Stokes' Guardrail Loss)
│   │   ├── alignment.py         # L_align (Anatomical Alignment Loss)
│   │   └── total.py             # L_total (Multi-Task Objective)
│   └── theorems/
│       ├── __init__.py
│       └── README.md            # Advanced Mathematical Roadmap
└── tests/
    └── test_guardrail.py        # Synthetic shape & gradient assertions

```

### Core Components

* **`ops/curl.py` (`DiscreteCurl3D`):** Implements a 3D Sobel-based central difference scheme to compute spatial curl $(\nabla \times \mathbf{F})_x, (\nabla \times \mathbf{F})_y, (\nabla \times \mathbf{F})_z$ efficiently via 3D group convolutions.
* **`losses/stokes.py` (`StokesGuardrailLoss`):** Calculates the L1/L2 penalty over localized curl magnitudes exceeding a tolerance threshold $\tau$:

$$\mathcal{L}_{stokes} = \frac{1}{V} \sum \max(0, \Vert{}\nabla \times \mathbf{F}\Vert{} - \tau)$$


* **`losses/alignment.py` (`AnatomicalAlignmentLoss`):** Enforces directional cosine similarity between the latent field $\mathbf{F}$ and the input image spatial gradient field $\mathbf{G} = \nabla I$:

$$\mathcal{L}_{align} = 1 - \frac{1}{V} \sum \left( \frac{\mathbf{F} \cdot \mathbf{G}}{\Vert{}\mathbf{F}\Vert{} \Vert{}\mathbf{G}\Vert{} + \epsilon} \right)^2$$


* **`losses/total.py` (`TotalObjectiveLoss`):** Combines standard classification loss with regularizations:

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{stokes} + \lambda_2 \mathcal{L}_{align}$$



---

## Roadmap

* [ ] Implementation of `LatentToVectorField3D` projection block for arbitrary feature channels.
* [ ] Integration of `DiscreteDivergence3D` for full Helmholtz Decomposition regularization.
* [ ] MONAI-compliant dataset loaders and pipeline integration.
* [ ] Extended mathematical modules (`theorems/`):
* Fundamental Theorem of Calculus (Volume Mass Conservation)
* Gauss-Bonnet Curvature Constraints



---

## Installation & Testing

To set up the repository in editable mode and run verification tests:

```bash
# Clone and install dependencies
pip install -e .

# Execute synthetic unit tests
python tests/test_guardrail.py

```
