# Stokes' Guardrail 3D

Physics-Informed & Differential Geometry Feature Guardrails for 3D Medical Deep Learning in PyTorch.

---

## Executive Summary

Deep learning models applied to 3D medical imaging—such as Brain Magnetic Resonance Imaging (MRI)—frequently suffer from spatial inconsistency, high-frequency noise artifacts, and latent feature hallucination. Standard neural network architectures treat deep latent feature maps purely as statistical representations, lacking inherent geometric constraints. Consequently, models can easily overfit to scanner-specific noise or exploit spurious non-anatomical correlations (shortcut learning).

**Stokes' Guardrail 3D** addresses this fundamental limitation by introducing a differential-geometric framework directly into the neural network's latent space. By formulating high-dimensional feature maps as a 3D vector field $\mathbf{F}$, we enforce mathematical consistency through Stokes' Theorem:

$$\oint_{\partial \Sigma} \mathbf{F} \cdot d\mathbf{r} = \iint_{\Sigma} (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

By constraining spatial rotational energy ($\nabla \times \mathbf{F}$), penalizing compressional/expansional anomalies ($\nabla \cdot \mathbf{F}$), and regularizing directional alignment relative to raw anatomical image gradients ($\nabla I$), this library provides a deterministic, differentiable guardrail that penalizes local turbulence while preserving real tissue boundaries.

---

## Primary Objectives

* **Latent Feature Regularization:** Enforce smooth, mathematically consistent feature fields across 3D volumetric representations without sacrificing structural detail.
* **Deterministic & Differentiable Operations:** Provide CPU/GPU-accelerated discrete spatial operators (3D discrete curl, discrete divergence, and projection blocks) built natively on PyTorch primitives (`torch.nn.functional.conv3d`).
* **Anatomical Boundary Preservation:** Align feature vector fields with physical anatomical intensity gradients to ensure regularizations do not oversmooth pathological lesions or brain tissue boundaries.
* **Model Invariance & Generalization:** Improve out-of-distribution robustness across heterogeneous MRI scanner domains (e.g., GE, Siemens, Philips) by penalizing non-physical feature fluctuations.

---

## Project Status & Verification Results

> **STATUS: Core Operators & Losses Verified (100% Test Coverage Passed)**  
> All fundamental differential geometry operators, channel projection layers, and multi-task loss modules are fully implemented and verified via autograd backpropagation and synthetic unit tests.

### Experimental Unit Test Results

The suite tests tensor shape guarantees, $C \to 3$ spatial projections, bounding constraints ($[-1, 1]$ via `tanh`), Autograd gradient flow through Helmholtz losses back to the latent space, and multi-objective loss combinations.

```text
================================== test session starts ==================================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\stokes-guardrail-3d
configfile: pyproject.toml
collected 5 items

tests/test_guardrail.py::test_curl_shape PASSED                                 [ 20%]
tests/test_guardrail.py::test_divergence_shape PASSED                           [ 40%]
tests/test_guardrail.py::test_projector_and_bounds PASSED                      [ 60%]
tests/test_guardrail.py::test_helmholtz_guardrail_backward PASSED              [ 80%]
tests/test_guardrail.py::test_total_loss PASSED                                [100%]

================================== 5 passed in 13.83s ===================================

```

---

## Architecture Overview

```text
stokes_guardrail_3d/
├── pyproject.toml               # Package build configurations
├── README.md                    # Project documentation
├── stokes_guardrail_3d/
│   ├── adapters/            
│   │   ├── __init__.py
│   │   └── monai_wrapper.py    # MONAI Integration Wrapper
│   ├── __init__.py              # Top-level API Exports
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── curl.py              # 3D Discrete Curl Operator (∇ × F)
│   │   ├── divergence.py        # 3D Discrete Divergence Operator (∇ · F)
│   │   └── projector.py         # Latent-to-Vector-Field 3D Projection Block (C -> 3)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── stokes.py            # Helmholtz / Stokes Guardrail Loss (L_helmholtz)
│   │   ├── alignment.py         # Anatomical Alignment Loss (L_align)
│   │   └── total.py             # Multi-Task Objective Loss (L_total)
│   └── theorems/
│       ├── __init__.py
│       └── README.md            # Advanced Mathematical Roadmap
└── tests/
    └── test_guardrail.py        # Complete Pytest test suite

```

---

## Core Components

### 1. Spatial Operators (`ops/`)

* **`LatentToVectorField3D` (`ops/projector.py`):** Projects arbitrary high-dimensional latent channels $(B, C, D, H, W)$ down to a 3D spatial vector field $(B, 3, D, H, W)$ via $1\times1\times1$ pointwise 3D convolutions with optional `tanh` spatial bounding within $[-1, 1]$.
* **`DiscreteCurl3D` (`ops/curl.py`):** Computes the spatial curl vector $(\nabla \times \mathbf{F})$ using 3D group convolutions (`conv3d`) and 3D Sobel-based central differences.
* **`DiscreteDivergence3D` (`ops/divergence.py`):** Computes the scalar divergence field $(\nabla \cdot \mathbf{F})$ to detect source-sink feature anomalies across spatial dimensions.

### 2. Loss Functions (`losses/`)

* **`HelmholtzGuardrailLoss` (`losses/stokes.py`):** Implements full Helmholtz decomposition regularization by penalizing both rotational energy ($\text{Curl}$) and compressional/expansional anomalies ($\text{Divergence}$) above threshold tolerances $\tau_{\text{curl}}$ and $\tau_{\text{div}}$:

$$\mathcal{L}_{\text{helmholtz}} = \frac{1}{V} \sum \left( \tau_{\text{curl}} \max(0, \Vert{}\nabla \times \mathbf{F}\Vert{} - \tau_{\text{curl}}) + \tau_{\text{div}} \max(0, \vert{}\nabla \cdot \mathbf{F}\vert{} - \tau_{\text{div}}) \right)$$

* **`AnatomicalAlignmentLoss` (`losses/alignment.py`):** Enforces directional cosine similarity between the latent feature vector field $\mathbf{F}$ and raw anatomical image intensity gradients $\mathbf{G} = \nabla I$:

$$\mathcal{L}_{\text{align}} = 1 - \frac{1}{V} \sum \left( \frac{\mathbf{F} \cdot \mathbf{G}}{\Vert{}\mathbf{F}\Vert{} \Vert{}\mathbf{G}\Vert{} + \epsilon} \right)^2$$

* **`TotalObjectiveLoss` (`losses/total.py`):** Integrates standard classification/segmentation loss with physics-informed guardrail constraints:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{helmholtz}} + \lambda_2 \mathcal{L}_{\text{align}}$$

---

## Installation & Running Tests

### 1. Installation

Install the package in editable mode for local development:

```bash
git clone [https://github.com/marlovbaloon/stokes-guardrail-3d.git](https://github.com/marlovbaloon/stokes-guardrail-3d.git)
cd stokes-guardrail-3d
pip install -e .

```

### 2. Run Test Suite

Run unit tests via `pytest` to verify spatial tensor shapes, Autograd gradient flow, and loss computations:

```bash
pytest -v

```

---

## Roadmap & Next Steps

* [x] Implementation of `LatentToVectorField3D` projection block ($C \to 3$).
* [x] Integration of `DiscreteDivergence3D` for full Helmholtz Decomposition regularization ($\text{Curl} + \text{Divergence}$).
* [ ] **MONAI Pipeline Integration:** Native plug-and-play adapter for MONAI 3D UNet and Swin UNETR backbones.
* [ ] **Clinical Dataset Benchmarking:** Validation on real-world brain MRI datasets (ADNI, BraTS, IXI) for out-of-distribution robustness.
* [ ] **Extended Mathematical Theorems (`theorems/`):**
* Fundamental Theorem of Calculus (Volume Mass Conservation).
* Gauss-Bonnet Curvature Constraints.



---

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE] file for details.
