import torch
import torch.nn as nn
from stokes_guardrail_3d import (
    DiscreteCurl3D,
    DiscreteDivergence3D,
    HelmholtzGuardrailLoss,
    LatentToVectorField3D,
    TotalObjectiveLoss,
)


def test_curl_shape():
    curl_op = DiscreteCurl3D()
    x = torch.randn(2, 3, 16, 16, 16)
    out = curl_op(x)
    assert out.shape == (2, 3, 16, 16, 16)


def test_divergence_shape():
    div_op = DiscreteDivergence3D()
    x = torch.randn(2, 3, 16, 16, 16)
    out = div_op(x)
    assert out.shape == (2, 1, 16, 16, 16)


def test_projector_and_bounds():
    # Test C-channel projection to 3D Vector Field
    projector = LatentToVectorField3D(in_channels=64, bound_mode="tanh")
    latent = torch.randn(2, 64, 16, 16, 16)
    field = projector(latent)

    assert field.shape == (2, 3, 16, 16, 16)
    # Tanh bounding check: values must stay strictly within [-1, 1]
    assert torch.all(field >= -1.0) and torch.all(field <= 1.0)


def test_helmholtz_guardrail_backward():
    # Verify gradient flow through Projector -> Helmholtz Loss -> Latent Tensor
    latent = torch.randn(2, 32, 12, 12, 12, requires_grad=True)
    projector = LatentToVectorField3D(in_channels=32, bound_mode="tanh")
    guardrail_loss_fn = HelmholtzGuardrailLoss(
        tau_curl=0.5, tau_div=0.5, gamma=1.0
    )

    vector_field = projector(latent)
    loss = guardrail_loss_fn(vector_field)

    assert loss.dim() == 0 and not torch.isnan(loss)

    # Test Backpropagation
    loss.backward()
    assert latent.grad is not None
    assert latent.grad.shape == latent.shape


def test_total_loss():
    loss_fn = TotalObjectiveLoss()
    logits = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    vector_field = torch.randn(2, 3, 16, 16, 16)
    input_image = torch.randn(2, 1, 16, 16, 16)

    loss = loss_fn(logits, targets, vector_field, input_image)
    assert loss.dim() == 0 and loss.item() >= 0


if __name__ == "__main__":
    test_curl_shape()
    test_divergence_shape()
    test_projector_and_bounds()
    test_helmholtz_guardrail_backward()
    test_total_loss()
    print(
        "All Stokes Guardrail 3D tests (including Projector & Helmholtz) passed successfully!"
    )
