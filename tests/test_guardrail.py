import torch
from stokes_guardrail_3d import TotalObjectiveLoss, DiscreteCurl3D

def test_curl_shape():
    curl_op = DiscreteCurl3D()
    x = torch.randn(2, 3, 16, 16, 16)
    out = curl_op(x)
    assert out.shape == (2, 3, 16, 16, 16)

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
    test_total_loss()
    print("All basic setup tests passed successfully!")
