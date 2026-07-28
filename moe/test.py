import torch
import config

from models import (
    GatingNetwork,
    ImageMoE,
    CNNEncoder,
)



def main():
    encoder = CNNEncoder(feature_dim=config.CNN_FEATURE_DIM)

    gating = GatingNetwork(
        image_encoder=encoder,
        feature_dim=config.CNN_FEATURE_DIM,
        num_experts=config.NUM_EXPERTS
    )

    model = ImageMoE(
        gating_network=gating,
        num_experts=config.NUM_EXPERTS,
        hidden_dims=config.HIDDEN_DIMS
    )

    images = torch.randn(8, 3, 224, 224)
    expert_scores = torch.rand(8, 4)
    labels = torch.randint(0, 2, (8, 1)).float()  # matches train.py's labels.unsqueeze(1)

    logits = model(images, expert_scores)
    print("logits shape:", logits.shape)
    assert logits.shape == (8, 1), f"expected (8, 1), got {logits.shape}"

    # The real point of this test: confirm gradients actually reach the
    # gating network despite any hard-threshold step inside it. If this
    # assert fails, the gate is silently disconnected from training.
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    grad_total = sum(
        p.grad.abs().sum().item()
        for p in gating.parameters()
        if p.grad is not None
    )
    print("gating network total gradient magnitude:", grad_total)
    assert grad_total > 0, "GatingNetwork received no gradient -- check for a non-differentiable hard threshold."

    print("OK")


if __name__ == "__main__":
    main()