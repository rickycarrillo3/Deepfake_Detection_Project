import torch
import torch.nn as nn


class ExpertCombiner(nn.Module):
    """
    Takes pretrained model probabilities,
    applies learned gates, and predicts fake probability.
    """

    def __init__(
        self,
        num_experts,
        hidden_dims=(32, 16)
    ):
        super().__init__()

        layers = []

        input_dim = num_experts

        for hidden_dim in hidden_dims:
            layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

            input_dim = hidden_dim


        # Output logit
        layers.append(
            nn.Linear(
                input_dim,
                1
            )
        )

        self.network = nn.Sequential(
            *layers
        )


    def forward(
        self,
        expert_scores,
        gate_values
    ):

        """
        expert_scores:
            (batch, num_experts)

        gate_values:
            (batch, num_experts)

        """

        # Apply gates
        gated_scores = (
            expert_scores *
            gate_values
        )


        # Predict logit
        logits = self.network(
            gated_scores
        )


        return logits