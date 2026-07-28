import torch.nn as nn

from .combiner import ExpertCombiner



class ImageMoE(nn.Module):

    def __init__(
        self,
        gating_network,
        num_experts,
        hidden_dims=(32,16)
    ):
        super().__init__()


        self.gating = gating_network


        self.combiner = ExpertCombiner(
            num_experts=num_experts,
            hidden_dims=hidden_dims
        )



    def forward(
        self,
        image,
        expert_scores
    ):


        # Image -> gate values
        gates = self.gating(
            image
        )


        # Expert scores + gates -> prediction
        logits = self.combiner(
            expert_scores,
            gates
        )


        return logits