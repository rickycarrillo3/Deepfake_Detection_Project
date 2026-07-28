import torch
import torch.nn as nn


class GatingNetwork(nn.Module):

    def __init__(
        self,
        image_encoder,
        feature_dim,
        num_experts
    ):
        super().__init__()

        self.encoder = image_encoder


        self.gate_head = nn.Sequential(

            nn.Linear(
                feature_dim,
                256
            ),

            nn.ReLU(inplace=True),


            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),


            nn.Linear(
                128,
                num_experts
            ),

            nn.Sigmoid()
        )


    def forward(self, image):

        features = self.encoder(
            image
        )

        gates = self.gate_head(
            features
        )


        return gates