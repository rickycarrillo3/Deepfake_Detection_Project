import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Simple CNN image encoder for the gating network.

    Input:
        (B,3,H,W)

    Output:
        (B, feature_dim)
    """

    def __init__(
        self,
        feature_dim=512
    ):
        super().__init__()


        self.features = nn.Sequential(

            # 224x224 -> 112x112
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),


            # 112x112 -> 56x56
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            # 56x56 -> 28x28
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),


            # 28x28 -> 14x14
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.AdaptiveAvgPool2d(
                (1,1)
            )
        )


        self.projection = nn.Linear(
            256,
            feature_dim
        )


    def forward(self,x):

        x = self.features(x)

        # (B,256,1,1)
        x = torch.flatten(
            x,
            1
        )

        # (B,feature_dim)
        x = self.projection(
            x
        )

        return x