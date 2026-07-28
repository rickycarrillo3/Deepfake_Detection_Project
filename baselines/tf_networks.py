"""Architectures shared by the TrueFake baselines, transcribed from source.

Each class here is a faithful transcription of the code the TrueFake authors
actually trained with, so `load_state_dict(..., strict=True)` succeeds against
the shipped `best.pt`. That strictness is the point: it turns a wrong
transcription into a loud error at load time instead of a plausible-looking
but meaningless score.

Sources
-------
- ImageClassifier / ScoresLayer
    MMLab-unitn/TrueFake-IJCNN25, detector/networks.py — the authors' own
    training code (linked from footnote 4 of the paper). Covers both R50-ND
    (`arch='baseline'`) and their proposed R50-E2P (`arch='nodown'` +
    prototype head).
- MISLNet
    Reconstructed from the checkpoint (see the class docstring); the Stamm-lab
    code for Vahdati et al., CVPRW 2024 is not public.

NPR lives in its own model.py instead of here: its upstream repo is vendored
by setup.sh, so its architecture is imported from the clone rather than
retranscribed.
"""

import torch
import torch.nn as nn
from torchvision import models


class ScoresLayer(nn.Module):
    """Learned-prototype scoring head (TrueFake §IV.6).

    Verbatim from TrueFake-IJCNN25 detector/networks.py. Implements

        s(f) = sgn(sum_k (f_k - c_k)) * ||f - c||^2 / (2 sigma_k^2) + D ln sigma_k

    with `logsigmas` passed through relu (enforcing log sigma_k > 0, the
    triangle-inequality constraint the paper requires for the distance to be a
    valid metric).

    Note the output is a *logit*, not a probability: train.py optimizes it with
    BCEWithLogitsLoss. model.py applies the sigmoid.
    """

    def __init__(self, input_dim, num_centers):
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.zeros(num_centers, input_dim), requires_grad=True)
        self.logsigmas = nn.Parameter(torch.zeros(num_centers), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, self.input_dim, 1, 1)  # [batch, C, 1, 1]

        centers = self.centers[None, :, :, None, None]  # [1, K, C, 1, 1]
        diff = out.unsqueeze(1) - centers  # [batch, K, C, 1, 1]

        sum_diff = torch.sum(diff, dim=2)  # [batch, K, 1, 1]
        sign = torch.sign(sum_diff)

        squared_diff = torch.sum(diff**2, dim=2)  # [batch, K, 1, 1]

        logsigmas = nn.functional.relu(self.logsigmas)
        denominator = 2 * torch.exp(2 * logsigmas)
        part1 = (sign * squared_diff) / denominator.view(1, -1, 1, 1)

        part2 = self.input_dim * logsigmas
        part2 = part2.view(1, -1, 1, 1)

        scores = part1 + part2
        return scores.sum(dim=(1, 2, 3)).view(-1, 1)  # [batch, 1]


class ImageClassifier(nn.Module):
    """TrueFake's ResNet-50 detector, both variants.

    Verbatim from TrueFake-IJCNN25 detector/networks.py, with the `settings`
    argparse namespace replaced by plain keyword arguments.

    arch='nodown' removes the stride from the first conv (7x7 stride 1 instead
    of stride 2), the modification Corvi et al. introduced so low-level
    synthesis traces survive the stem. Note this makes inference cost scale
    with the *native* image area — there is no resize at test time.

    The ImageNet-pretrained weights requested in the constructor are
    immediately overwritten by the checkpoint; they are kept only so the module
    tree matches the one that was trained.
    """

    def __init__(self, arch="nodown", freeze=False, prototype=False, num_centers=1):
        super().__init__()
        if arch == "baseline":
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

        elif arch == "nodown":
            self.backbone = models.resnet50(weights=None)

            # Replace first conv layer to avoid downsampling
            new_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            new_conv.weight = nn.Parameter(self.backbone.conv1.weight)
            self.backbone.conv1 = new_conv

            self.backbone.fc = nn.Sequential(
                nn.Linear(self.backbone.fc.in_features, 128), nn.Dropout(0.5)
            )

        else:
            raise NotImplementedError(f"Model not recognized: {arch!r}")

        self.prototype = prototype
        if self.prototype:
            self.proto = ScoresLayer(
                input_dim=self.backbone.fc[0].out_features, num_centers=num_centers
            )

    def forward(self, x):
        x = self.backbone(x)
        if self.prototype:
            x = self.proto(x)
        return x


class ConstrainedConv2d(nn.Conv2d):
    """Bayar & Stamm's constrained convolutional layer (TIFS 2018).

    The kernel is reprojected so each filter's non-center taps sum to +1 and
    the center tap is -1, forcing it to act as a learned prediction-error
    (high-pass) filter rather than a content detector — the property the
    TrueFake authors cite as MISLNet's reason for including it.

    The reprojection is applied **inside forward**, not baked into the stored
    weights. This is forced by the checkpoint rather than assumed: its
    `constrained_conv.weight` is plainly *un*constrained (center taps ~0.15,
    per-filter sums ~-0.26..0.22, neither -1 nor 0). Had the constraint been
    reapplied after each optimizer step — the other common implementation —
    the saved weights would already satisfy it. So the trained network must
    have normalized on the fly, and inference has to do the same; skipping it
    would still produce scores, just ones the trained filters never saw.

    `one_middle` is a checkpoint-shipped buffer marking the center tap, so the
    layout comes from the checkpoint rather than being hardcoded here.
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False
        )
        self.register_buffer("one_middle", torch.zeros(kernel_size * kernel_size))
        self.one_middle[kernel_size * kernel_size // 2] = 1.0

    def constrained_weight(self):
        w = self.weight.view(self.weight.size(0), self.weight.size(1), -1)
        w = w * (1 - self.one_middle)  # zero the center tap
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)  # remaining taps sum to +1
        w = w - self.one_middle  # center tap := -1
        return w.view_as(self.weight)

    def forward(self, x):
        return nn.functional.conv2d(
            x, self.constrained_weight(), self.bias, self.stride, self.padding
        )


class MISLNet(nn.Module):
    """MISLNet as retrained on TrueFake (paper ref [39], Vahdati et al. CVPRW'24).

    The Stamm-lab code is not public, so this is reconstructed from the
    checkpoint's own tensor shapes. Every shape below is forced by the
    checkpoint and verified by a strict load plus a forward pass (see
    verify_checkpoints.py):

        constrained_conv       (6, 3, 5, 5)      + one_middle (25,)
        conv_blocks.0.conv     (96, 6, 7, 7)     stride 2
        conv_blocks.1.conv     (64, 96, 5, 5)
        conv_blocks.2.conv     (64, 64, 5, 5)
        conv_blocks.3.conv     (128, 64, 1, 1)
        fc_blocks.0.fc         (200, 4608)
        fc_blocks.1.fc         (200, 200)
        output                 (2, 200)

    The 4608 pins the input size. With TensorFlow-style SAME padding and three
    3x3/stride-2 pools (after conv blocks 0, 1 and 2 — none after the 1x1
    block), a 96x96 input gives 96 -> 48 -> 24 -> 12 -> 6, i.e.
    128 x 6 x 6 = 4608. That is the only clean solution, and it agrees with the
    paper: MISLNet was trained with R50-E2P's augmentation, which ends in
    RandomCrop(96).

    `flatten_index_permutation` is a 4-element buffer shipped *in* the
    checkpoint. It records the dimension order used when flattening before the
    first FC — an artifact of the original TensorFlow (NHWC) implementation.
    It is applied rather than assumed, so the FC sees its features in the
    order it was trained on. Getting this wrong would still produce scores,
    just meaningless ones.

    Output is 2-class (real, fake), so model.py takes softmax()[1] rather than
    a sigmoid.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.register_buffer("flatten_index_permutation", torch.tensor([0, 1, 2, 3]))

        self.constrained_conv = ConstrainedConv2d(3, 6, kernel_size=5)

        def conv_block(cin, cout, k, stride=1):
            block = nn.Module()
            block.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=True)
            block.bn = nn.BatchNorm2d(cout)
            return block

        self.conv_blocks = nn.ModuleList(
            [
                conv_block(6, 96, 7, stride=2),
                conv_block(96, 64, 5),
                conv_block(64, 64, 5),
                conv_block(64, 128, 1),
            ]
        )
        # SAME-style 3x3/stride-2 pooling after the first three blocks only.
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        def fc_block(fin, fout):
            block = nn.Module()
            block.fc = nn.Linear(fin, fout)
            return block

        self.fc_blocks = nn.ModuleList([fc_block(4608, 200), fc_block(200, 200)])
        self.output = nn.Linear(200, num_classes)

    def forward(self, x):
        # Constrained conv has no padding (valid), so pad to keep 96 -> 96.
        x = nn.functional.pad(x, (2, 2, 2, 2))
        x = self.constrained_conv(x)

        for i, block in enumerate(self.conv_blocks):
            x = torch.tanh(block.bn(block.conv(x)))
            if i < 3:
                x = self.pool(x)

        x = x.permute(*self.flatten_index_permutation.tolist()).flatten(1)

        for block in self.fc_blocks:
            x = torch.tanh(block.fc(x))

        return self.output(x)
