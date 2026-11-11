"""Person re-ID network following "Bag of Tricks and a Strong Baseline for Deep Person Re-ID"
(Luo et al., 2019): ImageNet ResNet, last stride 1, global average pooling, BNNeck.

Training returns (logits, pre-BN features): cross-entropy uses the logits, the triplet loss
the features. Inference returns L2-normalised post-BN features for cosine matching.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


class ReIDNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 751, pretrained: bool = True,
                 last_stride: int = 1):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, not {backbone!r}")
        # IMAGENET1K_V1 is what the Bag-of-Tricks baseline was trained from; torchvision's
        # "DEFAULT" now resolves to V2 for some ResNets, which would make comparisons unfair.
        net = BACKBONES[backbone](weights="IMAGENET1K_V1" if pretrained else None)
        if last_stride == 1:
            for m in net.layer4[0].modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (1, 1)
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
        self.feat_dim = net.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bnneck = nn.BatchNorm1d(self.feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.config = {"backbone": backbone, "num_classes": num_classes, "last_stride": last_stride}

    def forward(self, x: torch.Tensor):
        feats = self.pool(self.backbone(x)).flatten(1)
        bn = self.bnneck(feats)
        if self.training:
            return self.classifier(bn), feats
        return F.normalize(bn, dim=1)


def load_checkpoint(path, device: str = "cpu") -> tuple[ReIDNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = ReIDNet(meta["backbone"], meta["num_classes"], pretrained=False, last_stride=meta["last_stride"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), meta
