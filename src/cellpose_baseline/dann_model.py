#!/usr/bin/env python3
"""Clean DANN-ResNet50 mitosis baseline (no domain-adversarial heads).

Backbone: torchvision ResNet50 (3-channel RGB by default).
Features: layer2/3/4 global-average-pooled and concatenated (512+1024+2048=3584).
Head: Flatten -> Linear(3584,512) -> ReLU -> Dropout(0.5) -> Linear(512,2).

There is no GradientReversal layer and no domain classifier. Weights may be
initialized from either a torchvision ResNet50 checkpoint (ImageNet v1) or a
checkpoint that still contains the old domain-classifier branches; the latter
keys are simply ignored by ``load_pretrained_dann(..., strict=False)``.
"""
import torch
import torch.nn as nn
from torchvision import models

TV_TO_MODEL = {
    'conv1': 'stem.0',
    'bn1': 'stem.1',
    'layer1': 'stem.4',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
}


class DannResNet50(nn.Module):
    def __init__(self, num_classes=2, in_chans=3):
        super().__init__()
        self.in_chans = in_chans
        backbone = models.resnet50(weights=None)

        conv1 = backbone.conv1
        if in_chans != 3:
            new_conv = nn.Conv2d(in_chans, conv1.out_channels,
                                 kernel_size=conv1.kernel_size,
                                 stride=conv1.stride,
                                 padding=conv1.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = conv1.weight
                if in_chans > 3:
                    new_conv.weight[:, 3:] = 0.0
            stem_conv = new_conv
        else:
            stem_conv = conv1

        self.stem = nn.Sequential(
            stem_conv,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveAvgPool2d(1)

        self.mitosis_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 + 1024 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _features(self, x):
        x = self.stem(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        f2 = self.pool2(x2).flatten(1)
        f3 = self.pool3(x3).flatten(1)
        f4 = self.pool4(x4).flatten(1)
        return torch.cat([f2, f3, f4], dim=1)

    def forward(self, x):
        return self.mitosis_classifier(self._features(x))

    def predict_only(self, x):
        return self.forward(x)

    def get_features(self, x):
        with torch.no_grad():
            return self._features(x)


def _map_torchvision(sd):
    mapped = {}
    for k, v in sd.items():
        top = k.split('.')[0]
        if top in TV_TO_MODEL:
            mapped[TV_TO_MODEL[top] + k[len(top):]] = v
    return mapped


def load_pretrained_dann(model, checkpoint):
    """Load torchvision ResNet50 or legacy DANN weights.

    - torchvision checkpoint (contains ``conv1.weight``): map backbone keys.
    - otherwise: treat as a model state dict; unknown keys (old
      ``domain_classifiers.*`` / ``grls.*``) are ignored.
    - if the checkpoint stem has 3 channels but the model uses more, copy the
      first three channels and zero the rest.
    """
    sd = torch.load(checkpoint, map_location='cpu')
    if 'conv1.weight' in sd:
        sd = _map_torchvision(sd)

    key = 'stem.0.weight'
    if key in sd and sd[key].shape != model.stem[0].weight.shape:
        w = sd[key]
        new = model.stem[0].weight.detach().clone()
        c = min(w.shape[1], new.shape[1])
        new[:, :c] = w[:, :c]
        sd[key] = new

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected
