"""Bilateral TinyNet-C architecture for the released image checkpoints."""

from __future__ import annotations


def _safe_backbone(network_name: str, image_size: int, pretrained: bool):
    import timm

    try:
        return timm.create_model(
            network_name,
            pretrained=pretrained,
            features_only=True,
            img_size=image_size,
        )
    except TypeError:
        # timm 0.6.7 TinyNet does not require the explicit image-size keyword,
        # while some other supported backbones accept it.
        return timm.create_model(
            network_name,
            pretrained=pretrained,
            features_only=True,
        )


def build_tinynet_c_classifier(pretrained: bool = False):
    """Construct the exact bilateral TinyNet-C epoch-19 inference graph."""

    import torch
    from torch import nn

    class BilateralTinyNetC(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            head_dim = 2048
            backbone = _safe_backbone("tinynet_c", 384, pretrained=pretrained)
            out_channels = int(backbone.feature_info[-1]["num_chs"])
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Conv2d(out_channels, head_dim, kernel_size=1),
                nn.BatchNorm2d(head_dim),
                nn.SiLU(inplace=True),
            )
            squeezed_channels = max(1, int(0.5 * head_dim))
            self.combined_meta_attention = nn.Sequential(
                nn.Linear(head_dim, squeezed_channels),
                nn.SiLU(inplace=True),
                nn.BatchNorm1d(squeezed_channels),
                nn.Linear(squeezed_channels, head_dim),
                nn.Sigmoid(),
            )
            self.avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(head_dim, 1)

        def forward(self, inputs):
            batch_size = inputs.size(0)
            eye_features = []
            for eye_index in range(inputs.size(1)):
                final_map = self.backbone(inputs[:, eye_index, ...])[-1]
                pooled = self.avg_pooling(self.head(final_map)).view(batch_size, -1)
                eye_features.append(pooled)
            mean_eye_feature = torch.mean(torch.stack(eye_features), dim=0)
            attention = self.combined_meta_attention(mean_eye_feature)
            final_feature = mean_eye_feature * attention
            return self.fc(final_feature), final_feature

    return BilateralTinyNetC()

