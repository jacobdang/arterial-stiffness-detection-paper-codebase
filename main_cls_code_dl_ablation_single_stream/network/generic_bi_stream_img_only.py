import torch
from torch import nn
import math
import timm
from .utils import ResNormLayer

# SingleStreamWrapper (reuses GenericBiStreamImgOnly internals)
import torch.nn as nn, torch
class SingleStreamWrapper(nn.Module):
    def __init__(self, network_name, head_dim, eye='left'):
        super().__init__()
        assert eye in ('left','right')
        self.eye = 0 if eye=='left' else 1
        self.wrapped = GenericBiStreamImgOnly(network_name, head_dim)
    def forward(self, inputs):
        img = inputs[:, self.eye, ...]
        img_stack = img.unsqueeze(1)
        out_score, out_feat = self.wrapped(img_stack)
        return out_score, out_feat



class GenericBiStreamImgOnly(nn.Module):
    def __init__(self, network_name, head_dim):
        super().__init__()
        backbone = timm.create_model(network_name, pretrained=True, features_only=True)
        out_channels = backbone.feature_info[-1]['num_chs']
        self.backbone = backbone
        self.head = nn.Sequential(nn.Conv2d(out_channels, head_dim, kernel_size=1), nn.BatchNorm2d(head_dim),
                                  nn.SiLU(inplace=True))
        num_squeezed_channels = max(1, int(0.5 * head_dim))
        self.combined_meta_attention = nn.Sequential(nn.Linear(head_dim, num_squeezed_channels),
                                                     nn.SiLU(inplace=True), nn.BatchNorm1d(num_squeezed_channels),
                                                     nn.Linear(num_squeezed_channels, head_dim),
                                                     nn.Sigmoid())
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(head_dim, 1)

    def forward(self, inputs):
        bs = inputs.size(0)
        x_list = list()
        for idx in range(inputs.size(1)):
            x_list.append(self.avg_pooling(self.head(self.backbone(inputs[:, idx, ...])[-1])).view(bs, -1))
        x = torch.stack(x_list)
        x = torch.mean(x, dim=0)
        att = self.combined_meta_attention(x)
        final_feat = x * att
        final_score = self.fc(final_feat)
        return final_score, final_feat