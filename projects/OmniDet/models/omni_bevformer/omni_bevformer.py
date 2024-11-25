


import torch
import torch.nn as nn


class OmniBEVFormer(nn.Module):
    def __init__(self, cfg):
        super(OmniBEVFormer, self).__init__()
        self.cfg = cfg
        self.backbone = OmniBEVBackbone(cfg)
        self.neck = OmniBEVNeck(cfg)
        self.head = OmniBEVHead(cfg)

    def forward(self, data_dict):
        x = self.backbone(data_dict)
        x = self.neck(x)
        x = self.head(x)
        return x