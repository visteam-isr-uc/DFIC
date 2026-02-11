
import torch.nn as nn
import torchvision
import torch
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
from torch.nn import functional as F
from torch.nn import init

import os
import cv2
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt



class SEAttention(nn.Module):

    def __init__(self, channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
                                             

class ICAO_DEEPLAB(nn.Module):
    def __init__(self, n_maps = 8, n_reqs = 19):
        super(ICAO_DEEPLAB, self).__init__()
        
        deep_model_with_weights = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)

        self.n_reqs = n_reqs
        self.n_maps = n_maps

        self.backbone = deep_model_with_weights.backbone
        self.classifier = deep_model_with_weights.classifier
        self.classifier[4] = nn.Conv2d(256, n_maps, kernel_size=(1, 1), stride=(1, 1))   


        self.squeeze4cat = nn.ModuleList([nn.Sequential(nn.Conv2d(960, 20, kernel_size=(1, 1)),
                                                        nn.BatchNorm2d(20),
                                                        nn.ReLU(),
                                                        nn.MaxPool2d(4)) for i in range(n_maps)])

        self.final_classifier = nn.Sequential(SEAttention(20*n_maps),
                                                        nn.Conv2d(20*n_maps, n_reqs, kernel_size=(8, 8)))




        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        
        x_small = features["out"]
        x_small = self.classifier(x_small)
        x_small_out = F.interpolate(x_small, size=input_shape, mode="bilinear", align_corners=False)

        fused_map_feats_list = []
        for i in range(self.n_maps):
            spacial_att_map = x_small[:, i, :, :].unsqueeze(1)
            spacial_att_feats = features["out"] * F.sigmoid(spacial_att_map).repeat(1, 960, 1, 1)
            fused_map_feats_list.append(self.squeeze4cat[i](spacial_att_feats))

        fused_map_feats = torch.cat(fused_map_feats_list, dim=1)

        final_logits = torch.flatten(self.final_classifier(fused_map_feats), 1)
    
        return x_small_out, final_logits