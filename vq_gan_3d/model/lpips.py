"""Adapted from https://github.com/SongweiGe/TATS"""

"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple
from torchvision import models
import torch.nn as nn
import torch
import os

# Define cluster storage paths
CACHE_DIR = "/home/andim/.cache/torch/hub/checkpoints"

# Paths for pretrained models
LPIPS_PATH = os.path.join(CACHE_DIR, "vgg.pth")
VGG16_PATH = os.path.join(CACHE_DIR, "vgg16-397923af.pth")


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""
    
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 feature map sizes
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self):
        """Load LPIPS model weights from the cache directory"""
        if not os.path.exists(LPIPS_PATH):
            raise FileNotFoundError(f"LPIPS model not found at {LPIPS_PATH}. Please transfer it manually.")
        
        self.load_state_dict(torch.load(LPIPS_PATH, map_location=torch.device("cpu")), strict=False)
        print(f"Loaded pretrained LPIPS model from {LPIPS_PATH}")

    def forward(self, input, target):
        """Computes LPIPS perceptual distance between two images"""
        in0_input, in1_input = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        return sum(res)


class ScalingLayer(nn.Module):
    """Applies scaling for normalization in LPIPS model"""
    
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer performing a 1x1 convolution"""
    
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    """VGG16 Feature Extractor for LPIPS"""

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        
        if not os.path.exists(VGG16_PATH):
            raise FileNotFoundError(f"VGG-16 model not found at {VGG16_PATH}. Please transfer it manually.")
        
        vgg_pretrained_features = models.vgg16()
        vgg_pretrained_features.load_state_dict(torch.load(VGG16_PATH, map_location=torch.device("cpu")))
        vgg_pretrained_features = vgg_pretrained_features.features

        self.slice1 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        self.slice2 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.slice3 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)])
        self.slice4 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)])
        self.slice5 = torch.nn.Sequential(*[vgg_pretrained_features[x] for x in range(23, 30)])
        self.N_slices = 5

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """Extracts multi-layer VGG16 features"""
        h_relu1_2 = self.slice1(X)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def normalize_tensor(x, eps=1e-10):
    """Normalizes a tensor by its L2 norm"""
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    """Computes spatial average of a tensor"""
    return x.mean([2, 3], keepdim=keepdim)
