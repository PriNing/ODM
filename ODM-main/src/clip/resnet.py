import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from torchvision.models._utils import IntermediateLayerGetter

from src.utils.nested_tensor import NestedTensor
# from model.block import FrozenBatchNorm2d

class ResNet(nn.Module):
    def __init__(self, 
                train_backbone: bool,
                return_interm_layers: bool,
                dilation: bool,
                freeze_bn: bool):
        super(ResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        backbone = getattr(torchvision.models, 'resnet50')(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=norm_layer)
        
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        self.input_proj = nn.Conv2d(self.num_channels, 256, kernel_size=1)

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        return self.input_proj(xs['0'])