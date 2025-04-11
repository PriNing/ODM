

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from torchvision.ops import roi_align
from .position_embedding import PositionEmbeddingLearned, PositionEmbeddingSine
import time
from .resnet import ResNet 

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        FPN的PyTorch实现
        :param in_channels_list: 输入特征图的通道数列表, 例如[256, 512, 1024, 2048]
        :param out_channels: 输出特征图的通道数
        """
        super(FPN, self).__init__()

        # 构建上采样模块
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 构建横向连接模块
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(lateral_conv)

        # 构建特征融合模块
        self.fpn_conv =  nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x):
        # 上采样最深层的特征图
        p5 = self.lateral_convs[-1](x[-1])
        p5_upsampled = p5

        # 从最深层开始向上融合特征
        fpn_feature = p5
        for i in range(len(x)-2, -1, -1):
            lateral_feature = self.lateral_convs[i](x[i])
            if i == len(x) - 2 and len(x) == 5:
                upsampled_feature = p5_upsampled
            else:
                upsampled_feature = self.upsample(p5_upsampled)

            fpn_feature = lateral_feature + upsampled_feature
            p5_upsampled = fpn_feature

        # 对融合后的特征进行卷积
        fpn_output = self.fpn_conv(fpn_feature)

        return fpn_output


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        # self.avgpool = nn.MaxPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                # ("-1", nn.MaxPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    
    def random_masking(self, x, mask_ratio=0.5):
        x = x.permute(1, 0, 2)
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device,
                                                               dtype=int)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        x_masked_add = x_masked_add.permute(1, 0, 2)
        return x_masked_add


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att_maps = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )

        return x, att_maps


class ResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        # self.avgpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5, att_maps = self.attnpool(x4)  
        return [x1, x2, x3, x4, x5], att_maps


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, mask):
        mask = mask.to(device=x.device) if mask is not None else None
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=mask, attn_mask=self.attn_mask)[0]

    def forward(self, x: list):
        x, mask = x
        x = x + self.attention(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return [x, mask]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class ResidualAttentionBlockDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, im_m):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask, key_padding_mask=im_m)

    def forward(self, x: list):
        if len(x) == 4:
            q, k, v, im_m = x
        else:
            q, k, v, im_m, m = x
        
        q_, m = self.attention(q, k, v, im_m)
        q = q + self.ln_1(q_)
        q = q + self.mlp(self.ln_2(q))
        return [q, k, v, im_m, m]


class TransformerDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)



@dataclass
class CLIPVisionCfg:
    name: str = 'ViT'
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    image_resolution: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

def _build_vision_encode(
    embed_dim: int,
    vision_cfg: dict,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None
):
    if vision_cfg['name'] == 'ResNet':
        vision_heads = vision_cfg['vision_width'] * 32 // 64
        visual = ResNet(
                layers=vision_cfg['vision_layers'],
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=vision_cfg['image_resolution'],
                width=vision_cfg['vision_width'],
        )

    else:
        raise "No current encode..."
    
    return visual


class MIM(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channel // 2, output_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(input_channel // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, ratio_boxs):
       
        x = roi_align(x, ratio_boxs, output_size=32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.flatten(1)
        return x

class SegMIM(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.deconv1 =  nn.Sequential(
            nn.Conv2d(input_channel, input_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel // 2),
            nn.ReLU(inplace=True)
        )
        
        self.conv1 = nn.Conv2d(input_channel // 2, input_channel // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channel // 4, output_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(input_channel // 4)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, output_size):
        x = F.interpolate(x, (output_size[0] // 2, output_size[1] // 2))
        x = self.deconv1(x)
        x = F.interpolate(x, output_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x

class SegAug(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.deconv1 =  nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1)
        
    
    def forward(self, x, output_size):
        x = self.deconv1(x)
        x = self.conv1(x)
        x = F.interpolate(x, output_size)

        return x

class oCLIP(nn.Module):
    def __init__(self,
                 first_stage: bool,
                 embed_dim: int,
                 # vision
                
                 vision_cfg: dict,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 transformer_decoder_layers: int,
                 init_logit_scale = None, 
                 init_logit_bias = None,
                 ):
        super().__init__()

        self.context_length = context_length
        self.first_stage = first_stage
        self.visual = _build_vision_encode(embed_dim, vision_cfg)
        self.vision_cfg = vision_cfg
  
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.transformer_heads = transformer_heads
        self.transformer_width = transformer_width

        if not self.first_stage:
            self.transformer_decoder = TransformerDecoder(
                width=embed_dim,
                layers=transformer_decoder_layers,
                heads=transformer_heads,
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final_decoder = LayerNorm(embed_dim)
        self.ln_final = LayerNorm(transformer_width)
        
        if vision_cfg['name'] == 'Swin':
            self.image_projection = nn.Parameter(torch.empty(vision_cfg['embed_dims'] * 8, embed_dim))
            nn.init.normal_(self.image_projection, std=vision_cfg['embed_dims'] * 8 ** -0.5)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        if init_logit_scale is not None:
            self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        if not self.first_stage:

            self.image_pos = nn.Parameter(torch.randn((vision_cfg['image_resolution'] // 32) ** 2 , embed_dim) / embed_dim ** 0.5)
           
            self.fpn_head = FPN([256, 512, 1024, 2048, 512], 256)
            self.mim = SegMIM(256, 1)
        
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if not self.first_stage:
            for block in self.transformer_decoder.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
       
    def build_attention_mask(self):
       
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            return self.visual.patch_embed.projection.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, mask):
        # text: [batch_size, n_ctx] 64 x 77
        batch_size, n_words, n_chars = text.shape
        text = text.reshape(batch_size * n_words, n_chars)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer([x, mask])[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1)
        x = x @ self.text_projection
        x = x.reshape(batch_size, n_words, x.shape[-1])

        return x

    def att_image_to_text(self, encoded_image, encoded_text, image_mask):
        x = encoded_text.permute(1, 0, 2)  # NLD -> LND
        tmp = self.transformer_decoder([encoded_image + self.image_pos[:, None, :].to(encoded_image.dtype), x, x, None])
        
        x = tmp[0]
        m = tmp[4]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_decoder(x).type(self.dtype)
        return x, m
   
    def build_position_embedding(self, type, hidden_dim):
        N_steps = hidden_dim // 2
        if type in ('v2', 'sine'):
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif type in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {type}")
        return position_embedding
    
    def forward(self, image, text, image_mask, output_size=(512, 512)):
        # image_mask = image_mask.flatten(1, 2)
        feature, att_maps = self.encode_image(image)
        encoded_image = feature[-1]
        encoded_texts = self.encode_text(text, None)
        
        logit_scale = self.logit_scale.exp()
        if self.vision_cfg['name'] == 'Swin':
            B, C = encoded_image.shape[:2]
            image_features = encoded_image.reshape(B, C, -1).permute(0, 2, 1) @ self.image_projection
            image_features = torch.mean(image_features, dim=1)
        else:
            image_features = encoded_image[0]
            encoded_image = encoded_image[1:]
            _, B_, C_ = encoded_image.shape
            encoded_image, att_maps2 = self.att_image_to_text(encoded_image, encoded_texts, image_mask)
            feature[-1] = encoded_image.permute(0, 2, 1).reshape(B_, C_, 16, 16)

        text_features = torch.mean(encoded_texts, dim=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if not self.first_stage:
            
            # B, 256, 128, 128
            
            if self.vision_cfg['name'] == 'Swin':
                img_fpn_f = self.fpn_head(feature)
                image_logits = self.mim(img_fpn_f, output_size)
            else:
                img_fpn_f = self.fpn_head(feature)
                image_logits = self.mim(img_fpn_f, output_size)


        if self.training:
            if self.logit_bias is not None:
                return image_features, text_features, image_logits, logit_scale, self.logit_bias
            else:
                return image_features, text_features, image_logits, logit_scale
        else:
            if not self.first_stage:
                return image_logits, att_maps
            else:
                return image_features, text_features, att_maps, logit_scale


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    try:
                        attr.data = attr.data.half()
                    except:
                        pass

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = 256
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = oCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()