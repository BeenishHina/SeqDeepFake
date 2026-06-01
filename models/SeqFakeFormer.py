import torch
from torch import nn
import torch.nn.functional as F

from tools.utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer_SECA import build_transformer


# =====================================================
# CBAM ATTENTION MODULE
# =====================================================

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // reduction,
                1,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels // reduction,
                channels,
                1,
                bias=False
            )
        )

        # Spatial attention
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

    def forward(self, x):

        # -----------------------------
        # Channel Attention
        # -----------------------------
        avg_out = self.fc(
            self.avg_pool(x)
        )

        max_out = self.fc(
            self.max_pool(x)
        )

        ca = torch.sigmoid(
            avg_out + max_out
        )

        x = x * ca

        # -----------------------------
        # Spatial Attention
        # -----------------------------
        avg = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        mx, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        sa = torch.sigmoid(
            self.spatial(
                torch.cat(
                    [avg, mx],
                    dim=1
                )
            )
        )

        x = x * sa

        return x


# =====================================================
# MLP TOKEN HEAD
# =====================================================

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers
    ):
        super().__init__()

        self.num_layers = num_layers

        h = [hidden_dim] * (
            num_layers - 1
        )

        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip(
                [input_dim] + h,
                h + [output_dim]
            )
        )

    def forward(self, x):

        for i, layer in enumerate(
            self.layers
        ):

            if i < self.num_layers - 1:
                x = F.relu(
                    layer(x)
                )
            else:
                x = layer(x)

        return x


# =====================================================
# MAIN MODEL
# =====================================================

class SeqFakeFormer(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        hidden_dim,
        vocab_size,
        imgsize
    ):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.imgsize = imgsize

        # -----------------------------------
        # layer2 + layer3 + layer4 fusion
        # ResNet101:
        # layer2 = 512
        # layer3 = 1024
        # layer4 = 2048
        # total = 3584
        # -----------------------------------
        self.fusion_conv = nn.Conv2d(
            3072,
            backbone.num_channels,
            kernel_size=1
        )

        # -----------------------------------
        # CBAM Attention
        # -----------------------------------
        self.cbam = CBAM(
            backbone.num_channels
        )

        # -----------------------------------
        # Project to transformer dim
        # -----------------------------------
        self.input_proj = nn.Conv2d(
            backbone.num_channels,
            hidden_dim,
            kernel_size=1
        )

        # -----------------------------------
        # Output head
        # -----------------------------------
        self.mlp = MLP(
            hidden_dim,
            512,
            vocab_size,
            3
        )

    def forward(
        self,
        samples,
        target,
        target_mask
    ):

        if not isinstance(
            samples,
            NestedTensor
        ):
            samples = nested_tensor_from_tensor_list(
                self.imgsize,
                samples
            )

        # -----------------------------------
        # Backbone outputs
        # layer1 layer2 layer3 layer4
        # -----------------------------------
        features, pos = self.backbone(
            samples
        )

        # -----------------------------------
        # Multi-scale features
        # layer2 + layer3 + layer4
        # -----------------------------------
        src3, _ = features[-2].decompose()
        src4, mask = features[-1].decompose()

        assert mask is not None
        src3 = F.interpolate(
          src3,
          size=src4.shape[-2:],
          mode="bilinear",
          align_corners=False
          )

        # -----------------------------------
        # Multi-scale fusion
        # -----------------------------------
        src = torch.cat(
            [src3, src4],
            dim=1
        )

        src = self.fusion_conv(
            src
        )

        # -----------------------------------
        # CBAM refinement
        # -----------------------------------
        src = self.cbam(
            src
        )

        # -----------------------------------
        # Image size tensor
        # -----------------------------------
        h_w = torch.tensor(
            [self.imgsize, self.imgsize]
        ).repeat(
            src.shape[0],
            1
        ).to(src.device)

        h_w = h_w.unsqueeze(0)

        # -----------------------------------
        # Transformer
        # -----------------------------------
        hs = self.transformer(
            self.input_proj(src),
            mask,
            pos[-1],
            target,
            target_mask,
            h_w
        )

        # -----------------------------------
        # Token prediction
        # -----------------------------------
        out = self.mlp(
            hs.permute(1, 0, 2)
        )

        return out


# =====================================================
# BUILD MODEL
# =====================================================

def build_model(config):

    backbone = build_backbone(
        config
    )

    transformer = build_transformer(
        config
    )

    model = SeqFakeFormer(
        backbone=backbone,
        transformer=transformer,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size,
        imgsize=config.imgsize
    )

    return model
