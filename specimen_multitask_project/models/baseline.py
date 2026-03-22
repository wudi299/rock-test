
from typing import Any, Dict

import torch
import torch.nn as nn

from .backbones import TimmBackbone, masked_backbone_forward
from .fusion import ConcatFusion
from .pooling import MaskedMeanMaxPooling, MaskedMeanPooling
from .tabular_encoder import TabularEncoder


class BaselineMultiTaskModel(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
        main_num_classes: int,
        aux_num_classes: int,
        tabular_input_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]["baseline"]
        pretrained = bool(cfg["model"].get("pretrained", True))
        drop_rate = float(cfg["model"].get("drop_rate", 0.0))
        drop_path_rate = float(cfg["model"].get("drop_path_rate", 0.0))

        self.macro_backbone = TimmBackbone(
            backbone_name=model_cfg["macro_backbone"],
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.micro_backbone = TimmBackbone(
            backbone_name=model_cfg["micro_backbone"],
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        pool_type = str(model_cfg.get("pool_type", "mean")).lower()
        if pool_type == "meanmax":
            self.macro_pool = MaskedMeanMaxPooling()
            self.micro_pool = MaskedMeanMaxPooling()
            pooled_dim_macro = self.macro_backbone.out_dim * 2
            pooled_dim_micro = self.micro_backbone.out_dim * 2
        else:
            self.macro_pool = MaskedMeanPooling()
            self.micro_pool = MaskedMeanPooling()
            pooled_dim_macro = self.macro_backbone.out_dim
            pooled_dim_micro = self.micro_backbone.out_dim

        branch_hidden_dim = int(model_cfg.get("branch_hidden_dim", 256))
        head_dropout = float(model_cfg.get("head_dropout", 0.2))

        self.macro_proj = nn.Sequential(
            nn.Linear(pooled_dim_macro, branch_hidden_dim),
            nn.LayerNorm(branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.micro_proj = nn.Sequential(
            nn.Linear(pooled_dim_micro, branch_hidden_dim),
            nn.LayerNorm(branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )

        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=model_cfg.get("tabular_hidden_dims", [128]),
            dropout=head_dropout,
        )

        self.fusion = ConcatFusion(
            macro_dim=branch_hidden_dim,
            micro_dim=branch_hidden_dim,
            tab_dim=self.tabular_encoder.output_dim,
            hidden_dims=model_cfg.get("fusion_hidden_dims", [512, 256]),
            dropout=float(model_cfg.get("fusion_dropout", 0.2)),
        )
        fusion_out_dim = self.fusion.output_dim

        self.main_head = nn.Linear(fusion_out_dim, main_num_classes)
        self.aux_head = nn.Linear(fusion_out_dim, aux_num_classes)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.macro_backbone.parameters():
            param.requires_grad = trainable
        for param in self.micro_backbone.parameters():
            param.requires_grad = trainable

    def _encode_branch(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        backbone: TimmBackbone,
        pooler: nn.Module,
        projector: nn.Module,
    ):
        instance_feats = masked_backbone_forward(backbone, images, mask)
        pooled, attn = pooler(instance_feats, mask)
        pooled = projector(pooled)
        return pooled, attn

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        macro_feat, macro_attn = self._encode_branch(
            images=batch["macro_images"],
            mask=batch["macro_mask"],
            backbone=self.macro_backbone,
            pooler=self.macro_pool,
            projector=self.macro_proj,
        )
        micro_feat, micro_attn = self._encode_branch(
            images=batch["micro_images"],
            mask=batch["micro_mask"],
            backbone=self.micro_backbone,
            pooler=self.micro_pool,
            projector=self.micro_proj,
        )

        tab_feat, tab_mask = self.tabular_encoder(batch["tabular"])

        has_macro = batch["macro_mask"].any(dim=1)
        has_micro = batch["micro_mask"].any(dim=1)
        branch_mask = torch.stack([has_macro, has_micro, tab_mask], dim=1)

        fused, fusion_weights = self.fusion(
            macro_feat=macro_feat,
            micro_feat=micro_feat,
            tab_feat=tab_feat,
            branch_mask=branch_mask,
        )

        return {
            "main_logits": self.main_head(fused),
            "aux_logits": self.aux_head(fused),
            "macro_attention": macro_attn,
            "micro_attention": micro_attn,
            "fusion_weights": fusion_weights,
        }
