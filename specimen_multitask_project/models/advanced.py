
import random
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .backbones import TimmBackbone, masked_backbone_forward
from .fusion import ConcatFusion, GatedFusion
from .pooling import (
    AttentionMILPooling,
    GatedAttentionMILPooling,
    TransformerMILPooling,
)
from .tabular_encoder import TabularEncoder


class AdvancedMultiTaskModel(nn.Module):
    def __init__(
        self,
        cfg: Dict[str, Any],
        main_num_classes: int,
        aux_num_classes: int,
        tabular_input_dim: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]["advanced"]
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

        branch_hidden_dim = int(model_cfg.get("branch_hidden_dim", 256))
        self.modality_dropout = float(model_cfg.get("modality_dropout", 0.0))
        self.head_dropout = float(model_cfg.get("head_dropout", 0.2))

        self.macro_pool = self._build_pool(model_cfg.get("pool_type", "gated_attention"), self.macro_backbone.out_dim)
        self.micro_pool = self._build_pool(model_cfg.get("pool_type", "gated_attention"), self.micro_backbone.out_dim)

        pooled_macro_dim = getattr(self.macro_pool, "output_dim", self.macro_backbone.out_dim)
        pooled_micro_dim = getattr(self.micro_pool, "output_dim", self.micro_backbone.out_dim)

        self.macro_proj = nn.Sequential(
            nn.Linear(pooled_macro_dim, branch_hidden_dim),
            nn.LayerNorm(branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
        )
        self.micro_proj = nn.Sequential(
            nn.Linear(pooled_micro_dim, branch_hidden_dim),
            nn.LayerNorm(branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
        )

        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dims=model_cfg.get("tabular_hidden_dims", [128]),
            dropout=self.head_dropout,
        )

        fusion_type = str(model_cfg.get("fusion_type", "gated")).lower()
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                macro_dim=branch_hidden_dim,
                micro_dim=branch_hidden_dim,
                tab_dim=self.tabular_encoder.output_dim,
                hidden_dims=model_cfg.get("fusion_hidden_dims", [512, 256]),
                dropout=float(model_cfg.get("fusion_dropout", 0.2)),
            )
        else:
            self.fusion = GatedFusion(
                macro_dim=branch_hidden_dim,
                micro_dim=branch_hidden_dim,
                tab_dim=self.tabular_encoder.output_dim,
                hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
                dropout=float(model_cfg.get("fusion_dropout", 0.2)),
            )

        fusion_out_dim = self.fusion.output_dim
        self.main_head = nn.Linear(fusion_out_dim, main_num_classes)
        self.aux_head = nn.Linear(fusion_out_dim, aux_num_classes)

    def _build_pool(self, pool_type: str, input_dim: int) -> nn.Module:
        pool_type = str(pool_type).lower()
        if pool_type == "attention":
            return AttentionMILPooling(input_dim=input_dim, hidden_dim=128)
        if pool_type == "transformer":
            return TransformerMILPooling(
                input_dim=input_dim,
                hidden_dim=256,
                num_heads=4,
                num_layers=2,
                dropout=0.1,
            )
        return GatedAttentionMILPooling(input_dim=input_dim, hidden_dim=128)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        instance_feats = masked_backbone_forward(backbone, images, mask)
        pooled, attn = pooler(instance_feats, mask)
        pooled = projector(pooled)
        if attn is None:
            attn = torch.zeros((images.shape[0], images.shape[1]), dtype=pooled.dtype, device=pooled.device)
        return pooled, attn

    def _apply_modality_dropout(
        self,
        macro_feat: torch.Tensor,
        micro_feat: torch.Tensor,
        has_macro: torch.Tensor,
        has_micro: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return macro_feat, micro_feat, has_macro, has_micro

        macro_feat = macro_feat.clone()
        micro_feat = micro_feat.clone()
        has_macro = has_macro.clone()
        has_micro = has_micro.clone()

        for i in range(macro_feat.shape[0]):
            if not bool(has_macro[i]) and not bool(has_micro[i]):
                continue
            if bool(has_macro[i]) and bool(has_micro[i]):
                drop_macro = random.random() < self.modality_dropout
                drop_micro = random.random() < self.modality_dropout
                if drop_macro and drop_micro:
                    if random.random() < 0.5:
                        drop_macro = False
                    else:
                        drop_micro = False
                if drop_macro:
                    has_macro[i] = False
                    macro_feat[i] = 0.0
                if drop_micro:
                    has_micro[i] = False
                    micro_feat[i] = 0.0
        return macro_feat, micro_feat, has_macro, has_micro

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
        macro_feat, micro_feat, has_macro, has_micro = self._apply_modality_dropout(
            macro_feat, micro_feat, has_macro, has_micro
        )
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
