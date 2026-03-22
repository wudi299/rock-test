
from typing import Any, Dict

from .advanced import AdvancedMultiTaskModel
from .baseline import BaselineMultiTaskModel


def build_model(
    cfg: Dict[str, Any],
    main_num_classes: int,
    aux_num_classes: int,
    tabular_input_dim: int,
):
    model_name = str(cfg["model"]["name"]).lower()
    if model_name == "advanced":
        return AdvancedMultiTaskModel(
            cfg=cfg,
            main_num_classes=main_num_classes,
            aux_num_classes=aux_num_classes,
            tabular_input_dim=tabular_input_dim,
        )
    return BaselineMultiTaskModel(
        cfg=cfg,
        main_num_classes=main_num_classes,
        aux_num_classes=aux_num_classes,
        tabular_input_dim=tabular_input_dim,
    )
