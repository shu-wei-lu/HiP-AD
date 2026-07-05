import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from ..blocks import linear_relu_ln
from projects.mmdet3d_plugin.models.utils import nerf_positional_encoding
from functools import partial

_HIPAD_ACTIVATION_INJECTOR = None
_HIPAD_ACTIVATION_IMPORT_FAILED = False


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "t", "yes", "y")


def _selected_plan_feature_layer(layer_index, num_layers):
    target = int(os.environ.get("HIPAD_PLAN_FEATURE_LAYER", "-1"))
    if target < 0:
        target = num_layers + target
    return layer_index == target


def _save_hipad_plan_feature(name, feature, layer_index, num_layers):
    if not _env_flag("SAVE_HIPAD_PLAN_FEATURES"):
        return
    if layer_index is None or num_layers is None:
        return
    if not _selected_plan_feature_layer(layer_index, num_layers):
        return

    root = os.environ.get("FUSED_FEATURES_PATH")
    if root is None:
        return
    run_id = os.environ.get("HIPAD_PLAN_FEATURE_RUN_ID", "hipad")
    frame = int(os.environ.get("HIPAD_PLAN_FEATURE_FRAME", "0"))
    save_dir = Path(root) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(feature.detach().cpu(), save_dir / f"{frame:06d}.pt")


def _parse_activation_alpha():
    value = os.environ.get("HIPAD_ACTIVATION_ALPHA")
    if value is None or value == "":
        return 0.0
    if "," in value:
        return [float(item.strip()) for item in value.split(",")]
    return float(value)


def _activation_layer_selected(layer_index, num_layers):
    target_raw = os.environ.get("HIPAD_ACTIVATION_LAYER", os.environ.get("HIPAD_PLAN_FEATURE_LAYER", "-1"))
    target = int(target_raw)
    if target < 0:
        target = num_layers + target
    return layer_index == target


def _selected_env_layer(name, layer_index, num_layers, default="-1"):
    if layer_index is None or num_layers is None:
        return False
    target = int(os.environ.get(name, default))
    if target < 0:
        target = num_layers + target
    return layer_index == target


def _hipad_activation_injector():
    global _HIPAD_ACTIVATION_INJECTOR, _HIPAD_ACTIVATION_IMPORT_FAILED
    if _HIPAD_ACTIVATION_INJECTOR is not None or _HIPAD_ACTIVATION_IMPORT_FAILED:
        return _HIPAD_ACTIVATION_INJECTOR
    try:
        from activation_steering.injector import ActivationInjector
    except Exception as exc:
        if _env_flag("ENABLE_ACTIVATION_STEERING") or _env_flag("ENABLE_ACTIVATION_INJECTOR"):
            print(f"[HiP-AD ActivationInjector] import failed: {exc}", flush=True)
        _HIPAD_ACTIVATION_IMPORT_FAILED = True
        return None

    default_vector = Path(os.environ.get("HIPAD_ACTIVATION_VECTOR_PATH", "steering_feats/brake_minus_normal.pt"))
    _HIPAD_ACTIVATION_INJECTOR = ActivationInjector.from_env(default_vector)
    return _HIPAD_ACTIVATION_INJECTOR


def _apply_hipad_activation(feature, layer_index, num_layers):
    # if layer_index is None or num_layers is None:
    #     return feature
    # if not _activation_layer_selected(layer_index, num_layers):
    #     return feature
    alpha = _parse_activation_alpha()
    injector = _hipad_activation_injector()
    if injector is None:
        return feature
    return injector.apply(feature, alpha=alpha)


def _apply_sanity_spatial_residual_shift(reg_output, anchor_type, layer_index, num_layers, ego_fut_ts):
    shift = float(os.environ.get("HIPAD_SANITY_RESIDUAL_SPAT_X_SHIFT", "0") or 0)
    if shift == 0.0:
        return reg_output
    if anchor_type != ("spat", "2m"):
        return reg_output
    if not _selected_env_layer("HIPAD_SANITY_RESIDUAL_LAYER", layer_index, num_layers):
        return reg_output
    frame = int(os.environ.get("HIPAD_PLAN_FEATURE_FRAME", "0"))
    start_frame = int(os.environ.get("HIPAD_SANITY_START_FRAME", "-1"))
    end_frame = int(os.environ.get("HIPAD_SANITY_END_FRAME", "1000000000"))
    if frame < start_frame or frame > end_frame:
        return reg_output

    shifted = reg_output.clone()
    mode = os.environ.get("HIPAD_SANITY_RESIDUAL_X_MODE", "ramp").lower()
    if mode == "constant":
        shifted[..., 0::2] += shift
    else:
        shifted[..., 0::2] += shift / float(ego_fut_ts)
    if _env_flag("HIPAD_SANITY_VERBOSE"):
        print(
            "[HiP-AD sanity] residual spatial x shift "
            f"layer={layer_index}, final_shift={shift:.3f}, mode={mode}",
            flush=True,
        )
    return shifted

@PLUGIN_LAYERS.register_module()
class SparsePlanRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_cmd=3, ego_fut_mode=3, add_anchor=False):
        super(SparsePlanRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode
        self.add_anchor = add_anchor

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )

        self.plan_reg_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, ego_fut_ts * 2),
            Scale([1.0] * ego_fut_ts * 2),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, instance_feature, anchor, anchor_embed, use_plan_anchor_embed=True):
        if use_plan_anchor_embed:
            output = self.plan_reg_branch(instance_feature + anchor_embed)
        else:
            output = self.plan_reg_branch(instance_feature)

        output = output + anchor

        cls = self.plan_cls_branch(instance_feature)

        return output, cls

@PLUGIN_LAYERS.register_module()
class SparsePlanAlignRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, ego_fut_ts=6, ego_fut_cmd=3, ego_fut_mode=3, anchor_types=None):
        super(SparsePlanAlignRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode

        self.anchor_types = anchor_types
        self.anchor_group = len(anchor_types)
        self.hipad_refine_layer_index = None
        self.hipad_num_refine_layers = None

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )

        # check speed planning
        speed_type_dict = dict()
        for anchor_type in anchor_types:
            if anchor_type[0] == "speed":
                if anchor_type[1] not in speed_type_dict:
                    speed_type_dict[anchor_type[1]] = [anchor_type[2]]
                else:
                    speed_type_dict[anchor_type[1]].append(anchor_type[2])

        if len(speed_type_dict):
            first_key = list(speed_type_dict.keys())[0]
            self.speed_areas = speed_type_dict[first_key]
            if len(speed_type_dict) > 1:
                for key, val in speed_type_dict.items():
                    assert self.speed_areas == val

            self.plan_cls_branch_speed = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(embed_dims, 1),
            )

        for anchor_type in anchor_types:
            reg_branch = nn.Sequential(
                *linear_relu_ln(embed_dims, 2, 2),
                Linear(embed_dims, ego_fut_ts * 2),
                Scale([1.0] * ego_fut_ts * 2),
            )
            setattr(self, "plan_reg_branch_{}_{}".format(anchor_type[0], anchor_type[1]), reg_branch)

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

        if hasattr(self, "plan_cls_branch_speed"):
            nn.init.constant_(self.plan_cls_branch_speed[-1].bias, bias_init)

    def forward(self, instance_feature, anchor, anchor_embed, use_plan_anchor_embed=True):
        if use_plan_anchor_embed:
            instance_feature = instance_feature + anchor_embed

        instance_features = torch.stack(instance_feature.chunk(self.anchor_group, dim=1))

        align_query = []
        speed_query_dict = dict()
        for index, anchor_type in enumerate(self.anchor_types):
            if anchor_type[0] in ["temp", "spat"]:
                align_query.append(instance_features[index])
            elif anchor_type[0] == "speed":
                if anchor_type[1] not in speed_query_dict:
                    speed_query_dict[anchor_type[1]] = [None] * len(self.speed_areas)
                speed_index = self.speed_areas.index(anchor_type[2])
                speed_query_dict[anchor_type[1]][speed_index] = instance_features[index]
            else:
                raise NotImplementedError

        align_query = sum(align_query)
        _save_hipad_plan_feature(
            "align_query",
            align_query,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )
        align_query = _apply_hipad_activation(
            align_query,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )

        if len(speed_query_dict):
            for speed_index in range(len(self.speed_areas)):
                speed_query = []
                for freq in speed_query_dict.keys():
                    speed_query.append(speed_query_dict[freq][speed_index])
                speed_query = sum(speed_query)
                for freq in speed_query_dict.keys():
                    speed_query_dict[freq][speed_index] = align_query + speed_query

        cls_outputs = []
        reg_outputs = []
        for anchor_type in self.anchor_types:
            reg_branch = getattr(self, "plan_reg_branch_{}_{}".format(anchor_type[0], anchor_type[1]))
            if anchor_type[0] in ["temp", "spat"]:
                reg_output = reg_branch(align_query)
                cls_output = self.plan_cls_branch(align_query)

            elif anchor_type[0] == "speed":
                speed_index = self.speed_areas.index(anchor_type[2])
                speed_query = speed_query_dict[anchor_type[1]][speed_index]
                reg_output = reg_branch(speed_query)
                cls_output = self.plan_cls_branch_speed(speed_query)

            reg_output = _apply_sanity_spatial_residual_shift(
                reg_output,
                anchor_type,
                self.hipad_refine_layer_index,
                self.hipad_num_refine_layers,
                self.ego_fut_ts,
            )
            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)

        cls_outputs = torch.cat(cls_outputs, dim=1)
        reg_outputs = torch.cat(reg_outputs, dim=1)

        reg_outputs = reg_outputs + anchor

        return reg_outputs, cls_outputs
