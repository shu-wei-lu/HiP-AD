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
# Edit these module parameters to choose where HiP-AD activation steering is applied.
# Layers support "-1", "0,2,-1", or "all"; features support one or more saved feature names.
_HIPAD_ACTIVATION_LAYER = "all"
_HIPAD_ACTIVATION_features = "align_query" # pre_instance_feature instance_feature_with_anchor_embed align_query


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "t", "yes", "y")


def _selected_plan_feature_layer(layer_index, num_layers):
    layers = os.environ.get("HIPAD_PLAN_FEATURE_LAYERS")
    if layers is None:
        legacy_layer = os.environ.get("HIPAD_PLAN_FEATURE_LAYER")
        if legacy_layer is None:
            return True
        layers = legacy_layer

    layers = layers.strip().lower()
    if layers in ("", "all", "*"):
        return True

    for item in layers.split(","):
        item = item.strip()
        if not item:
            continue
        target = int(item)
        if target < 0:
            target = num_layers + target
        if layer_index == target:
            return True
    return False


def _selected_plan_feature_name(name):
    names = os.environ.get("HIPAD_PLAN_FEATURE_NAMES")
    if names is None or names.strip().lower() in ("", "all", "*"):
        return True
    return name in {item.strip() for item in names.split(",") if item.strip()}


def _save_hipad_plan_feature(name, feature, layer_index, num_layers):
    if not _env_flag("SAVE_HIPAD_PLAN_FEATURES"):
        return
    if layer_index is None or num_layers is None:
        return
    if not _selected_plan_feature_layer(layer_index, num_layers):
        return
    if not _selected_plan_feature_name(name):
        return

    root = os.environ.get("FUSED_FEATURES_PATH")
    if root is None:
        return
    run_id = os.environ.get("HIPAD_PLAN_FEATURE_RUN_ID", "hipad")
    frame = int(os.environ.get("HIPAD_PLAN_FEATURE_FRAME", "0"))
    save_dir = Path(root) / run_id / name / f"layer_{layer_index:02d}"
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
    if layer_index is None or num_layers is None:
        return False
    layers = os.environ.get(
        "HIPAD_ACTIVATION_LAYER",
        os.environ.get("HIPAD_PLAN_FEATURE_LAYER", _HIPAD_ACTIVATION_LAYER),
    )
    layers = str(layers).strip().lower()
    if layers in ("", "all", "*"):
        return True

    for item in layers.split(","):
        item = item.strip()
        if not item:
            continue
        target = int(item)
        if target < 0:
            target = num_layers + target
        if layer_index == target:
            return True
    return False


def _activation_feature_selected(name):
    features = _HIPAD_ACTIVATION_features
    if isinstance(features, str):
        features = features.strip()
        if features.lower() in ("", "all", "*"):
            return True
        return name in {item.strip() for item in features.split(",") if item.strip()}
    if features is None:
        return False
    return name in set(features)


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


def _apply_hipad_activation(name, feature, layer_index, num_layers):
    if not _activation_layer_selected(layer_index, num_layers):
        return feature
    if not _activation_feature_selected(name):
        return feature
    alpha = _parse_activation_alpha()
    injector = _hipad_activation_injector()
    if injector is None:
        return feature

    cosine_gate_enabled = _env_flag("HIPAD_ACTIVATION_COSINE_GATE", True)
    cosine_gate_low = float(os.environ.get("HIPAD_ACTIVATION_COSINE_GATE_LOW", "0.30"))
    cosine_gate_high = float(os.environ.get("HIPAD_ACTIVATION_COSINE_GATE_HIGH", "0.60"))
    if cosine_gate_high <= cosine_gate_low:
        raise ValueError(
            "HIPAD_ACTIVATION_COSINE_GATE_HIGH must be greater than "
            "HIPAD_ACTIVATION_COSINE_GATE_LOW."
        )

    # The rank-3 call is the original align_query hook. It is reached once at
    # layer 0 before the rank-4 speed-bin hook, so it marks the start of a new
    # planner forward/frame and invalidates the previous frame's gates.
    if layer_index == 0 and name == "align_query" and feature.ndim == 3:
        injector._hipad_layer0_cosine_gates = {}

    def matches_hook(vector):
        # Reject broadcasting that would expand the feature itself. For
        # example, [1, 3, 48, 256] + [1, 48, 256] must not silently turn the
        # align query into [1, 3, 48, 256].
        return (
            vector.ndim == feature.ndim
            and all(vector_size in (1, feature_size)
                    for vector_size, feature_size in zip(vector.shape, feature.shape))
        )

    def layer0_cosine_gate(vector, cache_key):
        # Brake v5 is rank 4 [B, stop/slow/fast, modes, dims]. The adaptive
        # gate is intentionally limited to rank-3 left/right align vectors.
        if not cosine_gate_enabled or vector.ndim != 3:
            return 1.0

        gates = getattr(injector, "_hipad_layer0_cosine_gates", {})
        if layer_index == 0:
            reference_vector = vector
            if reference_vector.shape[0] == 1 and feature.shape[0] != 1:
                reference_vector = reference_vector.expand(
                    feature.shape[0], *reference_vector.shape[1:])
            feature_flat = feature.detach().float().reshape(feature.shape[0], -1)
            vector_flat = reference_vector.detach().float().reshape(reference_vector.shape[0], -1)
            similarity = torch.nn.functional.cosine_similarity(
                feature_flat,
                vector_flat,
                dim=1,
                eps=1e-8,
            )
            gate = ((cosine_gate_high - similarity) /
                    (cosine_gate_high - cosine_gate_low)).clamp(0.0, 1.0)
            gate = gate.to(device=feature.device, dtype=feature.dtype)
            gate = gate.reshape(feature.shape[0], *([1] * (feature.ndim - 1)))
            gates[cache_key] = gate
            injector._hipad_layer0_cosine_gates = gates
            if _env_flag("HIPAD_ACTIVATION_COSINE_GATE_VERBOSE"):
                print(
                    "[HiP-AD cosine gate] "
                    f"key={cache_key}, "
                    f"similarity={similarity.detach().cpu().tolist()}, "
                    f"gate={gate.detach().cpu().flatten().tolist()}",
                    flush=True,
                )
        return gates.get(cache_key, 1.0)

    # Route a legacy per-query vector to the original align-query hook and a
    # speed-bin-specific vector to the stacked speed-query hook. Both hooks
    # use the saved feature name "align_query", so shape compatibility keeps
    # the routing generic without action-specific branching.
    if injector.vector_path is not None:
        vector = injector.vector(
            feature,
            layer_index=layer_index,
            num_layers=num_layers,
        )
        if not matches_hook(vector):
            return feature
        if vector.ndim == 3 and cosine_gate_enabled:
            alpha_values = torch.as_tensor(
                alpha,
                device=feature.device,
                dtype=feature.dtype,
            ).flatten()
            if alpha_values.numel() != 1:
                raise ValueError(
                    "A single ACTIVATION_VECTOR_PATH requires a scalar "
                    "HIPAD_ACTIVATION_ALPHA when cosine gating is enabled."
                )
            gate = layer0_cosine_gate(vector, f"single:{injector.vector_path}")
            return feature + alpha_values[0] * gate * vector

    elif injector.vector_paths is not None:
        alpha_vector = injector.alpha_vector(alpha).to(
            device=feature.device,
            dtype=feature.dtype,
        )
        alpha_scales = torch.as_tensor(
            injector.action_alpha_scales,
            device=feature.device,
            dtype=feature.dtype,
        )
        vectors = injector.action_vectors(
            feature,
            layer_index=layer_index,
            num_layers=num_layers,
        )
        result = feature
        for action_index, (action_alpha, vector) in enumerate(
                zip(alpha_vector * alpha_scales, vectors)):
            if vector is None or float(action_alpha.detach().cpu()) == 0.0:
                continue
            if matches_hook(vector):
                gate = layer0_cosine_gate(
                    vector,
                    f"action:{action_index}:{injector.vector_paths[action_index]}",
                )
                result = result + action_alpha * gate * vector
        return result

    result = injector.apply(
        feature,
        alpha=alpha,
        layer_index=layer_index,
        num_layers=num_layers,
    )
    if result.shape != feature.shape:
        raise RuntimeError(
            "HiP-AD activation steering changed the feature shape from "
            f"{tuple(feature.shape)} to {tuple(result.shape)}."
        )
    return result


def _hipad_brake_activation_active(reference, layer_index, num_layers):
    """Return whether the current policy is applying the brake activation."""
    injector = _hipad_activation_injector()
    if injector is None:
        return False

    alpha = _parse_activation_alpha()
    if injector.vector_paths is not None:
        if not injector.vector_paths or injector.vector_paths[0] is None:
            return False
        alpha_vector = injector.alpha_vector(alpha)
        effective_brake_alpha = (
            alpha_vector[0] * float(injector.action_alpha_scales[0])
        )
        return float(effective_brake_alpha) > 0.0

    if injector.vector_path is None:
        return False

    alpha_values = torch.as_tensor(alpha).detach().cpu().flatten().float()
    if alpha_values.numel() != 1 or float(alpha_values[0]) <= 0.0:
        return False

    # A single rank-4 vector is the fused [stop, slow, fast] brake vector.
    vector = injector.vector(
        reference,
        layer_index=layer_index,
        num_layers=num_layers,
    )
    return vector.ndim == 4


def _cap_hipad_brake_stop_regression(
        reg_outputs,
        anchor_types,
        anchor_group,
        ego_fut_ts,
        layer_index,
        num_layers,
):
    """Cap final-layer 5 Hz stop trajectories while brake steering is active."""
    if not _env_flag("HIPAD_BRAKE_REG_CAP", True):
        return reg_outputs
    if layer_index is None or num_layers is None or layer_index != num_layers - 1:
        return reg_outputs
    if not _hipad_brake_activation_active(reg_outputs, layer_index, num_layers):
        return reg_outputs

    stop_group = ("speed", "5hz", (0, 0.4))
    if stop_group not in anchor_types:
        return reg_outputs

    target_speed = float(os.environ.get("HIPAD_BRAKE_REG_TARGET_SPEED", "0.25"))
    if target_speed < 0.0:
        raise ValueError("HIPAD_BRAKE_REG_TARGET_SPEED must be non-negative.")

    batch_size = reg_outputs.shape[0]
    grouped = reg_outputs.reshape(
        batch_size,
        anchor_group,
        -1,
        ego_fut_ts,
        2,
    ).clone()
    stop_index = anchor_types.index(stop_group)
    stop_traj = grouped[:, stop_index]
    stop_deltas = stop_traj[..., 1:, :] - stop_traj[..., :-1, :]
    desired_speed = (
        torch.linalg.vector_norm(stop_deltas.float(), dim=-1).mean(dim=-1)
        / 0.2
    )
    scale = (
        target_speed / desired_speed.clamp_min(1e-6)
    ).clamp(max=1.0)
    grouped[:, stop_index] = stop_traj * scale.to(
        dtype=stop_traj.dtype,
    )[..., None, None]

    if _env_flag("HIPAD_BRAKE_REG_CAP_VERBOSE"):
        print(
            "[HiP-AD brake reg cap] "
            f"target_speed={target_speed:.3f}, "
            f"desired_speed_before="
            f"(min={float(desired_speed.min()):.3f}, "
            f"mean={float(desired_speed.mean()):.3f}, "
            f"max={float(desired_speed.max()):.3f}), "
            f"scale="
            f"(min={float(scale.min()):.3f}, "
            f"mean={float(scale.mean()):.3f}, "
            f"max={float(scale.max()):.3f})",
            flush=True,
        )

    return grouped.reshape(batch_size, -1, ego_fut_ts * 2)


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
        _save_hipad_plan_feature(
            "pre_instance_feature",
            instance_feature,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )
        instance_feature = _apply_hipad_activation(
            "pre_instance_feature",
            instance_feature,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )
        if use_plan_anchor_embed:
            instance_feature = instance_feature + anchor_embed
        _save_hipad_plan_feature(
            "instance_feature_with_anchor_embed",
            instance_feature,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )
        instance_feature = _apply_hipad_activation(
            "instance_feature_with_anchor_embed",
            instance_feature,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )

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
            "align_query",
            align_query,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )

        if len(speed_query_dict):
            speed_queries = []
            for speed_index in range(len(self.speed_areas)):
                speed_query = []
                for freq in speed_query_dict.keys():
                    speed_query.append(speed_query_dict[freq][speed_index])
                speed_queries.append(sum(speed_query))

            # [B, num_speed_bins, num_modes, embed_dims]. Only a matching
            # bin-specific [1, num_speed_bins, num_modes, embed_dims] vector
            # is routed here; legacy [1, num_modes, embed_dims] vectors have
            # already been applied to align_query above.
            speed_queries = torch.stack(speed_queries, dim=1)
            speed_queries = _apply_hipad_activation(
                "align_query",
                speed_queries,
                self.hipad_refine_layer_index,
                self.hipad_num_refine_layers,
            )

            for speed_index in range(len(self.speed_areas)):
                speed_query = speed_queries[:, speed_index]
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

            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)

        cls_outputs = torch.cat(cls_outputs, dim=1)
        reg_outputs = torch.cat(reg_outputs, dim=1)

        reg_outputs = reg_outputs + anchor
        reg_outputs = _cap_hipad_brake_stop_regression(
            reg_outputs,
            self.anchor_types,
            self.anchor_group,
            self.ego_fut_ts,
            self.hipad_refine_layer_index,
            self.hipad_num_refine_layers,
        )

        return reg_outputs, cls_outputs
