"""Convolutional stem for TransD3 encoders."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


LOGGER = logging.getLogger(__name__)


class ConvStem(nn.Module):
    """Shallow convolutional stem producing stride-4 feature maps."""

    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.input_adapter: Optional[nn.Module] = None
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        return self.layers(x)


def load_imagenet_conv_stem(
    stem: ConvStem,
    *,
    model_name: str,
    in_channels: int,
    channel_strategy: str = "avg",
) -> None:
    """Load ImageNet pretrained weights from timm into ``ConvStem``.

    Parameters
    ----------
    stem:
        Target ``ConvStem`` instance to receive weights.
    model_name:
        timm model identifier (e.g. ``"resnet50.a1_in1k"``).
    in_channels:
        Number of input channels expected by the TransD3 stem.
    channel_strategy:
        How to adapt RGB weights to ``in_channels``. ``"avg"`` averages the RGB
        filters and replicates to the desired channel count. ``"repeat"`` tiles the
        weights without averaging. ``"avg"`` is recommended for grayscale inputs.
    """

    if channel_strategy not in {"avg", "repeat"}:
        raise ValueError(f"Unsupported channel_strategy: {channel_strategy}")

    try:
        import timm  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - dependency warning
        raise ImportError("timm is required to load ImageNet weights for ConvStem") from exc

    try:
        state_dict = timm.create_model(model_name, pretrained=True).state_dict()
    except Exception as exc:  # pragma: no cover - network/cache failures
        warnings.warn(
            "Failed to load pretrained ConvStem weights from timm; proceeding with random init. "
            "Set model.stem_imagenet_model=null to disable this attempt."
            f" (reason: {exc})",
            RuntimeWarning,
        )
        return

    conv_modules: List[nn.Conv2d] = [m for m in stem.layers if isinstance(m, nn.Conv2d)]
    bn_modules: List[nn.BatchNorm2d] = [m for m in stem.layers if isinstance(m, nn.BatchNorm2d)]

    conv_keys = _match_conv_weights(state_dict.items(), conv_modules)
    bn_keys = _match_bn_params(state_dict, bn_modules)

    with torch.no_grad():
        for module, key in zip(conv_modules, conv_keys):
            weight = state_dict[key]
            adapted = _adapt_conv_weight(weight, module.weight.shape, in_channels if module is conv_modules[0] else module.in_channels, channel_strategy)
            module.weight.copy_(adapted.to(module.weight.dtype))

        for module, key_prefix in zip(bn_modules, bn_keys):
            if key_prefix is None:
                continue
            module.weight.copy_(state_dict[f"{key_prefix}.weight"].to(module.weight.dtype))
            module.bias.copy_(state_dict[f"{key_prefix}.bias"].to(module.bias.dtype))
            module.running_mean.copy_(state_dict[f"{key_prefix}.running_mean"].to(module.running_mean.dtype))
            module.running_var.copy_(state_dict[f"{key_prefix}.running_var"].to(module.running_var.dtype))


def _match_conv_weights(
    state_items: Iterable[Tuple[str, torch.Tensor]],
    modules: List[nn.Conv2d],
) -> List[str]:
    keys: List[str] = []
    used: set[str] = set()
    for module in modules:
        target_out = module.out_channels
        matched_key: Optional[str] = None
        for name, tensor in state_items:
            if name in used or tensor.ndim != 4:
                continue
            if tensor.shape[0] >= target_out:
                matched_key = name
                used.add(name)
                break
        if matched_key is None:
            raise RuntimeError("Unable to find suitable pretrained conv weight for ConvStem")
        keys.append(matched_key)
    return keys


def _match_bn_params(state: Dict[str, torch.Tensor], modules: List[nn.BatchNorm2d]) -> List[Optional[str]]:
    keys: List[Optional[str]] = []
    used: set[str] = set()
    for module in modules:
        prefix: Optional[str] = None
        for name, tensor in state.items():
            if name in used or not name.endswith(".weight"):
                continue
            if "bn" not in name.split(".")[-2:]:
                continue
            if tensor.ndim == 1 and tensor.shape[0] == module.num_features:
                prefix = name[:-7]
                # Ensure accompanying parameters exist
                required = {f"{prefix}.bias", f"{prefix}.running_mean", f"{prefix}.running_var"}
                if required.issubset(state.keys()):
                    used.update({name, f"{prefix}.bias", f"{prefix}.running_mean", f"{prefix}.running_var"})
                    break
        keys.append(prefix)
        if prefix is None:
            LOGGER.warning("Falling back to random BatchNorm init for ConvStem (num_features=%d)", module.num_features)
    return keys


def _adapt_conv_weight(
    weight: torch.Tensor,
    target_shape: Tuple[int, int, int, int],
    target_in_channels: int,
    channel_strategy: str,
) -> torch.Tensor:
    out_ch, _, k_h, k_w = target_shape
    if weight.shape[0] != out_ch:
        if weight.shape[0] < out_ch:
            repeat = math.ceil(out_ch / weight.shape[0])
            weight = weight.repeat(repeat, 1, 1, 1)[:out_ch]
        else:
            weight = weight[:out_ch]

    desired_in = target_in_channels
    if weight.shape[1] != desired_in:
        if channel_strategy == "repeat":
            repeat = math.ceil(desired_in / weight.shape[1])
            weight = weight.repeat(1, repeat, 1, 1)[:, :desired_in]
        else:  # default to averaging strategy
            weight = weight.mean(dim=1, keepdim=True)
            weight = weight.repeat(1, desired_in, 1, 1)

    if weight.shape[2:] != (k_h, k_w):
        weight = F.interpolate(weight, size=(k_h, k_w), mode="bilinear", align_corners=True)
    return weight
