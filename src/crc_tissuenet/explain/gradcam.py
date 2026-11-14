"""Grad-CAM (Selvaraju et al., ICCV 2017) for convolutional backbones.

Complements SHAP: Grad-CAM shows *where* the last convolutional features responded,
SHAP shows how each input pixel / stain changed the predicted probability.
For transformers and MLP-Mixer use SHAP (Grad-CAM needs a spatial feature map).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def last_conv_layer(model: nn.Module, x: torch.Tensor | None = None) -> nn.Module:
    """The last Conv2d whose output still has a spatial map (> 1x1).

    Some networks (e.g. MobileNetV3) apply a 1x1 convolution *after* global pooling; a CAM taken
    there would be a single pixel, so with an example input we skip such layers.
    """
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Grad-CAM needs a model with Conv2d layers; use SHAP for transformers")
    if x is None:
        return convs[-1]
    sizes: dict[int, int] = {}
    hooks = [m.register_forward_hook(lambda mod, i, o: sizes.__setitem__(id(mod), o.shape[-1] * o.shape[-2]))
             for m in convs]
    try:
        with torch.no_grad():
            model.eval()(x[:1])
    finally:
        for h in hooks:
            h.remove()
    spatial = [m for m in convs if sizes.get(id(m), 0) > 1]
    return spatial[-1] if spatial else convs[-1]


def grad_cam(model: nn.Module, x: torch.Tensor, target: torch.Tensor | None = None,
             layer: nn.Module | None = None) -> np.ndarray:
    """Return heatmaps (N, H, W) in [0, 1] for ``target`` classes (default: predicted)."""
    layer = layer or last_conv_layer(model, x)
    store: dict[str, torch.Tensor] = {}

    def forward_hook(module, inputs, output):
        # Keep the original activation (and its gradient) and pass a CLONE downstream, so in-place
        # activations in the next layers (ReLU/Hardswish(inplace=True)) cannot modify it.
        store["a"] = output
        output.register_hook(lambda g: store.__setitem__("g", g))
        return output.clone()

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        x = x.clone().requires_grad_(True)
        logits = model(x)
        target = logits.argmax(1) if target is None else target
        model.zero_grad(set_to_none=True)
        logits.gather(1, target.view(-1, 1)).sum().backward()
        weights = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()
