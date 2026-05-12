"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import os
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.models as models
import yaml

from augmentation import get_transforms


def _load_config() -> dict:
    with Path("zo_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _balanced_indices(targets: list[int]) -> list[int]:
    counts = [0] * 100
    indices = []
    for index, target in enumerate(targets):
        if counts[target] < 81:
            counts[target] += 1
            indices.append(index)
        if len(indices) == 8100:
            break
    return indices


def _get_subset_loader(data_dir: str, device: torch.device) -> DataLoader:
    dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=False),
    )
    subset = Subset(dataset, _balanced_indices(dataset.targets))
    return DataLoader(
        subset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _extract_backbone_features(data_dir: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loader = _get_subset_loader(data_dir, device)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    features = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            features.append(model(images).cpu())
            labels.append(target)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _extract_imagenet_logits(data_dir: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, nn.Linear]:
    loader = _get_subset_loader(data_dir, device)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    logits = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            logits.append(model(images).cpu())
            labels.append(target)

    return torch.cat(logits, dim=0), torch.cat(labels, dim=0), model.fc.cpu()


def _init_lda(layer: nn.Linear, features: torch.Tensor, labels: torch.Tensor) -> None:
    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
        priors=np.ones(100) / 100.0,
    )
    lda.fit(features.numpy(), labels.numpy())
    layer.weight.copy_(torch.from_numpy(lda.coef_).to(dtype=layer.weight.dtype))
    layer.bias.copy_(torch.from_numpy(lda.intercept_).to(dtype=layer.bias.dtype))


def _init_ridge(layer: nn.Linear, features: torch.Tensor, labels: torch.Tensor, alpha: float) -> None:
    ridge = RidgeClassifier(alpha=alpha)
    ridge.fit(features.numpy(), labels.numpy())
    layer.weight.copy_(torch.from_numpy(ridge.coef_).to(dtype=layer.weight.dtype))
    layer.bias.copy_(torch.from_numpy(ridge.intercept_).to(dtype=layer.bias.dtype))


def _select_top_p_indices(raw_scores: torch.Tensor, top_p: float) -> torch.Tensor:
    positive = torch.clamp(raw_scores, min=0.0)
    if float(positive.sum()) == 0.0:
        return torch.argmax(raw_scores).reshape(1)

    values, indices = torch.sort(positive, descending=True)
    probs = values / values.sum()
    cdf = torch.cumsum(probs, dim=0)
    cutoff = int(torch.nonzero(cdf >= top_p, as_tuple=False)[0].item()) + 1
    cutoff = min(cutoff, 10)
    return indices[:cutoff]


def _init_imagenet_top_p(layer: nn.Linear, logits: torch.Tensor, labels: torch.Tensor, old_fc: nn.Linear, top_p: float) -> None:
    class_means = torch.stack([logits[labels == class_id].mean(dim=0) for class_id in range(100)])
    global_mean = logits.mean(dim=0)
    raw_scores = class_means - global_mean
    scores = torch.clamp(raw_scores, min=0.0)

    old_weight = old_fc.weight.detach()
    old_bias = old_fc.bias.detach()
    new_weight = []
    new_bias = []

    for class_id in range(100):
        selected = _select_top_p_indices(raw_scores[class_id], top_p)
        selected_scores = scores[class_id, selected]
        if float(selected_scores.sum()) == 0.0:
            weights = torch.ones(len(selected), dtype=old_weight.dtype) / len(selected)
        else:
            weights = selected_scores / selected_scores.sum()
        new_weight.append((weights[:, None] * old_weight[selected]).sum(dim=0))
        new_bias.append((weights * old_bias[selected]).sum())

    layer.weight.copy_(torch.stack(new_weight).to(dtype=layer.weight.dtype))
    layer.bias.copy_(torch.stack(new_bias).to(dtype=layer.bias.dtype))


def _init_zero(layer: nn.Linear) -> None:
    layer.weight.zero_()
    layer.bias.zero_()


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------
    config = _load_config()
    data_dir = os.environ.get("CIFAR100_DATA_DIR", "./data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        if config["init_method"] == "lda":
            features, labels = _extract_backbone_features(data_dir, device)
            _init_lda(layer, features, labels)
        elif config["init_method"] == "ridge":
            features, labels = _extract_backbone_features(data_dir, device)
            _init_ridge(layer, features, labels, float(config["ridge_alpha"]))
        elif config["init_method"] == "imagenet_top_p":
            logits, labels, old_fc = _extract_imagenet_logits(data_dir, device)
            _init_imagenet_top_p(layer, logits, labels, old_fc, float(config["top_p"]))
        elif config["init_method"] == "zero":
            _init_zero(layer)
        else:
            raise ValueError(f"Unknown init_method: {config['init_method']}")
    # -------------------------------------------------------------------------
