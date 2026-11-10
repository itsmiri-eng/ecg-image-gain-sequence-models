import json

import pytest
import torch

from src.common import ROOT
from src.models.ptbxl_resnet18 import WEIGHTS, LeadImageResNet18
from src.run_ptbxl_extra import extra_models
from src.training import ptbxl_trainer as trainer

needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="ImageNet ResNet-18 weights not downloaded")


def test_shape_and_parameters_without_weights():
    model = LeadImageResNet18(pretrained=False).eval()
    assert sum(p.numel() for p in model.parameters()) == 11_172_805
    with torch.no_grad():
        assert model(torch.zeros(2, 12, 120, 1120)).shape == (2, 5)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3, 224, 224))


@needs_weights
def test_imagenet_weights_load_with_summed_first_filter():
    reference = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = LeadImageResNet18()
    assert torch.equal(model.conv1.weight, reference["conv1.weight"].sum(dim=1, keepdim=True))
    assert torch.equal(model.layer4[1].bn2.running_var, reference["layer4.1.bn2.running_var"])


def test_rows_are_stacked_top_to_bottom():
    images = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1).expand(1, 12, 120, 1120)
    stacked = images.reshape(1, 1, 1440, 1120)
    assert stacked[0, 0, 0, 0] == 0 and stacked[0, 0, 119, 0] == 0 and stacked[0, 0, 120, 0] == 1
    assert stacked[0, 0, 1439, 1119] == 11


def test_registration_is_restored():
    with extra_models():
        assert {"cnn_pool_wide", "resnet18"} <= set(trainer.MODELS)
    assert "resnet18" not in trainer.MODELS and "cnn_pool_wide" not in trainer.MODELS


@pytest.mark.parametrize("head", ["cnn_pool", "cnn_pool_wide", "cnn_lstm", "temporal_conv", "hybrid_vit"])
def test_long40_configs_differ_only_in_budget_and_names(head):
    base = json.loads((ROOT / f"experiments/configs/ptbxl_v2_{head}.json").read_text(encoding="utf-8"))
    long = json.loads((ROOT / f"experiments/configs/ptbxl_v2_long40_{head}.json").read_text(encoding="utf-8"))
    assert long["training"] == {**base["training"], "max_epochs": 40}
    for key in ("cache_root", "cache_sha256", "preprocessing", "selection_rule", "threshold", "uncertainty"):
        assert long[key] == base[key]
    assert long["result_family"] == "ptbxl_v2_long40" and long["protocol"] == f"ptbxl_v2_long40_{head}"


def test_resnet_configs_share_training_settings_and_caches():
    for prefix, base_name in (("", "ptbxl_v2_cnn_pool"), ("fg_", "ptbxl_v2_fg_cnn_pool")):
        base = json.loads((ROOT / f"experiments/configs/{base_name}.json").read_text(encoding="utf-8"))
        cfg = json.loads((ROOT / f"experiments/configs/ptbxl_v2_{prefix}resnet18.json").read_text(encoding="utf-8"))
        assert cfg["training"] == {**base["training"], "model": "resnet18"}
        assert cfg["cache_root"] == base["cache_root"] and cfg["cache_sha256"] == base["cache_sha256"]
        assert cfg.get("dataset_loader") == base.get("dataset_loader")
