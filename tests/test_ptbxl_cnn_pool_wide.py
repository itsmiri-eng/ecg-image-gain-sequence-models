import torch

from src.models.ptbxl_cnn_pool import LeadImageCNNPool
from src.models.ptbxl_cnn_pool_wide import LeadImageCNNPoolWide
from src.run_ptbxl_wide_head import EXTRA_CODE, wide_head_trainer
from src import run_ptbxl_experiment as experiment
from src.training import ptbxl_trainer as trainer


def test_capacity_matches_sequence_heads():
    count = sum(p.numel() for p in LeadImageCNNPoolWide().parameters())
    assert count == 17065
    assert 17000 < count < 18100


def test_shared_encoder_is_identical():
    wide, pool = LeadImageCNNPoolWide(), LeadImageCNNPool()
    assert [tuple(p.shape) for p in wide.features.parameters()] == [tuple(p.shape) for p in pool.features.parameters()]


def test_head_does_not_mix_columns():
    torch.manual_seed(0)
    model = LeadImageCNNPoolWide().eval()
    sequence = torch.randn(2, 70, 96)
    head = lambda s: model.classifier(model.column_mlp(s).mean(dim=1))
    shuffled = sequence[:, torch.randperm(70)]
    torch.testing.assert_close(head(sequence), head(shuffled), rtol=0, atol=1e-6)


def test_registration_is_temporary():
    before = list(experiment.CODE_FILES)
    with wide_head_trainer():
        assert trainer.get_config({"model": "cnn_pool_wide"})["model"] == "cnn_pool_wide"
        assert experiment.CODE_FILES[-2:] == EXTRA_CODE
    assert "cnn_pool_wide" not in trainer.MODELS
    assert experiment.CODE_FILES == before
