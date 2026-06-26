import torch

from agn_real.models.feature_generator import rbf_mmd


def test_rbf_mmd_is_small_for_identical_samples_and_larger_for_shifted_samples():
    x = torch.randn(12, 4)
    same = rbf_mmd(x, x)
    shifted = rbf_mmd(x, x + 3.0)
    assert same.item() <= 1e-6
    assert shifted.item() > same.item()
