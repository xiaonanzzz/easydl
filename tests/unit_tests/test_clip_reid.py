import torch

from easydl.reid.clip_reid_config import clip_reid_default_config
from easydl.reid.loss import SupConLoss


def test_clip_reid_default_config_returns_cfgnode_clone():
    cfg = clip_reid_default_config()
    # Check that the returned object has expected attributes
    assert hasattr(cfg, "MODEL")
    assert hasattr(cfg, "SOLVER")
    assert hasattr(cfg, "INPUT")
    assert hasattr(cfg, "DATASETS")
    assert hasattr(cfg, "TEST")
    assert hasattr(cfg, "OUTPUT_DIR")
    # Check that modifying the returned config does not affect a new config (clone)
    cfg.MODEL.NAME = "test_model"
    cfg2 = clip_reid_default_config()
    assert cfg2.MODEL.NAME != "test_model"


def test_supconloss_forward_basic_cpu():
    device = "cpu"
    # Create dummy features and labels
    batch_size = 4
    feature_dim = 8
    # Make text_features and image_features random, but with some matching labels
    text_features = torch.randn(batch_size, feature_dim, device=device)
    image_features = torch.randn(batch_size, feature_dim, device=device)
    # Labels: two pairs with same label, two unique
    t_label = torch.tensor([0, 1, 0, 2], device=device)
    i_targets = torch.tensor([0, 1, 0, 2], device=device)
    # Instantiate loss
    loss_fn = SupConLoss(device)
    # Forward pass
    loss = loss_fn(text_features, image_features, t_label, i_targets)
    # Check that loss is a scalar tensor and is finite
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item(), ("loss is not finite", loss)
