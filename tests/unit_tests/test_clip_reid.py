
from easydl.reid.clip_reid_config import clip_reid_default_config

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
