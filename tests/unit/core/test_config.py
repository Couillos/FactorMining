from factor_mining.core.config import FactorMiningConfig


def test_config_from_yaml():
    import tempfile, yaml, os
    cfg = {
        "data": {"universe_size": 100},
        "gp": {"pop_size": 50},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    try:
        config = FactorMiningConfig.from_yaml(path)
        assert config.data.universe_size == 100
        assert config.gp.pop_size == 50
    finally:
        os.unlink(path)


def test_config_defaults():
    config = FactorMiningConfig()
    assert config.data.universe_size == 200
    assert config.gp.n_gen == 50
    assert config.validation.jaccard_threshold == 0.7
    assert len(config.factors.pool) == 16
