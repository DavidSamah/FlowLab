import numpy as np

from weather_agent import ModelConfig, ShallowWaterModel, WeatherAgent, make_vortex_state


def small_config() -> ModelConfig:
    return ModelConfig(nx=24, ny=16, dx_m=50_000.0, dt_s=60.0)


def test_model_produces_finite_state():
    cfg = small_config()
    model = ShallowWaterModel(cfg)
    initial = make_vortex_state(cfg)
    result = model.run(initial, hours=0.1, save_every=2)
    assert np.isfinite(result.final.h).all()
    assert np.isfinite(result.final.u).all()
    assert np.isfinite(result.final.v).all()


def test_mass_is_approximately_conserved():
    cfg = small_config()
    model = ShallowWaterModel(cfg)
    result = model.run(make_vortex_state(cfg), hours=0.1)
    assert abs(result.diagnostics["relative_mass_drift"]) < 1e-6


def test_agent_returns_plan_uncertainty_and_memory():
    cfg = small_config()
    agent = WeatherAgent(ShallowWaterModel(cfg))
    out = agent.solve(hours=0.05)
    assert out["plan"]["actions"]
    assert out["uncertainty"]["mean_height_spread_m"] >= 0.0
    assert out["memory_size"] == 1
