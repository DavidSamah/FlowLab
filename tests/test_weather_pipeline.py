import numpy as np

from weather_agent.advanced import CoupledWeatherModel, ExtendedConfig, make_extended_initial
from weather_agent.assimilation import sample_observations, assimilate_nudging, observation_rmse
from weather_agent.emulator import train_emulator
from weather_agent.pipeline import run_stage


def test_extended_model_stays_finite():
    cfg = ExtendedConfig()
    cfg.base.nx = 24
    cfg.base.ny = 16
    model = CoupledWeatherModel(cfg)
    initial = make_extended_initial(cfg)
    final = model.run(initial, hours=0.1)
    assert np.isfinite(final.h).all()
    assert np.isfinite(final.u).all()
    assert np.isfinite(final.v).all()
    assert np.isfinite(final.temperature_k).all()
    assert np.isfinite(final.specific_humidity).all()
    assert (final.h > 0).all()


def test_assimilation_reduces_station_temperature_rmse():
    cfg = ExtendedConfig()
    cfg.base.nx = 24
    cfg.base.ny = 16
    model = CoupledWeatherModel(cfg)
    truth = model.run(make_extended_initial(cfg, seed=1), hours=0.1)
    background = model.run(make_extended_initial(cfg, seed=3), hours=0.1)
    background.temperature_k += 1.5
    obs = sample_observations(truth, count=32, seed=4)
    before = observation_rmse(background, obs)
    analysis = assimilate_nudging(background, obs, gain=0.5, radius=2)
    after = observation_rmse(analysis, obs)
    assert after["temperature_rmse"] < before["temperature_rmse"]


def test_emulator_trains_and_predicts_finite_values():
    cfg = ExtendedConfig()
    cfg.base.nx = 20
    cfg.base.ny = 12
    model = CoupledWeatherModel(cfg)
    initial = make_extended_initial(cfg)
    emulator, metrics = train_emulator(model, initial, samples=12)
    pred = emulator.predict(np.array([
        np.mean(initial.h), np.std(initial.h), np.mean(initial.u), np.mean(initial.v),
        np.mean(initial.temperature_k), np.mean(initial.specific_humidity),
        np.mean(initial.rain_rate), np.max(np.sqrt(initial.u**2 + initial.v**2)),
    ]))
    assert np.isfinite(pred).all()
    assert all(np.isfinite(v) for v in metrics.values())


def test_pipeline_metadata_stage_runs():
    result = run_stage("terrain")
    assert result["terrain_max_m"] > result["terrain_min_m"]
    assert result["coriolis_max_s-1"] > result["coriolis_min_s-1"]
