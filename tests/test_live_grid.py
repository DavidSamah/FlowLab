import numpy as np

from weather_agent.live_api import LiveWeatherObservation
from weather_agent.live_grid import _interp_2d, _wind_components, observations_to_state
from weather_agent.model import ModelConfig


def make_obs(lat: float, lon: float, z: float, speed: float, direction: float):
    return LiveWeatherObservation(
        provider="test",
        latitude=lat,
        longitude=lon,
        fetched_at_utc="2026-01-01T00:00:00+00:00",
        model_timezone="UTC",
        temperature_2m_c=20.0,
        relative_humidity_2m_pct=50.0,
        surface_pressure_hpa=1000.0,
        precipitation_mm=0.0,
        wind_speed_10m_ms=2.0,
        wind_direction_10m_deg=90.0,
        wind_gusts_10m_ms=3.0,
        temperature_850hpa_c=10.0,
        relative_humidity_850hpa_pct=60.0,
        wind_speed_850hpa_ms=speed,
        wind_direction_850hpa_deg=direction,
        vertical_velocity_850hpa_pas=-0.05,
        geopotential_height_850hpa_m=z,
        source_url="https://example.test",
    )


def test_wind_conversion_easterly():
    u, v = _wind_components(10.0, 90.0)
    assert np.isclose(u, -10.0, atol=1e-8)
    assert np.isclose(v, 0.0, atol=1e-8)


def test_interpolation_preserves_constant_field():
    field = np.full((3, 4), 7.5)
    out = _interp_2d(field, 12, 16)
    assert out.shape == (12, 16)
    assert np.allclose(out, 7.5)


def test_observations_initialize_model_grid():
    grid = [
        [make_obs(0, 0, 1500, 10, 90), make_obs(0, 1, 1510, 10, 90)],
        [make_obs(1, 0, 1490, 10, 90), make_obs(1, 1, 1500, 10, 90)],
    ]
    cfg = ModelConfig(nx=12, ny=8)
    state, aux = observations_to_state(grid, cfg)
    assert state.h.shape == (8, 12)
    assert state.u.shape == (8, 12)
    assert state.v.shape == (8, 12)
    assert np.isclose(np.mean(state.h), 10000.0, atol=1e-6)
    assert np.all(np.isfinite(state.h))
    assert aux["temperature_850hpa_c"].shape == (8, 12)
