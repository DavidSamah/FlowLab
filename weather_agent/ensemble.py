from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .advanced import CoupledWeatherModel, ExtendedWeatherState


@dataclass
class EnsembleSummary:
    mean: ExtendedWeatherState
    members: list[ExtendedWeatherState]
    spread: dict[str, float]


def perturb_state(state: ExtendedWeatherState, seed: int) -> ExtendedWeatherState:
    rng = np.random.default_rng(seed)
    s = state.copy()
    s.h += rng.normal(0.0, 2.0, s.h.shape)
    s.u += rng.normal(0.0, 0.25, s.u.shape)
    s.v += rng.normal(0.0, 0.25, s.v.shape)
    s.temperature_k += rng.normal(0.0, 0.15, s.temperature_k.shape)
    s.specific_humidity = np.clip(
        s.specific_humidity + rng.normal(0.0, 1e-4, s.specific_humidity.shape), 0.0, 0.05
    )
    return s


def run_ensemble(
    model: CoupledWeatherModel,
    analysis: ExtendedWeatherState,
    hours: float = 3.0,
    members: int = 6,
    seed: int = 100,
) -> EnsembleSummary:
    forecasts = [model.run(perturb_state(analysis, seed+i), hours) for i in range(members)]

    def stack(name):
        return np.stack([getattr(s, name) for s in forecasts], axis=0)

    mean = ExtendedWeatherState(
        np.mean(stack("h"), axis=0),
        np.mean(stack("u"), axis=0),
        np.mean(stack("v"), axis=0),
        np.mean(stack("temperature_k"), axis=0),
        np.mean(stack("specific_humidity"), axis=0),
        np.mean(stack("rain_rate"), axis=0),
        analysis.terrain_m.copy(),
        forecasts[0].time_s,
    )
    spread = {
        "height_spread": float(np.mean(np.std(stack("h"), axis=0))),
        "u_spread": float(np.mean(np.std(stack("u"), axis=0))),
        "v_spread": float(np.mean(np.std(stack("v"), axis=0))),
        "temperature_spread": float(np.mean(np.std(stack("temperature_k"), axis=0))),
        "humidity_spread": float(np.mean(np.std(stack("specific_humidity"), axis=0))),
        "rain_spread": float(np.mean(np.std(stack("rain_rate"), axis=0))),
    }
    return EnsembleSummary(mean=mean, members=forecasts, spread=spread)


def field_rmse(forecast: ExtendedWeatherState, truth: ExtendedWeatherState) -> dict[str, float]:
    def rmse(a, b):
        return float(np.sqrt(np.mean((a-b)**2)))
    return {
        "height_rmse": rmse(forecast.h, truth.h),
        "u_rmse": rmse(forecast.u, truth.u),
        "v_rmse": rmse(forecast.v, truth.v),
        "temperature_rmse": rmse(forecast.temperature_k, truth.temperature_k),
        "humidity_rmse": rmse(forecast.specific_humidity, truth.specific_humidity),
        "rain_rmse": rmse(forecast.rain_rate, truth.rain_rate),
    }
