from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .advanced import ExtendedWeatherState


@dataclass
class ObservationBatch:
    y: np.ndarray
    x: np.ndarray
    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    temperature_k: np.ndarray
    specific_humidity: np.ndarray
    error_std: dict[str, float]


def sample_observations(
    truth: ExtendedWeatherState,
    count: int = 96,
    seed: int = 11,
) -> ObservationBatch:
    """Create sparse noisy station-like observations from a reference state."""
    rng = np.random.default_rng(seed)
    ny, nx = truth.h.shape
    y = rng.integers(0, ny, count)
    x = rng.integers(0, nx, count)
    errors = {"h": 3.0, "u": 0.7, "v": 0.7, "temperature_k": 0.5, "specific_humidity": 3e-4}
    return ObservationBatch(
        y=y,
        x=x,
        h=truth.h[y, x] + rng.normal(0, errors["h"], count),
        u=truth.u[y, x] + rng.normal(0, errors["u"], count),
        v=truth.v[y, x] + rng.normal(0, errors["v"], count),
        temperature_k=truth.temperature_k[y, x] + rng.normal(0, errors["temperature_k"], count),
        specific_humidity=np.clip(truth.specific_humidity[y, x] + rng.normal(0, errors["specific_humidity"], count), 0, None),
        error_std=errors,
    )


def _spread_increment(field: np.ndarray, y: int, x: int, increment: float, radius: int, gain: float) -> None:
    ny, nx = field.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r2 = dx*dx + dy*dy
            if r2 > radius*radius:
                continue
            w = np.exp(-0.5*r2/max((0.5*radius)**2, 1.0))
            field[(y+dy) % ny, (x+dx) % nx] += gain*w*increment


def assimilate_nudging(
    background: ExtendedWeatherState,
    obs: ObservationBatch,
    gain: float = 0.35,
    radius: int = 3,
) -> ExtendedWeatherState:
    """Localized objective-analysis/nudging update.

    This is deliberately simple and auditable: observation-minus-background
    innovations are spatially spread with a Gaussian kernel.
    """
    analysis = background.copy()
    fields = [
        (analysis.h, obs.h),
        (analysis.u, obs.u),
        (analysis.v, obs.v),
        (analysis.temperature_k, obs.temperature_k),
        (analysis.specific_humidity, obs.specific_humidity),
    ]
    for i, (y, x) in enumerate(zip(obs.y, obs.x)):
        for field, values in fields:
            innovation = float(values[i] - field[y, x])
            _spread_increment(field, int(y), int(x), innovation, radius, gain)
    analysis.specific_humidity = np.clip(analysis.specific_humidity, 0.0, 0.05)
    analysis.temperature_k = np.clip(analysis.temperature_k, 180.0, 330.0)
    return analysis


def observation_rmse(state: ExtendedWeatherState, obs: ObservationBatch) -> dict[str, float]:
    idx = (obs.y, obs.x)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a-b)**2)))
    return {
        "h_rmse": rmse(state.h[idx], obs.h),
        "u_rmse": rmse(state.u[idx], obs.u),
        "v_rmse": rmse(state.v[idx], obs.v),
        "temperature_rmse": rmse(state.temperature_k[idx], obs.temperature_k),
        "humidity_rmse": rmse(state.specific_humidity[idx], obs.specific_humidity),
    }
