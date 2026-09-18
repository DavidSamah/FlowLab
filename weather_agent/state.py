from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class WeatherState:
    """Minimal gridded atmospheric state for the shallow-water model."""

    h: np.ndarray  # fluid-layer height / pressure-like field [m]
    u: np.ndarray  # zonal wind [m/s]
    v: np.ndarray  # meridional wind [m/s]
    time_s: float = 0.0

    def copy(self) -> "WeatherState":
        return WeatherState(self.h.copy(), self.u.copy(), self.v.copy(), self.time_s)

    def diagnostics(self) -> Dict[str, float]:
        wind = np.sqrt(self.u**2 + self.v**2)
        return {
            "time_s": float(self.time_s),
            "mean_height_m": float(np.mean(self.h)),
            "height_std_m": float(np.std(self.h)),
            "max_wind_ms": float(np.max(wind)),
            "mean_wind_ms": float(np.mean(wind)),
            "mass_proxy": float(np.sum(self.h)),
        }


@dataclass
class ForecastResult:
    initial: WeatherState
    final: WeatherState
    trajectory: List[WeatherState]
    diagnostics: Dict[str, float]
    ensemble_spread: Dict[str, float] | None = None
