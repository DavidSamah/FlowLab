from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state import WeatherState, ForecastResult


@dataclass
class ModelConfig:
    nx: int = 96
    ny: int = 64
    dx_m: float = 50_000.0
    dt_s: float = 60.0
    gravity: float = 9.81
    coriolis_f: float = 1.0e-4
    viscosity: float = 2.5e4
    height_diffusivity: float = 2.5e4
    min_height_m: float = 100.0


class ShallowWaterModel:
    """Educational rotating shallow-water solver on a periodic grid.

    It models coupled horizontal wind and pressure/height evolution. This is a
    real PDE weather-model analogue, but not an operational forecast system.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.cfg = config or ModelConfig()

    @staticmethod
    def _ddx(a: np.ndarray, dx: float) -> np.ndarray:
        return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2.0 * dx)

    @staticmethod
    def _ddy(a: np.ndarray, dx: float) -> np.ndarray:
        return (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2.0 * dx)

    @staticmethod
    def _lap(a: np.ndarray, dx: float) -> np.ndarray:
        return (
            np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0)
            + np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1)
            - 4.0 * a
        ) / (dx * dx)

    def tendencies(self, state: WeatherState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.cfg
        h, u, v = state.h, state.u, state.v

        dhdx = self._ddx(h, c.dx_m)
        dhdy = self._ddy(h, c.dx_m)
        dudx = self._ddx(u, c.dx_m)
        dudy = self._ddy(u, c.dx_m)
        dvdx = self._ddx(v, c.dx_m)
        dvdy = self._ddy(v, c.dx_m)

        du = -(u * dudx + v * dudy) - c.gravity * dhdx + c.coriolis_f * v
        dv = -(u * dvdx + v * dvdy) - c.gravity * dhdy - c.coriolis_f * u

        hu = h * u
        hv = h * v
        dh = -(self._ddx(hu, c.dx_m) + self._ddy(hv, c.dx_m))

        du += c.viscosity * self._lap(u, c.dx_m)
        dv += c.viscosity * self._lap(v, c.dx_m)
        dh += c.height_diffusivity * self._lap(h, c.dx_m)

        return dh, du, dv

    def step(self, state: WeatherState) -> WeatherState:
        """Advance one step with midpoint RK2 integration."""
        c = self.cfg
        dh1, du1, dv1 = self.tendencies(state)
        mid = WeatherState(
            np.maximum(state.h + 0.5 * c.dt_s * dh1, c.min_height_m),
            state.u + 0.5 * c.dt_s * du1,
            state.v + 0.5 * c.dt_s * dv1,
            state.time_s + 0.5 * c.dt_s,
        )
        dh2, du2, dv2 = self.tendencies(mid)
        return WeatherState(
            np.maximum(state.h + c.dt_s * dh2, c.min_height_m),
            state.u + c.dt_s * du2,
            state.v + c.dt_s * dv2,
            state.time_s + c.dt_s,
        )

    def run(self, initial: WeatherState, hours: float = 6.0, save_every: int = 30) -> ForecastResult:
        steps = max(1, int(hours * 3600.0 / self.cfg.dt_s))
        state = initial.copy()
        trajectory = [state.copy()]
        initial_mass = float(np.sum(initial.h))

        for i in range(steps):
            state = self.step(state)
            if (i + 1) % save_every == 0 or i == steps - 1:
                trajectory.append(state.copy())

        final_diag = state.diagnostics()
        final_diag["relative_mass_drift"] = (
            float(np.sum(state.h)) - initial_mass
        ) / max(abs(initial_mass), 1e-12)
        final_diag["forecast_hours"] = float(hours)
        return ForecastResult(initial.copy(), state, trajectory, final_diag)


def make_vortex_state(config: ModelConfig | None = None, amplitude_m: float = 80.0) -> WeatherState:
    """Create a balanced-looking pressure disturbance for repeatable experiments."""
    c = config or ModelConfig()
    y, x = np.mgrid[0:c.ny, 0:c.nx]
    cx, cy = 0.5 * (c.nx - 1), 0.5 * (c.ny - 1)
    sx, sy = 0.16 * c.nx, 0.20 * c.ny
    r2 = ((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2
    anomaly = amplitude_m * np.exp(-0.5 * r2)
    h0 = 10_000.0 + anomaly

    # Approximate geostrophic balance: f v = g dh/dx, f u = -g dh/dy.
    model = ShallowWaterModel(c)
    u0 = -(c.gravity / c.coriolis_f) * model._ddy(h0, c.dx_m)
    v0 = +(c.gravity / c.coriolis_f) * model._ddx(h0, c.dx_m)
    return WeatherState(h0, u0, v0, 0.0)
