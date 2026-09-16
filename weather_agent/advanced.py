from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state import WeatherState
from .model import ModelConfig, ShallowWaterModel


@dataclass
class ExtendedWeatherState:
    """Weather state with thermodynamics, moisture, rain and terrain."""

    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    temperature_k: np.ndarray
    specific_humidity: np.ndarray
    rain_rate: np.ndarray
    terrain_m: np.ndarray
    time_s: float = 0.0

    def copy(self) -> "ExtendedWeatherState":
        return ExtendedWeatherState(
            self.h.copy(), self.u.copy(), self.v.copy(),
            self.temperature_k.copy(), self.specific_humidity.copy(),
            self.rain_rate.copy(), self.terrain_m.copy(), self.time_s,
        )

    def base(self) -> WeatherState:
        return WeatherState(self.h.copy(), self.u.copy(), self.v.copy(), self.time_s)

    def diagnostics(self) -> dict[str, float]:
        wind = np.sqrt(self.u**2 + self.v**2)
        return {
            "time_s": float(self.time_s),
            "mean_height_m": float(np.mean(self.h)),
            "max_wind_ms": float(np.max(wind)),
            "mean_temperature_k": float(np.mean(self.temperature_k)),
            "min_temperature_k": float(np.min(self.temperature_k)),
            "max_temperature_k": float(np.max(self.temperature_k)),
            "mean_specific_humidity": float(np.mean(self.specific_humidity)),
            "total_rain_proxy": float(np.sum(self.rain_rate)),
            "max_rain_rate_proxy": float(np.max(self.rain_rate)),
            "mass_proxy": float(np.sum(self.h)),
        }


@dataclass
class ExtendedConfig:
    base: ModelConfig = None  # type: ignore[assignment]
    beta: float = 1.6e-11
    reference_temperature_k: float = 288.0
    thermal_relaxation_s: float = 5.0 * 86400.0
    scalar_diffusivity: float = 1.5e4
    latent_heating_k_per_q: float = 1200.0
    condensation_timescale_s: float = 1800.0
    rain_fallout_timescale_s: float = 3600.0
    terrain_drag: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.base is None:
            self.base = ModelConfig()


class CoupledWeatherModel:
    """Compact beta-plane shallow-water + thermodynamic/moisture model.

    This is intentionally inspectable. It captures several mechanisms used in
    conceptual atmospheric models, but it is not an operational NWP core.
    """

    def __init__(self, config: ExtendedConfig | None = None):
        self.cfg = config or ExtendedConfig()
        self.base_model = ShallowWaterModel(self.cfg.base)

    def coriolis_field(self) -> np.ndarray:
        c = self.cfg
        b = c.base
        y = (np.arange(b.ny) - 0.5 * (b.ny - 1)) * b.dx_m
        return b.coriolis_f + c.beta * y[:, None]

    @staticmethod
    def saturation_specific_humidity(temp_k: np.ndarray) -> np.ndarray:
        """Smooth toy Clausius-Clapeyron-like saturation curve."""
        return np.clip(0.010 * np.exp(0.065 * (temp_k - 288.0)), 0.001, 0.040)

    def tendencies(self, s: ExtendedWeatherState):
        b = self.cfg.base
        m = self.base_model
        f = self.coriolis_field()
        h, u, v = s.h, s.u, s.v

        # Effective geopotential includes terrain; gradients steer flow.
        geop = h + s.terrain_m
        dhdx = m._ddx(geop, b.dx_m)
        dhdy = m._ddy(geop, b.dx_m)
        dudx, dudy = m._ddx(u, b.dx_m), m._ddy(u, b.dx_m)
        dvdx, dvdy = m._ddx(v, b.dx_m), m._ddy(v, b.dx_m)

        speed = np.sqrt(u*u + v*v)
        terrain_drag = self.cfg.terrain_drag * (1.0 + s.terrain_m / max(np.max(s.terrain_m), 1.0))
        du = -(u*dudx + v*dudy) - b.gravity*dhdx + f*v - terrain_drag*u
        dv = -(u*dvdx + v*dvdy) - b.gravity*dhdy - f*u - terrain_drag*v
        du += b.viscosity * m._lap(u, b.dx_m)
        dv += b.viscosity * m._lap(v, b.dx_m)

        hu, hv = h*u, h*v
        dh = -(m._ddx(hu, b.dx_m) + m._ddy(hv, b.dx_m))
        dh += b.height_diffusivity * m._lap(h, b.dx_m)

        t = s.temperature_k
        q = s.specific_humidity
        rt = s.rain_rate
        dtdx, dtdy = m._ddx(t, b.dx_m), m._ddy(t, b.dx_m)
        dqdx, dqdy = m._ddx(q, b.dx_m), m._ddy(q, b.dx_m)
        dt = -(u*dtdx + v*dtdy)
        dq = -(u*dqdx + v*dqdy)
        dt += self.cfg.scalar_diffusivity * m._lap(t, b.dx_m)
        dq += self.cfg.scalar_diffusivity * m._lap(q, b.dx_m)
        dt += -(t - self.cfg.reference_temperature_k) / self.cfg.thermal_relaxation_s

        qsat = self.saturation_specific_humidity(t)
        excess = np.maximum(q - qsat, 0.0)
        condensation = excess / self.cfg.condensation_timescale_s
        dq -= condensation
        dt += self.cfg.latent_heating_k_per_q * condensation
        dr = condensation - rt / self.cfg.rain_fallout_timescale_s

        # Weak orographic moisture lifting: upslope flow encourages condensation.
        dzdx = m._ddx(s.terrain_m, b.dx_m)
        dzdy = m._ddy(s.terrain_m, b.dx_m)
        upslope = np.maximum(u*dzdx + v*dzdy, 0.0)
        oro_cond = np.minimum(q, 2e-8 * upslope)
        dq -= oro_cond
        dr += oro_cond
        dt += self.cfg.latent_heating_k_per_q * oro_cond

        return dh, du, dv, dt, dq, dr

    def step(self, s: ExtendedWeatherState) -> ExtendedWeatherState:
        b = self.cfg.base
        dh, du, dv, dt, dq, dr = self.tendencies(s)
        return ExtendedWeatherState(
            np.maximum(s.h + b.dt_s*dh, b.min_height_m),
            s.u + b.dt_s*du,
            s.v + b.dt_s*dv,
            np.clip(s.temperature_k + b.dt_s*dt, 180.0, 330.0),
            np.clip(s.specific_humidity + b.dt_s*dq, 0.0, 0.05),
            np.maximum(s.rain_rate + b.dt_s*dr, 0.0),
            s.terrain_m,
            s.time_s + b.dt_s,
        )

    def run(self, initial: ExtendedWeatherState, hours: float) -> ExtendedWeatherState:
        steps = max(1, int(hours * 3600.0 / self.cfg.base.dt_s))
        s = initial.copy()
        for _ in range(steps):
            s = self.step(s)
        return s


def make_extended_initial(config: ExtendedConfig | None = None, seed: int = 7) -> ExtendedWeatherState:
    c = config or ExtendedConfig()
    b = c.base
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:b.ny, 0:b.nx]
    cx, cy = 0.43*(b.nx-1), 0.52*(b.ny-1)
    r2 = ((xx-cx)/(0.17*b.nx))**2 + ((yy-cy)/(0.20*b.ny))**2
    anomaly = 65.0*np.exp(-0.5*r2)
    h = 10000.0 + anomaly

    mountain = 1400.0*np.exp(-0.5*(((xx-0.72*b.nx)/(0.10*b.nx))**2 + ((yy-0.48*b.ny)/(0.16*b.ny))**2))
    m = ShallowWaterModel(b)
    u = -(b.gravity/b.coriolis_f)*m._ddy(h, b.dx_m)
    v = +(b.gravity/b.coriolis_f)*m._ddx(h, b.dx_m)

    lat_frac = (yy - 0.5*(b.ny-1))/max(b.ny-1, 1)
    temp = 288.0 - 18.0*lat_frac + 2.0*np.exp(-0.5*r2)
    q = 0.010 + 0.004*np.exp(-0.5*r2) + rng.normal(0.0, 1e-4, h.shape)
    q = np.clip(q, 0.0, 0.03)
    rain = np.zeros_like(h)
    return ExtendedWeatherState(h, u, v, temp, q, rain, mountain, 0.0)
