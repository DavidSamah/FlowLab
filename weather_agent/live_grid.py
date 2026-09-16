from __future__ import annotations

"""Initialize FlowLab's PDE atmosphere from a live Open-Meteo spatial sample.

The provider endpoint is point-oriented, so this module samples a configurable
lat/lon mesh, reconstructs 850-hPa wind/geopotential fields, interpolates them
onto FlowLab's numerical grid, and launches a short shallow-water forecast.

This is a research bridge from operational NWP data into FlowLab, not a claim
that the reconstructed field has the resolution or fidelity of the source NWP.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .live_api import LiveWeatherObservation, fetch_live_weather
from .model import ModelConfig, ShallowWaterModel
from .state import WeatherState


@dataclass
class SpatialGridConfig:
    center_latitude: float
    center_longitude: float
    lat_span_deg: float = 4.0
    lon_span_deg: float = 4.0
    rows: int = 5
    cols: int = 5
    forecast_hours: float = 1.0
    max_workers: int = 4


def _wind_components(speed_ms: float | None, direction_deg: float | None) -> tuple[float, float]:
    """Convert meteorological wind direction ('from') to east/north components."""
    if speed_ms is None or direction_deg is None:
        return 0.0, 0.0
    theta = math.radians(float(direction_deg))
    speed = float(speed_ms)
    u = -speed * math.sin(theta)
    v = -speed * math.cos(theta)
    return u, v


def sample_coordinates(cfg: SpatialGridConfig) -> list[tuple[int, int, float, float]]:
    lats = np.linspace(
        cfg.center_latitude - cfg.lat_span_deg / 2.0,
        cfg.center_latitude + cfg.lat_span_deg / 2.0,
        cfg.rows,
    )
    lons = np.linspace(
        cfg.center_longitude - cfg.lon_span_deg / 2.0,
        cfg.center_longitude + cfg.lon_span_deg / 2.0,
        cfg.cols,
    )
    return [
        (iy, ix, float(lat), float(lon))
        for iy, lat in enumerate(lats)
        for ix, lon in enumerate(lons)
    ]


def fetch_spatial_observations(cfg: SpatialGridConfig) -> list[list[LiveWeatherObservation]]:
    grid: list[list[LiveWeatherObservation | None]] = [
        [None for _ in range(cfg.cols)] for _ in range(cfg.rows)
    ]

    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
        futures = {
            pool.submit(fetch_live_weather, lat, lon): (iy, ix, lat, lon)
            for iy, ix, lat, lon in sample_coordinates(cfg)
        }
        for future in as_completed(futures):
            iy, ix, lat, lon = futures[future]
            try:
                grid[iy][ix] = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Open-Meteo spatial sample failed at ({lat:.4f}, {lon:.4f}): {exc}"
                ) from exc

    return [[cell for cell in row if cell is not None] for row in grid]


def _interp_2d(field: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Dependency-light bilinear-style interpolation via two 1-D passes."""
    sy, sx = field.shape
    old_x = np.linspace(0.0, 1.0, sx)
    new_x = np.linspace(0.0, 1.0, nx)
    x_resampled = np.vstack([np.interp(new_x, old_x, row) for row in field])

    old_y = np.linspace(0.0, 1.0, sy)
    new_y = np.linspace(0.0, 1.0, ny)
    out = np.empty((ny, nx), dtype=float)
    for ix in range(nx):
        out[:, ix] = np.interp(new_y, old_y, x_resampled[:, ix])
    return out


def observations_to_state(
    observations: list[list[LiveWeatherObservation]],
    model_cfg: ModelConfig | None = None,
) -> tuple[WeatherState, dict[str, np.ndarray]]:
    cfg = model_cfg or ModelConfig()
    rows = len(observations)
    cols = len(observations[0])

    z = np.empty((rows, cols), dtype=float)
    u = np.empty((rows, cols), dtype=float)
    v = np.empty((rows, cols), dtype=float)
    temperature = np.empty((rows, cols), dtype=float)
    humidity = np.empty((rows, cols), dtype=float)
    vertical_velocity = np.empty((rows, cols), dtype=float)

    for iy, row in enumerate(observations):
        if len(row) != cols:
            raise ValueError("Observation grid is ragged")
        for ix, obs in enumerate(row):
            z[iy, ix] = float(obs.geopotential_height_850hpa_m or 1500.0)
            u[iy, ix], v[iy, ix] = _wind_components(
                obs.wind_speed_850hpa_ms,
                obs.wind_direction_850hpa_deg,
            )
            temperature[iy, ix] = float(obs.temperature_850hpa_c or 0.0)
            humidity[iy, ix] = float(obs.relative_humidity_850hpa_pct or 0.0)
            vertical_velocity[iy, ix] = float(obs.vertical_velocity_850hpa_pas or 0.0)

    # The shallow-water model's h is a pressure/height-like layer around 10 km.
    # Preserve the live geopotential anomaly while shifting to a stable base state.
    z_anomaly = z - float(np.mean(z))
    h_sample = 10_000.0 + z_anomaly

    state = WeatherState(
        h=_interp_2d(h_sample, cfg.ny, cfg.nx),
        u=_interp_2d(u, cfg.ny, cfg.nx),
        v=_interp_2d(v, cfg.ny, cfg.nx),
        time_s=0.0,
    )
    auxiliary = {
        "temperature_850hpa_c": _interp_2d(temperature, cfg.ny, cfg.nx),
        "relative_humidity_850hpa_pct": _interp_2d(humidity, cfg.ny, cfg.nx),
        "vertical_velocity_850hpa_pas": _interp_2d(vertical_velocity, cfg.ny, cfg.nx),
        "geopotential_height_850hpa_m": _interp_2d(z, cfg.ny, cfg.nx),
    }
    return state, auxiliary


def run_live_grid(cfg: SpatialGridConfig, artifact_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = fetch_spatial_observations(cfg)
    model_cfg = ModelConfig()
    state, auxiliary = observations_to_state(observations, model_cfg)

    raw_points = [obs.to_dict() for row in observations for obs in row]
    raw_path = output_dir / "live-grid-points.json"
    raw_path.write_text(json.dumps(raw_points, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        output_dir / "live-initial-condition.npz",
        h=state.h,
        u=state.u,
        v=state.v,
        **auxiliary,
    )

    model = ShallowWaterModel(model_cfg)
    forecast = model.run(state, hours=cfg.forecast_hours, save_every=15)
    np.savez_compressed(
        output_dir / "live-forecast-final.npz",
        h=forecast.final.h,
        u=forecast.final.u,
        v=forecast.final.v,
    )

    speed0 = np.sqrt(state.u**2 + state.v**2)
    speed1 = np.sqrt(forecast.final.u**2 + forecast.final.v**2)
    summary: dict[str, Any] = {
        "provider": "Open-Meteo",
        "mode": "spatial-grid-to-PDE-initial-condition",
        "sampling": asdict(cfg),
        "sample_count": cfg.rows * cfg.cols,
        "model_grid": {"ny": model_cfg.ny, "nx": model_cfg.nx, "dx_m": model_cfg.dx_m},
        "initial_state": {
            "mean_height_m": float(np.mean(state.h)),
            "height_std_m": float(np.std(state.h)),
            "mean_wind_ms": float(np.mean(speed0)),
            "max_wind_ms": float(np.max(speed0)),
            "mean_temperature_850hpa_c": float(np.mean(auxiliary["temperature_850hpa_c"])),
            "mean_relative_humidity_850hpa_pct": float(np.mean(auxiliary["relative_humidity_850hpa_pct"])),
            "mean_vertical_velocity_850hpa_pas": float(np.mean(auxiliary["vertical_velocity_850hpa_pas"])),
        },
        "forecast": {
            **forecast.diagnostics,
            "final_mean_wind_ms": float(np.mean(speed1)),
            "final_max_wind_ms": float(np.max(speed1)),
        },
        "artifacts": [
            "live-grid-points.json",
            "live-initial-condition.npz",
            "live-forecast-final.npz",
        ],
        "scientific_boundary": (
            "The source samples are real operational-model data, but FlowLab reconstructs "
            "a coarse 2-D field and evolves an educational shallow-water analogue."
        ),
    }
    (output_dir / "live-grid-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize FlowLab from a live spatial atmosphere grid")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--lat-span", type=float, default=4.0)
    parser.add_argument("--lon-span", type=float, default=4.0)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--forecast-hours", type=float, default=1.0)
    parser.add_argument("--artifact-dir", default="artifacts/live-grid")
    args = parser.parse_args()

    cfg = SpatialGridConfig(
        center_latitude=args.latitude,
        center_longitude=args.longitude,
        lat_span_deg=args.lat_span,
        lon_span_deg=args.lon_span,
        rows=args.rows,
        cols=args.cols,
        forecast_hours=args.forecast_hours,
    )
    result = run_live_grid(cfg, args.artifact_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
