from __future__ import annotations

"""Live external atmosphere adapter for FlowLab.

Provider: Open-Meteo (https://open-meteo.com/)
The free non-commercial endpoint requires no API key. Open-Meteo aggregates and
normalizes numerical weather prediction data from national weather services.
Data attribution is required by the provider's CC BY 4.0 terms.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class LiveWeatherObservation:
    provider: str
    latitude: float
    longitude: float
    fetched_at_utc: str
    model_timezone: str
    temperature_2m_c: float | None
    relative_humidity_2m_pct: float | None
    surface_pressure_hpa: float | None
    precipitation_mm: float | None
    wind_speed_10m_ms: float | None
    wind_direction_10m_deg: float | None
    wind_gusts_10m_ms: float | None
    temperature_850hpa_c: float | None
    relative_humidity_850hpa_pct: float | None
    wind_speed_850hpa_ms: float | None
    wind_direction_850hpa_deg: float | None
    vertical_velocity_850hpa_pas: float | None
    geopotential_height_850hpa_m: float | None
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kmh_to_ms(value: float | None) -> float | None:
    return None if value is None else float(value) / 3.6


def _first(hourly: dict[str, Any], key: str) -> float | None:
    values = hourly.get(key)
    if not values:
        return None
    value = values[0]
    return None if value is None else float(value)


def build_url(latitude: float, longitude: float) -> str:
    current = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]
    hourly = [
        "temperature_850hPa",
        "relative_humidity_850hPa",
        "wind_speed_850hPa",
        "wind_direction_850hPa",
        "vertical_velocity_850hPa",
        "geopotential_height_850hPa",
    ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(current),
        "hourly": ",".join(hourly),
        "forecast_days": 1,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    return f"{BASE_URL}?{urlencode(params)}"


def parse_response(payload: dict[str, Any], source_url: str) -> LiveWeatherObservation:
    current = payload.get("current", {})
    hourly = payload.get("hourly", {})
    return LiveWeatherObservation(
        provider="Open-Meteo",
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        model_timezone=str(payload.get("timezone", "UTC")),
        temperature_2m_c=current.get("temperature_2m"),
        relative_humidity_2m_pct=current.get("relative_humidity_2m"),
        surface_pressure_hpa=current.get("surface_pressure"),
        precipitation_mm=current.get("precipitation"),
        wind_speed_10m_ms=_kmh_to_ms(current.get("wind_speed_10m")),
        wind_direction_10m_deg=current.get("wind_direction_10m"),
        wind_gusts_10m_ms=_kmh_to_ms(current.get("wind_gusts_10m")),
        temperature_850hpa_c=_first(hourly, "temperature_850hPa"),
        relative_humidity_850hpa_pct=_first(hourly, "relative_humidity_850hPa"),
        wind_speed_850hpa_ms=_kmh_to_ms(_first(hourly, "wind_speed_850hPa")),
        wind_direction_850hpa_deg=_first(hourly, "wind_direction_850hPa"),
        vertical_velocity_850hpa_pas=_first(hourly, "vertical_velocity_850hPa"),
        geopotential_height_850hpa_m=_first(hourly, "geopotential_height_850hPa"),
        source_url=source_url,
    )


def fetch_live_weather(latitude: float, longitude: float, timeout_s: float = 20.0) -> LiveWeatherObservation:
    url = build_url(latitude, longitude)
    request = Request(url, headers={"User-Agent": "FlowLab-Weather-Research/1.0"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_response(payload, url)


def fetch_to_file(latitude: float, longitude: float, output: str | Path) -> LiveWeatherObservation:
    observation = fetch_live_weather(latitude, longitude)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observation.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return observation
