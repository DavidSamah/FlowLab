"""FlowLab Weather Intelligence.

A compact, inspectable agent architecture wrapped around a rotating
shallow-water atmosphere model. It is an AGI-inspired research prototype,
not a claim of artificial general intelligence or operational NWP.
"""

from .state import WeatherState, ForecastResult
from .model import ShallowWaterModel, ModelConfig, make_vortex_state
from .agent import WeatherAgent, AgentConfig

__all__ = [
    "WeatherState",
    "ForecastResult",
    "ShallowWaterModel",
    "ModelConfig",
    "make_vortex_state",
    "WeatherAgent",
    "AgentConfig",
]
