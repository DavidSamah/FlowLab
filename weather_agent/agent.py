from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .model import ShallowWaterModel, ModelConfig, make_vortex_state
from .state import WeatherState, ForecastResult


@dataclass
class AgentConfig:
    ensemble_members: int = 6
    perturbation_height_m: float = 2.0
    perturbation_wind_ms: float = 0.2
    random_seed: int = 7
    instability_wind_threshold_ms: float = 120.0
    mass_drift_threshold: float = 1e-3


@dataclass
class MemoryRecord:
    goal: str
    forecast_hours: float
    diagnostics: dict[str, float]
    notes: list[str] = field(default_factory=list)


class WeatherAgent:
    """AGI-inspired observe-model-simulate-evaluate-memory loop for weather experiments."""

    def __init__(
        self,
        model: ShallowWaterModel | None = None,
        config: AgentConfig | None = None,
    ):
        self.model = model or ShallowWaterModel(ModelConfig())
        self.cfg = config or AgentConfig()
        self.memory: list[MemoryRecord] = []
        self.rng = np.random.default_rng(self.cfg.random_seed)

    def perceive(self, state: WeatherState) -> dict[str, float]:
        """Convert the numerical atmosphere into an inspectable belief state."""
        d = state.diagnostics()
        d["height_range_m"] = float(np.max(state.h) - np.min(state.h))
        return d

    def plan(self, goal: str, horizon_hours: float) -> dict[str, Any]:
        """Create a simple explicit plan rather than hide decisions in a black box."""
        return {
            "goal": goal,
            "horizon_hours": float(horizon_hours),
            "actions": [
                "validate initial state",
                "run deterministic forecast",
                "run perturbed ensemble",
                "estimate uncertainty",
                "check conservation and numerical stability",
                "store experiment memory",
            ],
        }

    def _perturb(self, state: WeatherState) -> WeatherState:
        return WeatherState(
            np.maximum(
                state.h + self.rng.normal(0.0, self.cfg.perturbation_height_m, state.h.shape),
                self.model.cfg.min_height_m,
            ),
            state.u + self.rng.normal(0.0, self.cfg.perturbation_wind_ms, state.u.shape),
            state.v + self.rng.normal(0.0, self.cfg.perturbation_wind_ms, state.v.shape),
            state.time_s,
        )

    def simulate_ensemble(self, initial: WeatherState, hours: float) -> dict[str, float]:
        finals = []
        for _ in range(self.cfg.ensemble_members):
            result = self.model.run(self._perturb(initial), hours=hours)
            finals.append(result.final)

        h_stack = np.stack([s.h for s in finals])
        wind_stack = np.stack([np.sqrt(s.u**2 + s.v**2) for s in finals])
        return {
            "mean_height_spread_m": float(np.mean(np.std(h_stack, axis=0))),
            "max_height_spread_m": float(np.max(np.std(h_stack, axis=0))),
            "mean_wind_spread_ms": float(np.mean(np.std(wind_stack, axis=0))),
            "max_wind_spread_ms": float(np.max(np.std(wind_stack, axis=0))),
        }

    def evaluate(self, result: ForecastResult) -> list[str]:
        notes: list[str] = []
        if abs(result.diagnostics["relative_mass_drift"]) > self.cfg.mass_drift_threshold:
            notes.append("mass-conservation warning")
        if result.diagnostics["max_wind_ms"] > self.cfg.instability_wind_threshold_ms:
            notes.append("possible numerical-instability warning")
        if not np.isfinite(result.final.h).all():
            notes.append("non-finite height values detected")
        if not np.isfinite(result.final.u).all() or not np.isfinite(result.final.v).all():
            notes.append("non-finite wind values detected")
        if not notes:
            notes.append("forecast passed basic numerical sanity checks")
        return notes

    def solve(self, goal: str = "simulate an idealized weather disturbance", hours: float = 6.0,
              initial: WeatherState | None = None) -> dict[str, Any]:
        """Run the complete agent loop and return transparent structured output."""
        initial = initial or make_vortex_state(self.model.cfg)
        belief = self.perceive(initial)
        plan = self.plan(goal, hours)
        forecast = self.model.run(initial, hours=hours)
        spread = self.simulate_ensemble(initial, hours)
        forecast.ensemble_spread = spread
        evaluation = self.evaluate(forecast)

        self.memory.append(
            MemoryRecord(
                goal=goal,
                forecast_hours=hours,
                diagnostics={**forecast.diagnostics, **spread},
                notes=evaluation,
            )
        )
        return {
            "goal": goal,
            "belief": belief,
            "plan": plan,
            "forecast": forecast,
            "uncertainty": spread,
            "evaluation": evaluation,
            "memory_size": len(self.memory),
        }
