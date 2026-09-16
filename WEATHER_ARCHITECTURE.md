# FlowLab Weather Intelligence Architecture

This branch extends FlowLab with an **AGI-inspired weather research architecture**. It is not AGI and it is not an operational weather forecasting system. Its purpose is to make the core loop of a general agent concrete and testable against a physical dynamical system.

## Master loop

```text
Goal / experiment question
        |
        v
Perception / state estimation
        |
        v
Internal belief state
        |
        v
Planner ------------------------+
        |                        |
        v                        |
Physics world model              |
(rotating shallow-water PDE)     |
        |                        |
        v                        |
Deterministic simulation         |
        |                        |
        +--> perturbed ensemble -+
        |                        |
        v                        |
Uncertainty estimate             |
        |                        |
        v                        |
Evaluator / sanity checks        |
        |                        |
        v                        |
Memory / experiment record       |
        |                        |
        +------ feedback --------+
```

## Mapping to a general-agent architecture

| General-agent component | FlowLab implementation |
|---|---|
| Perception | `WeatherAgent.perceive()` summarizes a gridded atmospheric state |
| Internal representation | `WeatherState(h, u, v, time_s)` |
| Working memory | Structured dictionary returned by each solve cycle |
| Episodic memory | `MemoryRecord` list stored by the agent |
| World model | `ShallowWaterModel` |
| Imagination | Deterministic and perturbed future rollouts |
| Planning | Explicit action list from `WeatherAgent.plan()` |
| Action | Numerical integration of the atmosphere |
| Uncertainty | Ensemble spread from perturbed initial conditions |
| Evaluation | Stability, finite-value and conservation checks |
| Learning placeholder | Memory is stored; parameter learning is intentionally not automated yet |

## Physical model

The prototype uses the rotating shallow-water equations. In simplified form:

```text
Du/Dt - f v = -g dh/dx + diffusion
Dv/Dt + f u = -g dh/dy + diffusion
Dh/Dt + div(h U) = diffusion
```

where:

- `h` is fluid-layer height, used as a pressure-like field,
- `u` and `v` are horizontal wind components,
- `g` is gravity,
- `f` is the Coriolis parameter,
- nonlinear advection transports momentum,
- diffusion provides a simple unresolved-scale dissipation model.

The solver uses periodic boundaries and midpoint RK2 time integration.

## Why this is weather-like

Large-scale atmospheric motion is governed by rotating fluid dynamics. The shallow-water equations retain several important mechanisms:

- pressure-gradient acceleration,
- Coriolis deflection,
- nonlinear advection,
- wave propagation,
- conservation behavior,
- sensitivity to initial conditions.

They omit the full vertical atmosphere, moisture, radiation, clouds, topography, chemistry, data assimilation, spherical geometry, and many sub-grid processes used in operational numerical weather prediction.

## Uncertainty mechanism

One deterministic forecast is not enough for chaotic systems. The agent therefore perturbs the initial pressure-like and wind fields, runs multiple forecasts, and measures their spread.

Conceptually:

```text
observed state
   |-- perturbation 1 --> future 1
   |-- perturbation 2 --> future 2
   |-- perturbation 3 --> future 3
   `-- ...

ensemble disagreement -> uncertainty proxy
```

## Repository structure

```text
FlowLab/
├── app/                      existing FlowLab UI
├── research/                 existing incompressible-flow experiments
├── weather_agent/
│   ├── __init__.py
│   ├── state.py              atmospheric state and forecast types
│   ├── model.py              rotating shallow-water world model
│   ├── agent.py              perceive/plan/simulate/evaluate/memory loop
│   └── cli.py                runnable command-line interface
├── tests/
│   └── test_weather_agent.py
├── requirements-weather.txt
└── WEATHER_ARCHITECTURE.md
```

## Run

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements-weather.txt
python -m weather_agent.cli --hours 6
```

Run tests:

```bash
pytest -q
```

## Realistic next steps

1. Add latitude-dependent Coriolis (`beta` plane).
2. Add thermodynamics and moisture.
3. Add observed initial conditions from public meteorological datasets.
4. Add data assimilation to combine observations and model state.
5. Add learned emulators to compare neural and physics forecasts.
6. Add verification metrics against held-out observations.
7. Add spatial visualization to the existing web application.
8. Persist experiment memory in SQLite or PostgreSQL.
9. Add a controller that selects model resolution and ensemble size from uncertainty and compute budget.
10. Keep human-readable diagnostics and falsifiable tests at every layer.

## Scientific boundary

The architecture is deliberately transparent. A successful run means the numerical prototype executed and passed its internal checks. It does **not** mean it accurately forecasts real weather, solves the full Navier-Stokes equations, or demonstrates AGI.
