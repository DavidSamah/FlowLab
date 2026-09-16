# FlowLab

An explainable fluid-dynamics research project that now includes two connected tracks:

1. the original 2-D incompressible-flow experiments, and
2. an **AGI-inspired weather intelligence prototype** built around a rotating shallow-water world model.

The weather branch is designed to make the architecture of a general agent concrete: perception, internal state, planning, simulation, ensemble uncertainty, evaluation, and memory are all implemented as inspectable code rather than hidden behind a single prompt.

## Weather intelligence

Architecture guide: [`WEATHER_ARCHITECTURE.md`](WEATHER_ARCHITECTURE.md)

Core package:

```text
weather_agent/
├── state.py   # atmospheric state + forecast records
├── model.py   # rotating shallow-water PDE solver
├── agent.py   # perceive -> plan -> simulate -> evaluate -> remember
└── cli.py     # runnable interface
```

### Run the weather prototype

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements-weather.txt
python -m weather_agent.cli --hours 6
```

Tests:

```bash
pytest -q
```

The CLI prints the initial belief state, explicit plan, forecast diagnostics, ensemble uncertainty, numerical checks, and memory count.

## Existing FlowLab research

The original project contains an explainable 2-D incompressible-flow research prototype with advection, pressure projection, and analytical comparison building blocks under `research/`.

## Scientific scope

This project is a research and learning system, **not** an operational weather service, a solution to the general 3-D Navier–Stokes problem, or a demonstrated AGI. The weather model intentionally uses a much smaller rotating shallow-water system so its assumptions and numerical behavior remain understandable and testable.

A realistic progression is:

```text
idealized rotating fluid
        -> moisture + thermodynamics
        -> real observations
        -> data assimilation
        -> ensemble forecasting
        -> verification against observations
        -> learned/physics hybrid models
```

Live demo for the earlier FlowLab interface: https://flow-lab-research.carinengeny.chatgpt.site
