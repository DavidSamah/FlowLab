from __future__ import annotations

import argparse
import json

from .agent import WeatherAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FlowLab weather-intelligence prototype")
    parser.add_argument("--hours", type=float, default=6.0, help="Forecast horizon in hours")
    parser.add_argument(
        "--goal",
        default="simulate an idealized rotating weather disturbance",
        help="Experiment goal recorded by the agent",
    )
    args = parser.parse_args()

    agent = WeatherAgent()
    output = agent.solve(goal=args.goal, hours=args.hours)
    forecast = output["forecast"]

    serializable = {
        "goal": output["goal"],
        "belief": output["belief"],
        "plan": output["plan"],
        "forecast_diagnostics": forecast.diagnostics,
        "uncertainty": output["uncertainty"],
        "evaluation": output["evaluation"],
        "memory_size": output["memory_size"],
    }
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
