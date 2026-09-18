from __future__ import annotations

import argparse
import json

from .live_api import fetch_to_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live Open-Meteo atmospheric data for FlowLab")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    obs = fetch_to_file(args.latitude, args.longitude, args.out)
    print(json.dumps(obs.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
