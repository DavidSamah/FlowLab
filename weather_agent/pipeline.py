from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .advanced import CoupledWeatherModel, ExtendedConfig, make_extended_initial
from .assimilation import sample_observations, assimilate_nudging, observation_rmse
from .ensemble import run_ensemble, field_rmse
from .emulator import train_emulator


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def stage_physics() -> dict:
    cfg = ExtendedConfig()
    model = CoupledWeatherModel(cfg)
    initial = make_extended_initial(cfg)
    final = model.run(initial, hours=0.5)
    di = initial.diagnostics()
    df = final.diagnostics()
    return {
        "stage": "physics",
        "description": "beta-plane shallow-water dynamics with terrain, diffusion and Coriolis rotation",
        "initial": di,
        "final": df,
        "relative_mass_drift": (df["mass_proxy"]-di["mass_proxy"])/max(abs(di["mass_proxy"]), 1e-12),
        "finite": bool(np.isfinite(final.h).all() and np.isfinite(final.u).all() and np.isfinite(final.v).all()),
    }


def stage_moisture() -> dict:
    cfg = ExtendedConfig()
    model = CoupledWeatherModel(cfg)
    initial = make_extended_initial(cfg)
    initial.specific_humidity *= 1.35
    final = model.run(initial, hours=1.0)
    return {
        "stage": "moisture",
        "description": "temperature advection, humidity, condensation, latent heating, rain fallout and orographic lift",
        "initial": initial.diagnostics(),
        "final": final.diagnostics(),
        "produced_rain": bool(np.max(final.rain_rate) > 0.0),
    }


def stage_terrain() -> dict:
    cfg = ExtendedConfig()
    model = CoupledWeatherModel(cfg)
    state = make_extended_initial(cfg)
    f = model.coriolis_field()
    return {
        "stage": "terrain-coriolis",
        "description": "terrain geopotential forcing and latitude-varying beta-plane Coriolis parameter",
        "terrain_min_m": float(np.min(state.terrain_m)),
        "terrain_max_m": float(np.max(state.terrain_m)),
        "coriolis_min_s-1": float(np.min(f)),
        "coriolis_max_s-1": float(np.max(f)),
    }


def _assimilation_case(gain: float = 0.35):
    cfg = ExtendedConfig()
    model = CoupledWeatherModel(cfg)
    truth0 = make_extended_initial(cfg, seed=2)
    truth = model.run(truth0, hours=0.5)
    background0 = make_extended_initial(cfg, seed=19)
    background0.h += 10.0
    background0.temperature_k += 1.0
    background = model.run(background0, hours=0.5)
    obs = sample_observations(truth, count=96, seed=12)
    before = observation_rmse(background, obs)
    analysis = assimilate_nudging(background, obs, gain=gain, radius=3)
    after = observation_rmse(analysis, obs)
    return cfg, model, truth, background, obs, analysis, before, after


def stage_observations() -> dict:
    _, _, truth, _, obs, _, _, _ = _assimilation_case()
    return {
        "stage": "observations",
        "description": "sparse noisy station-like observations with explicit error assumptions",
        "count": int(len(obs.x)),
        "error_std": obs.error_std,
        "truth_diagnostics": truth.diagnostics(),
    }


def stage_assimilation() -> dict:
    _, _, _, _, _, _, before, after = _assimilation_case()
    improvement = {k: before[k]-after[k] for k in before}
    return {
        "stage": "assimilation",
        "description": "localized observation-minus-background nudging analysis",
        "before_rmse": before,
        "after_rmse": after,
        "rmse_improvement": improvement,
    }


def stage_ensemble() -> dict:
    cfg, model, truth, _, _, analysis, _, _ = _assimilation_case()
    ensemble = run_ensemble(model, analysis, hours=0.5, members=5)
    verification_truth = model.run(truth, hours=0.5)
    rmse = field_rmse(ensemble.mean, verification_truth)
    return {
        "stage": "ensemble",
        "description": "perturbed-initial-condition ensemble forecast and ensemble-mean verification",
        "members": len(ensemble.members),
        "spread": ensemble.spread,
        "verification_rmse": rmse,
    }


def stage_emulator() -> dict:
    cfg = ExtendedConfig()
    model = CoupledWeatherModel(cfg)
    initial = make_extended_initial(cfg)
    _, metrics = train_emulator(model, initial, samples=48)
    return {
        "stage": "emulator",
        "description": "ridge-regression learned surrogate of one-step diagnostic evolution",
        "metrics": metrics,
        "boundary": "diagnostic emulator only; it is not a gridded operational forecast replacement",
    }


def stage_verification() -> dict:
    cfg, model, truth, background, _, analysis, _, _ = _assimilation_case()
    lead_h = 0.5
    verifying_truth = model.run(truth, hours=lead_h)
    background_fc = model.run(background, hours=lead_h)
    analysis_fc = model.run(analysis, hours=lead_h)
    b = field_rmse(background_fc, verifying_truth)
    a = field_rmse(analysis_fc, verifying_truth)
    return {
        "stage": "verification",
        "description": "compare background-forecast and analysis-forecast against a held-out synthetic truth",
        "background_forecast_rmse": b,
        "analysis_forecast_rmse": a,
        "analysis_minus_background": {k: a[k]-b[k] for k in a},
    }


def stage_learning() -> dict:
    candidates = [0.15, 0.30, 0.45, 0.60]
    scores = {}
    for gain in candidates:
        _, _, _, _, _, _, _, after = _assimilation_case(gain=gain)
        score = float(np.mean([after["h_rmse"]/3.0, after["u_rmse"], after["v_rmse"], after["temperature_rmse"], after["humidity_rmse"]*1000.0]))
        scores[str(gain)] = score
    best = min(scores, key=scores.get)
    return {
        "stage": "learning",
        "description": "forecast-system self-tuning by selecting the assimilation gain with the lowest validation score",
        "candidate_scores": scores,
        "selected_assimilation_gain": float(best),
        "scope": "bounded parameter learning; not autonomous self-modification of arbitrary code",
    }


STAGES = {
    "physics": stage_physics,
    "moisture": stage_moisture,
    "terrain": stage_terrain,
    "observations": stage_observations,
    "assimilation": stage_assimilation,
    "ensemble": stage_ensemble,
    "emulator": stage_emulator,
    "verification": stage_verification,
    "learning": stage_learning,
}


def run_stage(name: str) -> dict:
    return STAGES[name]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an inspectable FlowLab weather-intelligence stage")
    parser.add_argument("--stage", choices=[*STAGES.keys(), "all"], default="all")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.stage == "all":
        result = {name: fn() for name, fn in STAGES.items()}
    else:
        result = run_stage(args.stage)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
