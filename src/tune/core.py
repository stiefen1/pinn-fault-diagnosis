"""Optuna hyperparameter search driven by configs/tune.yaml.

All parameters (storage, study name, n_trials, search space) live in the
config file. The only CLI argument is --config.

Usage (local):
    python scripts/tune.py --config configs/tune.yaml

Usage (SLURM array — launched via launch_slurm.py):
    python scripts/launch_slurm.py --config configs/tune.yaml --submit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna
from optuna.storages import JournalFileStorage

from src.utils.configs import load_config, apply_overrides
from src.train import train


def sample_config(trial: optuna.Trial, cfg: dict) -> dict:
    """Sample one trial's hyperparameters from the search_space defined in the config."""
    search_space = cfg["hyperparameter_search"]["search_space"]
    overrides = []

    for param_path, spec in search_space.items():
        kind = spec["type"]
        match kind:
            case "categorical":
                values = [str(v) if isinstance(v, list) else v for v in spec["values"]] # list arg parsed as string
                value = trial.suggest_categorical(param_path, values)
            case "loguniform":
                value = trial.suggest_float(param_path, spec["low"], spec["high"], log=True)
            case "uniform":
                value = trial.suggest_float(param_path, spec["low"], spec["high"])
            case "int":
                value = trial.suggest_int(param_path, spec["low"], spec["high"])
            case _:
                raise ValueError(f"Unknown search space type {kind!r} for param {param_path!r}")
        overrides.append(f"{param_path}={value}")

    cfg = apply_overrides(cfg, overrides)
    cfg["project"]["experiment_name"] = f"{trial.study.study_name}_trial_{trial.number:04d}"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/tune.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    hpo = cfg["hyperparameter_search"]

    study_name = hpo["study_name"]
    n_trials = hpo["n_trials"]
    direction = hpo["objective"]

    storage_raw = hpo.get("storage")
    if storage_raw:
        storage_path = Path(storage_raw)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = (
            f"sqlite:///{storage_path.as_posix()}"
            if storage_raw.endswith(".db")
            else JournalFileStorage(str(storage_path))
        )
    else:
        storage = None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,  # type: ignore
        direction=direction,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: train(sample_config(trial, cfg)),
        n_trials=n_trials,
    )

    print("\n--- Best trial ---")
    best = study.best_trial
    print(f"  value : {best.value:.6f}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
