"""Generate (and optionally submit) a SLURM batch script from a config file.

All job parameters come from the config's ``hpc`` section — nothing is
hardcoded here. Use a different config file for training vs. tuning vs. testing.

Usage
-----
    python scripts/launch_slurm.py --config configs/train.yaml --dry-run
    python scripts/launch_slurm.py --config configs/train.yaml --submit
    python scripts/launch_slurm.py --config configs/tune.yaml  --submit
"""

import argparse, json, re, subprocess, sys
from itertools import product as cartesian
from pathlib import Path
from src.utils.configs import load_config

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_combinations(params: dict) -> list[dict]:
    """Return all cartesian-product combinations of the param sweep lists."""
    if not params:
        return []
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, combo)) for combo in cartesian(*values)]


def _resolve_refs(text: str, cfg: dict) -> str:
    """Replace ``${section.key}`` placeholders with values from *cfg*."""

    def _get(node: dict, keys: list[str]):
        for k in keys:
            node = node[k]
        return node

    def _replacer(match: re.Match) -> str:
        path = match.group(1).split(".")
        try:
            return str(_get(cfg, path))
        except (KeyError, TypeError):
            return match.group(0)  # leave unresolved refs intact

    return re.sub(r"\$\{([^}]+)\}", _replacer, text)


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def build_slurm_script(cfg: dict) -> str:
    hpc = cfg["hpc"]
    slurm_cfg = hpc.get("slurm_script", {})
    env_cfg = hpc.get("environment", {})
    array_cfg = hpc.get("array", {})

    output_dir = slurm_cfg.get("output_dir", "slurm_files/").rstrip("/")
    job_name = slurm_cfg.get("job_name", "job")
    is_array = array_cfg.get("enabled", False)
    array_params = array_cfg.get("params", {})
    combos = compute_combinations(array_params) if is_array else []
    array_size = len(combos) if combos else array_cfg.get("size", 1)
    log_suffix = "%A_%a" if is_array else "%j"

    lines: list[str] = ["#!/bin/bash", ""]

    # ---- SBATCH directives ------------------------------------------------
    if is_array:
        lines.append(f"#SBATCH --array=0-{array_size - 1}")

    lines += [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={hpc['partition']}",
        f"#SBATCH --time={hpc['time']}",
        f"#SBATCH --nodes={hpc['nodes']}",
        f"#SBATCH --ntasks={hpc['ntasks']}",
        f"#SBATCH --cpus-per-task={hpc['cpus_per_task']}",
        f"#SBATCH --gpus={hpc['gpus']}",
        f"#SBATCH --mem={hpc['mem']}",
        f"#SBATCH --output={output_dir}/{log_suffix}.out",
        f"#SBATCH --error={output_dir}/{log_suffix}.err",
    ]

    for extra in slurm_cfg.get("extra_sbatch_args", []):
        lines.append(f"#SBATCH {extra}")

    lines.append("")

    # ---- Environment setup -----------------------------------------------
    modules = env_cfg.get("modules", [])
    if modules:
        lines.append("# Load modules")
        lines.append("module purge")
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")

    conda_env = env_cfg.get("conda_env")
    if conda_env:
        lines += [
            "# Activate conda environment",
            'eval "$(conda shell.bash hook)"',
            f"conda activate {conda_env}",
            "",
        ]

    # ---- Diagnostics (logged to .out before training starts) -------------
    lines += [
        "# Diagnostics",
        "echo '=== nvidia-smi ==='",
        "nvidia-smi",
        "echo '=== Python/PyTorch ==='",
        "python -c \"import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)\"",
        "",
    ]

    # ---- Launch command ---------------------------------------------------
    raw_command = hpc["launcher"]["command"].strip()
    launch_command = _resolve_refs(raw_command, cfg)
    if combos:
        combos_path = Path(output_dir) / "combinations.json"
        launch_command += f" --combinations {combos_path}"
    lines += ["# Launch", launch_command, ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (and optionally submit) a SLURM job script from a config file."
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training config (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the generated .slurm file. "
             "Defaults to <output_dir>/<job_name>.slurm as defined in the config.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the generated script immediately via sbatch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated script to stdout without writing or submitting.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    script = build_slurm_script(cfg)

    if args.dry_run:
        print(script)
        return

    # Determine output path
    hpc = cfg["hpc"]
    slurm_cfg = hpc.get("slurm_script", {})
    output_dir = Path(slurm_cfg.get("output_dir", "slurm_files"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or output_dir / f"{slurm_cfg.get('job_name', 'train')}.slurm"
    output_path.write_text(script, encoding="utf-8")
    print(f"Script written to: {output_path}")

    # Write combinations file alongside the script (needed at job runtime)
    array_cfg = cfg["hpc"].get("array", {})
    if array_cfg.get("enabled") and array_cfg.get("params"):
        combos = compute_combinations(array_cfg["params"])
        combos_path = output_dir / "combinations.json"
        combos_path.write_text(json.dumps(combos, indent=2))
        print(f"Combinations ({len(combos)} total) written to: {combos_path}")

    if args.submit:
        result = subprocess.run(
            ["sbatch", str(output_path)],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
