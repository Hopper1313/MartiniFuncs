"""Compare full-frame MD forces with a supplied ASE calculator.

Run ``python -m MartiniFuncs.force_validation --help`` for a local UMA check.
No checkpoint is bundled or downloaded. Matching forces validates reproduction,
not the accuracy of UMA or the suitability of MD forces as an external load.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from time import perf_counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .lammpsio import iter_lammps_frames, read_lammps_input_settings


# Confirmed settings for the collaboration's uma-s-1p1 MD runs.
MD_INFERENCE_SETTINGS = {
    "tf32": True,
    "merge_mole": True,
    "compile": True,
    "activation_checkpointing": False,
    "internal_graph_gen_version": 2,
    "external_graph_gen": False,
}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in sorted(value)] if isinstance(value, set) else [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def compare_frame_forces(frame, calculator, *, charge, spin, atol):
    """Evaluate the full source frame; tolerance is absolute eV/Angstrom.

    No constraints, wrapping, relaxation, or net-force/torque projection is
    applied. ``passed`` requires every Cartesian component to satisfy ``atol``.
    The source Atoms and calculator association are left unchanged.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if not isinstance(charge, (int, np.integer)) or not isinstance(spin, (int, np.integer)) or spin < 1:
        raise ValueError("charge must be an integer and spin a positive multiplicity")
    if "lammps_forces" not in frame.atoms.arrays:
        raise ValueError("Frame has no source forces")
    if frame.atoms.info.get("force_units") != "eV/Angstrom":
        raise ValueError("Source forces must be normalized to eV/Angstrom")
    atoms = frame.atoms.copy()
    atoms.set_constraint()
    atoms.info.update(charge=int(charge), spin=int(spin))
    atoms.calc = calculator
    reference = np.asarray(frame.atoms.arrays["lammps_forces"])
    predicted = np.asarray(atoms.get_forces(apply_constraint=False))
    if predicted.shape != reference.shape or not np.isfinite(predicted).all():
        raise ValueError("Calculator returned invalid force shape or non-finite forces")
    error = predicted - reference
    norms = np.linalg.norm(error, axis=1)
    worst = int(np.argmax(norms))
    return {
        "source": frame.atoms.info["source"], "frame_index": frame.index,
        "timestep": frame.timestep, "n_atoms": len(atoms),
        "lammps_units": frame.units, "force_units": "eV/Angstrom",
        "charge": int(charge), "spin": int(spin), "atol": float(atol),
        "passed": bool(np.all(np.abs(error) <= atol)),
        "component_rmse": float(np.sqrt(np.mean(error**2))),
        "component_mae": float(np.mean(np.abs(error))),
        "max_component_error": float(np.max(np.abs(error))),
        "max_atom_error_norm": float(norms[worst]),
        "worst_lammps_id": int(atoms.arrays["lammps_id"][worst]),
        "reference_net_force": reference.sum(axis=0).tolist(),
        "predicted_net_force": predicted.sum(axis=0).tolist(),
    }


def local_uma_calculator(checkpoint, *, task="omol", device="cuda", inference_settings=None):
    """Load a user-owned model once, with the confirmed MD settings by default.

    ``inference_settings`` may be a mapping of overrides or a fairchem preset
    name. No Hydra target is executed. Energy offsets are unnecessary for this
    force-only comparison. The tested interface is fairchem-core 2.22.0.
    """
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("checkpoint must be a local file")
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, guess_inference_settings

    if inference_settings is None or isinstance(inference_settings, Mapping):
        options = {} if inference_settings is None else dict(inference_settings)
        if "if_settings" in options:
            if len(options) != 1:
                raise ValueError("if_settings must be the only top-level configuration key")
            options = dict(options["if_settings"])
        target = options.pop("_target_", None)
        if target not in (None, "fairchem.core.units.mlip_unit.api.inference.InferenceSettings"):
            raise ValueError("Unexpected inference settings target")
        settings = InferenceSettings(**(MD_INFERENCE_SETTINGS | options))
    else:
        settings = guess_inference_settings(inference_settings)
    predictor = load_predict_unit(checkpoint, device=device, inference_settings=settings)
    calculator = FAIRChemCalculator(predictor, task_name=task)
    calculator.martini_inference_settings = _json_safe(settings.to_omegaconf())
    return calculator


def validate_forces(dump, checkpoint, *, input_file=None, datafile=None, units=None,
                    indices=(0,), charge, spin, atol, task="omol", device="cuda",
                    inference_settings=None):
    """Reproduce source forces for selected full periodic frames with UMA.

    Supply a local uma-s-1p1 checkpoint and a dump with fx/fy/fz. Units come from
    ``input_file`` or the explicit ``units`` argument; conflicting values fail.
    Indices are zero-based nonnegative frame numbers, evaluated in file order.
    Defaults to frame 0. The calculator is loaded once; only one frame is held
    at a time. Timings separate loading and per-frame evaluation (the first
    evaluation can include compilation). All components must satisfy ``atol``
    in eV/Angstrom. Returns a JSON-compatible report; does not write files.
    """
    started = perf_counter()
    requested = list(indices)
    if not requested or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in requested):
        raise ValueError("indices must contain nonnegative integer frame numbers")
    if len(set(requested)) != len(requested):
        raise ValueError("indices must not contain duplicates")
    requested = sorted(int(i) for i in requested)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if not isinstance(charge, (int, np.integer)) or not isinstance(spin, (int, np.integer)) or spin < 1:
        raise ValueError("charge must be an integer and spin a positive multiplicity")
    input_settings = read_lammps_input_settings(input_file) if input_file else None
    if input_settings is not None:
        input_units = input_settings.get("units")
        if units is not None and input_units is not None and units != input_units:
            raise ValueError("units conflicts with the LAMMPS input file")
        units = units or input_units
    if units is None:
        raise ValueError("Supply units or an input_file containing a units command")
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    calculator = None
    reports = []
    missing = set(requested)
    frames = iter_lammps_frames(dump, units=units, input_datafile=datafile, wrap=False)
    try:
        for frame in frames:
            if frame.index not in missing:
                continue
            if "lammps_forces" not in frame.atoms.arrays:
                raise ValueError(f"Frame {frame.index} has no source forces")
            if calculator is None:
                load_start = perf_counter()
                calculator = local_uma_calculator(checkpoint, task=task, device=device,
                                                  inference_settings=inference_settings)
                load_seconds = perf_counter() - load_start
            evaluation_start = perf_counter()
            report = compare_frame_forces(frame, calculator, charge=charge, spin=spin, atol=atol)
            report["evaluation_seconds"] = perf_counter() - evaluation_start
            reports.append(report)
            missing.remove(frame.index)
            if not missing:
                break
    finally:
        frames.close()
    if missing:
        raise IndexError(f"Requested frames not found: {sorted(missing)}")
    return {
        "model_label": "uma-s-1p1", "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest.hexdigest(), "task": task, "device": device,
        "inference_settings": calculator.martini_inference_settings,
        "input_settings": input_settings, "requested_indices": requested,
        "model_load_seconds": load_seconds,
        "total_seconds": perf_counter() - started,
        "passed": all(r["passed"] for r in reports), "frames": reports,
        "versions": {name: version(name) for name in ("ase", "numpy", "fairchem-core", "torch")},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--units", choices=("metal", "real"))
    parser.add_argument("--input", help="LAMMPS input file (literal units and provenance)")
    parser.add_argument("--task", default="omol")
    parser.add_argument("--charge", required=True, type=int)
    parser.add_argument("--spin", required=True, type=int)
    parser.add_argument("--atol", required=True, type=float, help="absolute force tolerance in eV/Angstrom")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int, help="one zero-based frame number (default: 0)")
    selection.add_argument("--indices", nargs="+", type=int, help="selected zero-based frame numbers")
    parser.add_argument("--datafile", help="optional LAMMPS data file for species mapping")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--inference-settings", help="JSON overrides file or fairchem preset; default: confirmed MD settings")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    settings = args.inference_settings
    if settings and Path(settings).is_file():
        settings = json.loads(Path(settings).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            parser.error("inference settings JSON must contain an object")
    indices = args.indices if args.indices is not None else [0 if args.index is None else args.index]
    report = validate_forces(args.dump, args.checkpoint, input_file=args.input,
                             datafile=args.datafile, units=args.units, indices=indices,
                             charge=args.charge, spin=args.spin, atol=args.atol,
                             task=args.task, device=args.device, inference_settings=settings)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
