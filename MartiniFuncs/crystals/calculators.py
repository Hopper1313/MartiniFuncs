"""ASE calculators suited to periodic organic crystals.

    crystal_calculator("uma")          fairchem UMA, task "omc" (molecular crystals, PBE-D3 data)
    crystal_calculator("mace-off")     MACE-OFF23 (organic molecules and crystals; H C N O F P S Cl Br I)
    crystal_calculator("mace-mp")      MACE-MP-0 foundation model, with D3 dispersion by default
    crystal_calculator("mace", model=path)  any MACE model file
    crystal_calculator("emt")          ASE EMT: no physics for organics, for tests only

UMA task heads are different levels of theory: "omc" is PBE+D3 molecular
crystals, "omol" is wB97M-V molecules (no periodic training data) and "omat"
is PBE(+U) inorganic materials without dispersion. Use "omc" for crystals and
slabs, and use the SAME calculator for the gas-phase molecule references of a
lattice energy: mixing heads mixes levels of theory.

model: for UMA a checkpoint path or a pretrained name (default: the
UMA_CHECKPOINT environment variable, else "uma-s-1p2", downloaded from
Hugging Face, which needs access to the facebook/UMA repository); for MACE a
model path or a size ("small", "medium", "large").

Every function in this package that takes a calculator also accepts any ASE
calculator object, so you can build one yourself.
"""
from __future__ import annotations

import os
from pathlib import Path

CALCULATORS = ("uma", "mace-off", "mace-mp", "mace", "emt")


def crystal_calculator(name, *, model=None, task="omc", device="cpu", dispersion=None, dtype="float64",
                       ncpus=None, inference_settings=None):
    """A ready ASE calculator; see the module docstring for names and models."""
    if ncpus:
        from ..optimize.engine import set_threads
        set_threads(ncpus)
    name = name.lower()
    if name == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        model = model or os.environ.get("UMA_CHECKPOINT") or "uma-s-1p2"
        if inference_settings is None:
            inference_settings = uma_settings()
        if Path(str(model)).is_file():
            unit = pretrained_mlip.load_predict_unit(str(model), device=device,
                                                     inference_settings=inference_settings)
        else:
            unit = pretrained_mlip.get_predict_unit(str(model), device=device,
                                                    inference_settings=inference_settings)
        return FAIRChemCalculator(unit, task_name=task)
    if name == "mace-off":
        from mace.calculators import mace_off
        return mace_off(model=model or "medium", device=device, default_dtype=dtype)
    if name == "mace-mp":
        from mace.calculators import mace_mp
        return mace_mp(model=model or "medium", device=device, default_dtype=dtype,
                       dispersion=True if dispersion is None else dispersion)
    if name == "mace":
        if model is None:
            raise ValueError("crystal_calculator('mace') needs model=<path to a .model file>")
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=str(model), device=device, default_dtype=dtype)
    if name == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    raise ValueError(f"unknown calculator {name!r}; choose from {CALCULATORS} or pass an ASE calculator")


def uma_settings(compile=False, merge_mole=False):
    """fairchem InferenceSettings for crystal workflows: no torch.compile and no MoLE merging by default.

    fairchem's "default" preset compiles (needs a C++ compiler, absent on most
    Windows machines) and merges the mixture of experts for ONE composition;
    a crystal workflow evaluates bulk, gas molecules and slabs of several sizes,
    so the general path is used. Pass inference_settings="turbo" (or this with
    compile=True) for long MD runs on a GPU node.
    """
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    return InferenceSettings(tf32=False, activation_checkpointing=False, merge_mole=merge_mole, compile=compile,
                             external_graph_gen=False, internal_graph_gen_version=2)


def resolve(calculator, **options):
    """A calculator object from a name or an existing calculator."""
    return crystal_calculator(calculator, **options) if isinstance(calculator, str) else calculator


def describe(calculator):
    """Short label of a calculator for result files."""
    label = type(calculator).__name__
    task = getattr(calculator, "task_name", None)
    return f"{label}(task={task})" if task else label
