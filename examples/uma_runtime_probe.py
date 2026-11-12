"""Check that a Python runtime can run the crystal workflows with UMA, and time one force call.

    python examples/uma_runtime_probe.py --model uma-s-1p2p1.pt --structure antcen_uma/bulk_relaxed.extxyz
"""
import argparse
import importlib
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--structure", required=True)
parser.add_argument("--task", default="omc")
args = parser.parse_args()

report = {"threads": os.environ.get("OMP_NUM_THREADS")}
for name in ("numpy", "scipy", "networkx", "ase", "torch", "fairchem.core", "pymatgen.core", "spglib"):
    try:
        module = importlib.import_module(name)
        report[name] = getattr(module, "__version__", "ok")
    except Exception as error:                     # report every missing piece, not just the first
        report[name] = f"MISSING: {error!r}"
print("IMPORTS", json.dumps(report), flush=True)

from ase.filters import FrechetCellFilter  # noqa: F401  (needs ase >= 3.23)
from ase.io import read
from MartiniFuncs.crystals import build_slab, crystal_calculator, crystal_from_atoms

crystal = crystal_from_atoms(read(args.structure))
slab = build_slab(crystal, (0, 0, 1), layers=3)
t0 = time.time()
calc = crystal_calculator("uma", model=args.model, task=args.task, ncpus=int(os.environ.get("OMP_NUM_THREADS", 1)))
load = time.time() - t0
atoms = slab.atoms.copy()
atoms.calc = calc
atoms.get_forces()
t0 = time.time()
atoms.positions[0] += 0.01
atoms.get_forces()
print("PROBE_OK", json.dumps({"load_s": load, "force_call_s": time.time() - t0, "n_atoms": len(atoms),
                              "energy_eV": float(atoms.get_potential_energy())}), flush=True)
