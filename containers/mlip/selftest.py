"""Check the container's models against the tested reference energies.

Runs every model available in the current environment (uma env: UMA and
AIMNet2; mace env: MACE-OMOL) on the anthracene + MA reference structures
(MartiniFuncs package data; charge 0, singlet) and compares the energies with
reference_energies.json. Model files come from UMA_CHECKPOINT, MACE_MODEL and
AIMNET_MODEL; a model whose file is missing is reported and skipped.

usage: python selftest.py [--tolerance EV]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from ase import Atoms

from MartiniFuncs.chemistry.anthracene_ma.build import reference_structure

REFERENCE = json.loads((Path(__file__).with_name("reference_energies.json")).read_text())["energies_eV"]


def calculators():
    """(name, factory) for each model whose package is installed in this environment."""
    found = []
    if importlib.util.find_spec("fairchem"):
        def uma():
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            return FAIRChemCalculator(pretrained_mlip.load_predict_unit(os.environ["UMA_CHECKPOINT"], device="cpu"),
                                      task_name="omol")
        found.append(("uma", "UMA_CHECKPOINT", uma))
    if importlib.util.find_spec("aimnet"):
        def aimnet():
            from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
            return AIMNet2ASE(AIMNet2Calculator(os.environ["AIMNET_MODEL"], device="cpu"), charge=0, mult=1)
        found.append(("aimnet", "AIMNET_MODEL", aimnet))
    if importlib.util.find_spec("mace"):
        def mace():
            from mace.calculators import mace_omol
            return mace_omol(model=os.environ["MACE_MODEL"], device="cpu", default_dtype="float64")
        found.append(("mace", "MACE_MODEL", mace))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="allowed |dE| in eV (default 1e-4)")
    args = parser.parse_args()
    failures = 0
    for name, variable, factory in calculators():
        path = os.environ.get(variable, "")
        if not Path(path).is_file():
            print(f"[skip] {name}: {variable}={path!r} is not a file (bind the model into the container)")
            continue
        start = time.perf_counter()
        calc = factory()
        load = time.perf_counter() - start
        worst = 0.0
        for structure, expected in REFERENCE[name].items():
            source = reference_structure(structure)
            atoms = Atoms(source.get_chemical_symbols(), positions=source.positions)
            atoms.info.update(charge=0, spin=1)
            atoms.calc = calc
            worst = max(worst, abs(atoms.get_potential_energy() - expected))
        ok = worst <= args.tolerance
        failures += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}: max |E - reference| = {worst:.2e} eV over "
              f"{len(REFERENCE[name])} structures (load {load:.1f} s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
