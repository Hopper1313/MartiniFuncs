"""Attachment energies per face and termination, and the attachment-energy growth shape.

Uses a finished crystal workflow run (bulk_relaxed.extxyz, and results.json for the bulk energy per
molecule and the lattice energy). Each slice is one d_hkl layer cut at the termination, evaluated
without relaxation. Writes attachment_energies.json in the run directory: one row per face and
termination, the growth shape (area fractions, the termination chosen for each face), and the
slices themselves under attachment_slices/ when --write-slices is given.

    python examples/attachment_energies.py mleica_uma --faces 110 010 011 120 101 111 100 001 \
        --model uma_data/uma-s-1p2p1.pt --ncpus 24
"""
import argparse
import json
from pathlib import Path

from ase.io import read, write

from MartiniFuncs.crystals import build_slab, crystal_calculator, crystal_from_atoms
from MartiniFuncs.crystals.morphology import attachment_energy_shape
from MartiniFuncs.crystals.surfaces import attachment_energies
from MartiniFuncs.crystals.workflow import _parse_face


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--faces", nargs="+", required=True)
    parser.add_argument("--max-terminations", type=int, default=None, help="default: every termination")
    parser.add_argument("--vacuum", type=float, default=15.0)
    parser.add_argument("--out", default="attachment_energies.json")
    parser.add_argument("--write-slices", action="store_true")
    parser.add_argument("--calculator", default="uma")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default="omc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ncpus", type=int, default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    results = json.loads((run / "results.json").read_text())
    e_bulk = results["bulk"]["energy_per_molecule"]
    latt = results["lattice_energy"]["lattice_energy_eV"]
    crystal = crystal_from_atoms(read(run / "bulk_relaxed.extxyz"))
    calc = crystal_calculator(args.calculator, model=args.model, task=args.task, device=args.device, ncpus=args.ncpus)
    faces = [_parse_face(f) for f in args.faces]
    rows = attachment_energies(crystal, calc, e_bulk, faces, lattice_energy=latt, vacuum=args.vacuum,
                               max_terminations=args.max_terminations)
    for r in rows:
        print(f"{r['hkl']} T{r['termination']}: E_att {r['attachment_energy_kJ_mol']:7.1f} kJ/mol "
              f"({100 * r['attachment_fraction']:4.1f} % of E_latt), slice {r['slice_energy_kJ_mol']:7.1f} kJ/mol, "
              f"{r['n_molecules']} molecules, gamma_slice {1000 * r['gamma_slice_J_m2']:6.1f} mJ/m^2", flush=True)
        if args.write_slices:
            sdir = run / "attachment_slices"
            sdir.mkdir(exist_ok=True)
            piece = build_slab(crystal, r["hkl"], layers=1, termination=r["termination"], vacuum=args.vacuum)
            write(sdir / "slice_{}_t{}.extxyz".format("".join(map(str, r["hkl"])).replace("-", "m"), r["termination"]),
                  piece.atoms)
    out = {"bulk_energy_per_molecule": e_bulk, "lattice_energy_eV": latt, "settings": vars(args),
           "rows": [dict(r, hkl=list(r["hkl"])) for r in rows]}
    try:
        shape = attachment_energy_shape(crystal, rows)
        out["growth_shape"] = {"area_fractions": {",".join(map(str, h)): a for h, a in shape["area_fractions"].items()},
                               "chosen_termination": {",".join(map(str, h)): r["termination"]
                                                      for h, r in shape["chosen"].items()}}
        print("growth shape:", {k: round(v, 3) for k, v in out["growth_shape"]["area_fractions"].items() if v > 1e-3})
    except ValueError as error:
        out["growth_shape"] = {"error": str(error)}
    (run / args.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
