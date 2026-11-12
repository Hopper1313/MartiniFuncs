"""Energy to remove one surface molecule, for faces of a finished crystal workflow run.

Reuses the run's relaxed bulk (bulk_relaxed.extxyz), gas-phase energies and
lattice energy (results.json). For each face/termination it builds a slab at
least --min-thickness thick (>= 3 layers) with in-plane vectors at least
--min-lengths long, removes the outermost molecule on --side, and writes the
vertical and adiabatic removal energies to --out (JSON) and the relaxed
vacancy slab to removal_<face>_t<k>.extxyz. See
MartiniFuncs.crystals.surfaces.molecule_removal_energy.

    python examples/removal_energies.py antcen_uma --faces 110 --terminations 0 \
        --model uma_data/uma-s-1p2p1.pt --out removal_110_t0.json
"""
import argparse
import json
from pathlib import Path
import time

from ase.io import read, write

from MartiniFuncs.crystals import build_slab, crystal_calculator, crystal_from_atoms, terminations
from MartiniFuncs.crystals.surfaces import molecule_removal_energy
from MartiniFuncs.crystals.workflow import _parse_face


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--faces", nargs="+", required=True)
    parser.add_argument("--terminations", type=int, nargs="*", default=[0])
    parser.add_argument("--side", choices=("top", "bottom"), default="top")
    parser.add_argument("--min-thickness", type=float, default=20.0)
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument("--min-lengths", type=float, default=15.0)
    parser.add_argument("--vacuum", type=float, default=15.0)
    parser.add_argument("--n-fixed-layers", type=int, default=1)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument("--out", default="removal_energies.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default="omc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ncpus", type=int, default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    previous = json.loads((run / "results.json").read_text())
    crystal = crystal_from_atoms(read(run / "bulk_relaxed.extxyz"))
    latt = previous["lattice_energy"]["lattice_energy_eV"]
    calc = crystal_calculator("uma", model=args.model, task=args.task, device=args.device, ncpus=args.ncpus)
    out_path = run / args.out
    done = json.loads(out_path.read_text()) if out_path.exists() else {"rows": []}
    have = {(tuple(r["hkl"]), r["termination"], r["side"]) for r in done["rows"]}

    for face in args.faces:
        hkl = _parse_face(face)
        terms = terminations(crystal, hkl)
        for k in args.terminations:
            if k >= len(terms) or (hkl, k, args.side) in have:
                continue
            probe = build_slab(crystal, hkl, min_thickness=args.min_thickness, termination=terms[k])
            slab = build_slab(crystal, hkl, layers=max(args.min_layers, probe.layers), termination=terms[k],
                              vacuum=args.vacuum, min_lengths=args.min_lengths)
            t0 = time.time()
            r = molecule_removal_energy(slab, calc, previous["gas"], side=args.side, relax=not args.no_relax,
                                        n_fixed_layers=args.n_fixed_layers, fmax=args.fmax, steps=args.steps,
                                        lattice_energy=latt)
            name = "removal_{}_t{}_{}.extxyz".format("".join(map(str, hkl)).replace("-", "m"), k, args.side)
            write(run / name, r.pop("vacancy_atoms"))
            write(run / name.replace("removal_", "intact_"), r.pop("slab_atoms"))
            r.update(hkl=list(hkl), file=name, seconds=time.time() - t0, lattice_energy_eV=latt)
            done["rows"].append(r)
            out_path.write_text(json.dumps(done, indent=2, default=str))
            print(f"{hkl} T{k} {args.side}: {r['n_atoms_slab']} atoms, vertical {r['vertical_kJ_mol']:.1f} kJ/mol "
                  f"({r['vertical_over_lattice']:.2f} |E_latt|)"
                  + (f", vertical on relaxed surface {r['vertical_relaxed_kJ_mol']:.1f} kJ/mol"
                     f", adiabatic {r['adiabatic_kJ_mol']:.1f} kJ/mol ({r['adiabatic_over_lattice']:.2f} |E_latt|), "
                     f"converged={r['converged']}" if "adiabatic_kJ_mol" in r else "")
                  + f", {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
