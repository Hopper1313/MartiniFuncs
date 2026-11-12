"""Relaxed surface energies of a molecular crystal from a finished crystal workflow run.

Reuses the relaxed bulk (bulk_relaxed.extxyz) and its energy per molecule
(results.json) written by `python -m MartiniFuncs.crystals.workflow`, so the
slabs share that run's bulk reference. For each face and each of the first
--n-terminations terminations it builds a slab at least --min-thickness thick
(and at least 3 layers, so a central layer can be fixed while both faces relax),
relaxes it, and writes relaxed_surfaces.json after every slab. A rerun skips
slabs already in that file, so an interrupted job can simply be restarted.

    python examples/relax_surface_energies.py antcen_uma --faces 001 011 11-1 110 \
        --model uma_data/uma-s-1p2p1.pt --device cuda
"""
import argparse
import json
from pathlib import Path
import time

from ase.io import read, write

from MartiniFuncs.crystals import build_slab, crystal_calculator, crystal_from_atoms, terminations, wulff
from MartiniFuncs.crystals.surfaces import surface_energy
from MartiniFuncs.crystals.workflow import _parse_face


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="output directory of a crystal workflow run")
    parser.add_argument("--faces", nargs="*", default=None, help="default: the faces of the run")
    parser.add_argument("--n-terminations", type=int, default=2)
    parser.add_argument("--terminations", type=int, nargs="*", default=None,
                        help="only these termination indices (overrides --n-terminations)")
    parser.add_argument("--out", default="relaxed_surfaces.json",
                        help="results file in run_dir (give each parallel job its own)")
    parser.add_argument("--min-thickness", type=float, default=20.0)
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument("--vacuum", type=float, default=15.0)
    parser.add_argument("--n-fixed-layers", type=int, default=1)
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--optimizer", default="BFGS")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default="omc")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ncpus", type=int, default=None)
    args = parser.parse_args(argv)

    run = Path(args.run_dir)
    previous = json.loads((run / "results.json").read_text())
    e_bulk = previous["bulk"]["energy_per_molecule"]
    crystal = crystal_from_atoms(read(run / "bulk_relaxed.extxyz"))
    faces = [_parse_face(f) for f in args.faces] if args.faces else [tuple(f) for f in previous["faces"]]
    calc = crystal_calculator("uma", model=args.model, task=args.task, device=args.device, ncpus=args.ncpus)

    out_path = run / args.out
    done = json.loads(out_path.read_text()) if out_path.exists() else {"rows": []}
    have = {(tuple(r["hkl"]), r["termination"]) for r in done["rows"]}
    done.update(bulk_energy_per_molecule=e_bulk, settings=vars(args))

    for hkl in faces:
        terms = terminations(crystal, hkl)
        chosen = [terms[i] for i in args.terminations if i < len(terms)] if args.terminations is not None             else terms[:args.n_terminations]
        for term in chosen:
            if (tuple(hkl), term.index) in have:
                continue
            probe = build_slab(crystal, hkl, min_thickness=args.min_thickness, termination=term, vacuum=args.vacuum)
            layers = max(args.min_layers, probe.layers)
            slab = build_slab(crystal, hkl, layers=layers, termination=term, vacuum=args.vacuum)
            t0 = time.time()
            r = surface_energy(slab, calc, e_bulk, relax=True, n_fixed_layers=args.n_fixed_layers, fmax=args.fmax,
                               steps=args.steps, optimizer=args.optimizer, crystal=crystal)
            name = "relaxed_{}_t{}.extxyz".format("".join(map(str, hkl)).replace("-", "m"), term.index)
            write(run / name, r["atoms"])
            row = {k: v for k, v in r.items() if k not in ("atoms",)}
            row.update(hkl=list(hkl), layers=layers, thickness=slab.thickness, n_atoms=len(slab.atoms),
                       relaxation_energy_mJ_m2=(r["gamma_J_m2"] - r["gamma_unrelaxed_J_m2"]) * 1000,
                       seconds=time.time() - t0, file=name)
            done["rows"].append(row)
            out_path.write_text(json.dumps(done, indent=2, default=str))
            print(f"{hkl} T{term.index}: {layers} layers, {len(slab.atoms)} atoms, "
                  f"gamma {r['gamma_unrelaxed_J_m2'] * 1000:.1f} -> {r['gamma_J_m2'] * 1000:.1f} mJ/m^2, "
                  f"converged={r['converged']} in {r['steps']} steps, {time.time() - t0:.0f} s", flush=True)

    best = {}
    for r in done["rows"]:
        h = tuple(r["hkl"])
        if h not in best or r["gamma_J_m2"] < best[h]["gamma_J_m2"]:
            best[h] = r
    try:
        shape = wulff(crystal, {h: r["gamma_J_m2"] for h, r in best.items()})
        done["wulff"] = {",".join(map(str, h)): a for h, a in shape["area_fractions"].items()}
    except Exception as error:             # too few faces, or pymatgen not installed
        done["wulff"] = {"error": str(error)}
    out_path.write_text(json.dumps(done, indent=2, default=str))
    print("Wulff area fractions:", done["wulff"], flush=True)


if __name__ == "__main__":
    main()
