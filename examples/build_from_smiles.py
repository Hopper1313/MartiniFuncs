"""Pack any molecules given as SMILES into a starting box and write UMA-driver run templates.

Run from the repository root, e.g. anthracene + maleic anhydride, 40 each:

    python -m examples.build_from_smiles --output reports/anthracene_ma_start \\
        --species anthracene "c1ccc2cc3ccccc3cc2c1" 40 \\
        --species maleic_anhydride "O=C1OC(=O)C=C1" 40 --box-length 50

Each species is NAME SMILES COUNT. Templates are RDKit ETKDG + MMFF94 geometries
(structure.molecules.atoms_from_smiles), packed rigidly at random
(structure.packing.pack_molecules): a dilute insertion, not a target density
or an equilibrated structure. The data file's atom types follow --type-order
(default: elements in alphabetical order); the dump_modify element line of
every input and the element_list of the restart templates must use the same
order. Writes initial.data (+ .json identity sidecar), initial.extxyz and
npt.in / nvt.in / nve.in.
"""
import argparse
from pathlib import Path

from ase.io import write

from MartiniFuncs.io.lammps_inputs import render_lammps_input
from MartiniFuncs.structure.molecules import atoms_from_smiles
from MartiniFuncs.structure.packing import MoleculeSpec, pack_molecules, write_initial_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new folder for the box and inputs")
    parser.add_argument("--species", nargs=3, action="append", required=True, metavar=("NAME", "SMILES", "COUNT"))
    parser.add_argument("--box-length", type=float, default=50., help="cubic box edge, Angstrom")
    parser.add_argument("--min-distance", type=float, default=2.0, help="minimum intermolecular atom distance, Angstrom")
    parser.add_argument("--seed", type=int, default=24673, help="packing and embedding seed")
    parser.add_argument("--type-order", nargs="+", help="element of each LAMMPS atom type, e.g. C H O")
    parser.add_argument("--temperature", type=float, default=400., help="K")
    parser.add_argument("--pressure-bar", type=float, default=25000., help="NPT target, bar")
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args(argv)

    species = [MoleculeSpec(name, atoms_from_smiles(smiles, args.seed), int(count))
               for name, smiles, count in args.species]
    packed = pack_molecules(species, [args.box_length] * 3, seed=args.seed, min_distance=args.min_distance)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = write_initial_data(packed, args.output / "initial.data", type_symbols=args.type_order)
    write(args.output / "initial.extxyz", packed)
    for ensemble in ("npt", "nvt", "nve"):
        content = render_lammps_input(data_file="initial.data", type_symbols=metadata["type_symbols"],
                                      ensemble=ensemble, steps=args.steps, force_source="external_uma",
                                      output_prefix=ensemble, temperature_K=args.temperature,
                                      pressure_bar=args.pressure_bar if ensemble == "npt" else None)
        (args.output / f"{ensemble}.in").write_text(content)
    for spec in species:
        print(f"{spec.name}: {spec.atoms.get_chemical_formula()} ({len(spec.atoms)} atoms) x {spec.count}")
    print(f"Atom types: {' '.join(f'{i}={s}' for i, s in enumerate(metadata['type_symbols'], 1))}")
    print(f"Packed {len(packed)} atoms in a {args.box_length:g} Angstrom box; "
          f"density {packed.get_masses().sum() * 1.66053906660 / packed.get_volume():.5f} g/cm^3")
    print(f"Saved {args.output}. Geometry only: no UMA simulation was run.")


if __name__ == "__main__":
    main()
