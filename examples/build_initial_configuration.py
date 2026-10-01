"""Generate a dilute 40+40 starting box and UMA-driver run templates.

Run: python -m examples.build_initial_configuration
The 50 Angstrom box is an insertion example, not a target-density prediction.
"""
import argparse
from pathlib import Path

from ase import Atoms
from ase.io import write

from MartiniFuncs.structure import graph
from MartiniFuncs.structure.packing import MoleculeSpec, pack_molecules, write_initial_data
from MartiniFuncs.io.lammps_inputs import render_lammps_input
from MartiniFuncs.io.lammps import read_lammps_frame


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "reports/initial_configuration")
    parser.add_argument("--seed", type=int, default=24673)
    parser.add_argument("--box-length", type=float, default=50.)
    args = parser.parse_args()
    # Extract one whole template of each species from the existing small fixture.
    atoms = read_lammps_frame(root / "tests/lammpsio_tests/frame_0.lammpstrj", units="metal").atoms
    bonds = graph.build_bond_graph(atoms, scale=1.3)
    rows = {int(i): row for row, i in enumerate(atoms.arrays["global_index"])}
    templates = {}
    for component in graph.connected_components(bonds, atom_ids=atoms.arrays["global_index"]):
        ids, xyz, _ = graph.place_component(atoms, bonds, component, strict=True)
        selection = atoms[[rows[int(i)] for i in ids]]
        formula = selection.get_chemical_formula()
        if formula in {"C5H6", "C11H9NO2"} and formula not in templates:
            templates[formula] = Atoms(selection.get_chemical_symbols(), positions=xyz, masses=selection.get_masses())
    packed = pack_molecules([MoleculeSpec("CPD", templates["C5H6"], 40),
                             MoleculeSpec("dienophile", templates["C11H9NO2"], 40)],
                            [args.box_length]*3, seed=args.seed, min_distance=2.0)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = write_initial_data(packed, args.output / "initial.data", type_symbols=["C", "H", "N", "O"])
    write(args.output / "initial.extxyz", packed)
    for ensemble in ("npt", "nvt", "nve"):
        content = render_lammps_input(data_file="initial.data", type_symbols=metadata["type_symbols"],
                    ensemble=ensemble, steps=100000, force_source="external_uma",
                    output_prefix=ensemble, temperature_K=400.,
                    pressure_bar=25000. if ensemble == "npt" else None)
        (args.output / f"{ensemble}.in").write_text(content)
    print(f"Packed {len(packed)} atoms / 80 molecules in {args.box_length:g} Angstrom box")
    print(f"Density: {packed.get_masses().sum()*1.66053906660/packed.get_volume():.5f} g/cm^3")
    print(f"Placement attempts: {packed.info['placement_attempts']}; saved {args.output}")
    print("Templates are independent fresh starts, not a sequential protocol. No UMA simulation was run.")


if __name__ == "__main__":
    main()
