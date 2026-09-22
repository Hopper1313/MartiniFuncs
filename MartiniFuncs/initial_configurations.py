"""Rigid-molecule random packing and atomic-style LAMMPS export.

Packing is geometry-only, not energy minimization or equilibration. ASE molecule
templates must be whole, finite structures in Angstrom. This first implementation
supports fully periodic orthorhombic cells with an origin at zero.
"""
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.io.lammpsdata import write_lammps_data


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    atoms: Atoms
    count: int


class PackingError(RuntimeError):
    """Requested composition could not be packed within the attempt budget."""


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _orthorhombic_lengths(cell):
    matrix = np.asarray(Cell.new(cell))
    lengths = matrix.diagonal()
    if (not np.isfinite(matrix).all() or np.any(lengths <= 0)
            or not np.allclose(matrix, np.diag(lengths), rtol=0, atol=1e-12)):
        raise ValueError("Require positive orthorhombic cell lengths in Angstrom")
    return lengths.copy()


def _rotation(rng):
    # Uniform SO(3) rotation from a normalized Gaussian quaternion.
    w, x, y, z = rng.normal(size=4)
    norm = np.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = np.array([w, x, y, z])/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pack_molecules(species, cell, *, seed, min_distance=2.0, max_attempts_per_molecule=5000):
    """Place molecule copies by random rotation/translation with periodic rejection.

    ``species`` is a sequence of MoleculeSpec; counts may be zero. min_distance
    is a uniform minimum *intermolecular atom-atom* distance in Angstrom, not a
    COM distance or an intramolecular bond cutoff. Input geometry is preserved.
    Species insertion order is randomized, with bounded trials for each copy.

    A conservative bounding-sphere check prevents self-image overlaps: every
    molecule's diameter bound plus min_distance must fit the shortest box edge.
    This can reject small boxes that an orientation-aware packer could accept.
    On failure, raise PackingError without returning an incomplete composition.

    Returned positions are wrapped; image_flags preserve each whole molecule.
    Atoms carry 1-based molecule_id/lammps_id, 0-based global_index and
    template_atom_index, and molecule_name. Velocities, constraints, calculators,
    and charges are not copied: this is neutral geometry-only MLIP input.
    """
    lengths = _orthorhombic_lengths(cell)
    min_distance = _positive(min_distance, "min_distance")
    budget = _integer(max_attempts_per_molecule, "max_attempts_per_molecule")
    seed = _integer(seed, "seed", minimum=0)
    species = list(species)
    if not species or len({s.name for s in species}) != len(species):
        raise ValueError("Provide species with distinct names")
    centered = []
    for spec in species:
        _integer(spec.count, "count", minimum=0)
        if not isinstance(spec.name, str) or not spec.name.strip():
            raise ValueError("Species names must be nonempty strings")
        template = spec.atoms
        if not len(template) or template.pbc.any() or not np.isfinite(template.positions).all():
            raise ValueError("Templates must be nonempty, whole, finite, nonperiodic ASE molecules")
        if np.any(template.get_initial_charges() != 0):
            raise ValueError("Charged templates need an explicit charge-aware export workflow")
        if not np.isfinite(template.get_masses()).all() or np.any(template.get_masses() <= 0):
            raise ValueError("Template masses must be finite and positive")
        xyz = template.positions-template.get_center_of_mass()
        if spec.count and 2*np.linalg.norm(xyz, axis=1).max()+min_distance > lengths.min():
            raise ValueError(f"Cell too small for conservative self-image check of {spec.name}")
        centered.append(xyz)
    order = np.repeat(np.arange(len(species)), [s.count for s in species])
    if not len(order):
        raise ValueError("Request at least one molecule")
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    positions, numbers, masses, molecule_ids, names, template_indices = [], [], [], [], [], []
    occupied = np.empty((0, 3))
    attempts_total = 0
    for molecule_id, index in enumerate(order, start=1):
        spec = species[index]
        for _ in range(budget):
            attempts_total += 1
            xyz = centered[index] @ _rotation(rng).T + rng.uniform(size=3)*lengths
            delta = xyz[:, None, :]-occupied[None, :, :]
            delta -= lengths*np.floor(delta/lengths+.5)
            if np.any(np.sum(delta**2, axis=-1) < min_distance**2):
                continue
            positions.extend(xyz)
            numbers.extend(spec.atoms.numbers)
            masses.extend(spec.atoms.get_masses())
            molecule_ids.extend([molecule_id]*len(xyz))
            names.extend([spec.name]*len(xyz))
            template_indices.extend(range(len(xyz)))
            occupied = np.vstack((occupied, xyz % lengths))
            break
        else:
            raise PackingError(f"Could not place molecule {molecule_id}/{len(order)} ({spec.name}) "
                               f"after {budget} trials; enlarge the cell or use a dense-packing method")
    whole = np.asarray(positions)
    flags = np.floor(whole/lengths).astype(int)
    atoms = Atoms(numbers=numbers, positions=whole-flags*lengths, masses=masses, cell=lengths, pbc=True)
    for name, values in {"global_index": np.arange(len(atoms)), "lammps_id": np.arange(1,len(atoms)+1),
                         "molecule_id": molecule_ids, "molecule_name": names,
                         "template_atom_index": template_indices, "image_flags": flags}.items():
        atoms.new_array(name, np.asarray(values))
    atoms.info.update(packing_seed=seed, min_distance_angstrom=min_distance,
                      placement_attempts=attempts_total,
                      composition={s.name: int(s.count) for s in species}, units="metal")
    return atoms


def write_initial_data(atoms, path, *, type_symbols=None, overwrite=False):
    """Write atomic-style metal-unit data plus ``<path>.json`` identity metadata.

    One type per element; isotope/type-specific masses are not supported. The
    data file has Masses and image flags, no force-field bonds or velocities.
    Atomic style cannot store molecule IDs, so the sidecar maps written atom IDs
    to molecular identity. No forces/model weights are serialized.
    """
    path = Path(path)
    sidecar = Path(str(path)+".json")
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError(f"Refusing to overwrite {path} or its metadata")
    lengths = _orthorhombic_lengths(atoms.cell)
    if not atoms.pbc.all() or not np.allclose(atoms.get_celldisp(), 0):
        raise ValueError("Require fully periodic cell with origin at zero")
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("Require nonempty finite positions")
    if "momenta" in atoms.arrays or np.any(atoms.get_initial_charges() != 0):
        raise ValueError("Initial export is geometry-only; velocities/charges require a separate workflow")
    symbols = atoms.get_chemical_symbols()
    types = sorted(set(symbols)) if type_symbols is None else list(type_symbols)
    if len(set(types)) != len(types) or set(types) != set(symbols):
        raise ValueError("type_symbols must list every element exactly once")
    for symbol in types:
        values = atoms.get_masses()[np.array(symbols) == symbol]
        if not np.isfinite(values).all() or np.any(values <= 0) or not np.allclose(values, values[0], rtol=0, atol=1e-10):
            raise ValueError("Require a single positive mass per element/type")
    for key in ("molecule_id", "molecule_name", "template_atom_index", "image_flags"):
        if key not in atoms.arrays:
            raise ValueError(f"Missing packing metadata {key}")
    whole = atoms.copy()
    whole.positions += atoms.arrays["image_flags"]*lengths
    # ASE generates new consecutive file IDs and image flags from whole positions.
    stream = StringIO()
    write_lammps_data(stream, whole, specorder=types, atom_style="atomic", units="metal",
                      masses=True, velocities=False, write_image_flags=True)
    metadata = {"schema_version": 1, "units": "metal", "atom_style": "atomic",
                "type_symbols": types, "cell_angstrom": atoms.cell.array.tolist(),
                "packing": atoms.info,
                "atoms": [{"lammps_id": i+1, "global_index": i,
                           "molecule_id": int(atoms.arrays["molecule_id"][i]),
                           "molecule_name": str(atoms.arrays["molecule_name"][i]),
                           "template_atom_index": int(atoms.arrays["template_atom_index"][i])}
                          for i in range(len(atoms))]}
    encoded = json.dumps(metadata, indent=2)+"\n"
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(stream.getvalue())
    with sidecar.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(encoded)
    return metadata
