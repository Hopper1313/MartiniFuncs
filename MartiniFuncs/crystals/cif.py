"""Read organic crystal structures from CIF files into whole-molecule unit cells.

read_cif applies the symmetry operators listed in the CIF itself
(_space_group_symop_operation_xyz or _symmetry_equiv_pos_as_xyz), so it does
not depend on looking up a space-group symbol or setting; only when a file
lists no operators is the Hermann-Mauguin symbol or number looked up in ASE.
Symmetry copies that land on the same site (special positions, or a CIF that
already lists every atom) are merged.

Disorder: with disorder="major" (default) sites in disorder groups other than
1/-1 are dropped, and so are sites with occupancy below occupancy_cutoff
(0.5) when the file has no disorder groups. disorder="all" keeps every site
and fails if two of them overlap; "error" fails on any partial occupancy.

Molecules are found with the package bond graph (covalent radii x bond_scale,
default 1.3) and made whole: each molecule is unwrapped across the cell
boundary and translated so its centre of mass lies inside the cell. A bonded
network that never closes into a finite molecule (a framework or polymer)
raises ValueError.

X-ray structures place hydrogens about 0.1 Angstrom too close to their parent
atom; normalize_hydrogens=True moves them along the bond to the neutron
lengths in X_H_LENGTHS (C-H 1.089, N-H 1.015, O-H 0.993 Angstrom), which is the
usual step before computing lattice energies. An MLIP relaxation fixes them in
any case.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import io
from pathlib import Path
import re
import warnings

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io.cif import parse_cif
from ase.spacegroup.spacegroup import parse_sitesym

from ..structure.graph import build_bond_graph, connected_components, periodic_pairs, unwrap_component

SYMOP_TAGS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz",
              "_space_group_symop.operation_xyz", "_symmetry_equiv.pos_as_xyz")
X_H_LENGTHS = {"C": 1.089, "N": 1.015, "O": 0.993, "B": 1.19, "S": 1.338}
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


@dataclass(frozen=True)
class Symmetry:
    """Space-group operators in the fractional basis of the crystal's cell: x' = R x + t."""

    rotations: np.ndarray
    translations: np.ndarray
    symbol: str | None = None
    number: int | None = None

    def __len__(self):
        return len(self.rotations)

    @property
    def centrosymmetric(self):
        """True when the group contains the inversion -1 (with any translation)."""
        return any(np.array_equal(R, -np.eye(3, dtype=int)) for R in self.rotations)

    @property
    def point_group_rotations(self):
        """The distinct rotation parts (the crystal class)."""
        unique = {tuple(R.ravel()) for R in self.rotations.astype(int)}
        return np.array([np.reshape(r, (3, 3)) for r in sorted(unique)], dtype=int)

    @classmethod
    def identity(cls):
        return cls(np.eye(3, dtype=int)[None], np.zeros((1, 3)), "P 1", 1)


@dataclass
class MolecularCrystal:
    """A periodic crystal of whole molecules.

    atoms carries the per-atom array "mol_id" (0..n_molecules-1); each molecule's
    atoms are contiguous in space (unwrapped) and its centre of mass is inside the
    cell. symmetry holds the CIF operators in this cell's basis (None if unknown);
    source records where the structure came from (path, block, experimental
    cell, temperature, sites dropped as disorder).
    """

    atoms: Atoms
    symmetry: Symmetry | None = None
    name: str = ""
    source: dict = field(default_factory=dict)

    @property
    def mol_id(self):
        return self.atoms.arrays["mol_id"]

    @property
    def n_molecules(self):
        return int(self.mol_id.max()) + 1 if len(self.atoms) else 0

    @property
    def molecules(self):
        """Atom indices of each molecule, in mol_id order."""
        return [np.flatnonzero(self.mol_id == m) for m in range(self.n_molecules)]

    def molecule(self, m):
        """Molecule m as a finite (pbc=False) Atoms object, coordinates unchanged."""
        sub = self.atoms[self.molecules[m]]
        sub.set_cell(None)
        sub.pbc = False
        del sub.arrays["mol_id"]
        return sub

    @property
    def species(self):
        """Hill formula of each molecule (molecules with the same formula count as one species)."""
        return [self.atoms[idx].get_chemical_formula(mode="hill") for idx in self.molecules]

    @property
    def composition(self):
        """{formula: number of molecules in the cell}."""
        counts = {}
        for s in self.species:
            counts[s] = counts.get(s, 0) + 1
        return counts

    def representative_molecules(self):
        """{formula: Atoms} with the first molecule of each species (for gas-phase references)."""
        found = {}
        for m, s in enumerate(self.species):
            found.setdefault(s, self.molecule(m))
        return found

    @property
    def centers_of_mass(self):
        masses = self.atoms.get_masses()
        return np.array([masses[idx] @ self.atoms.positions[idx] / masses[idx].sum() for idx in self.molecules])

    @property
    def density(self):
        """Density in g/cm^3."""
        return float(self.atoms.get_masses().sum() / self.atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3)

    @property
    def volume_per_molecule(self):
        return float(self.atoms.get_volume() / self.n_molecules)

    @property
    def z_prime(self):
        """Molecules per asymmetric unit (Z / number of operators); None without symmetry."""
        return None if self.symmetry is None else self.n_molecules / len(self.symmetry)

    def with_atoms(self, atoms):
        """The same crystal (symmetry, name, source) with new atoms, e.g. after a relaxation.

        Molecules are re-made whole, so the new structure must have the same bonding.
        """
        if len(atoms) != len(self.atoms) or atoms.get_chemical_symbols() != self.atoms.get_chemical_symbols():
            raise ValueError("the new atoms do not have the same atoms (or order) as this crystal")
        try:
            whole = make_molecules_whole(atoms, bond_scale=self.source.get("bond_scale", 1.3))
        except ValueError as error:
            raise ValueError("the bonding changed (molecules fused into a network): the relaxation or "
                             f"calculator is not reasonable for this crystal ({error})") from error
        if not np.array_equal(whole.arrays["mol_id"], self.mol_id):
            raise ValueError("the bonding changed: the new atoms do not form the same molecules as this crystal")
        return replace(self, atoms=whole, source=dict(self.source))

    def summary(self):
        cellpar = self.atoms.cell.cellpar()
        return {
            "name": self.name, "formula": self.atoms.get_chemical_formula(mode="hill"),
            "n_atoms": len(self.atoms), "Z": self.n_molecules, "Z_prime": self.z_prime,
            "composition": self.composition,
            "space_group": None if self.symmetry is None else self.symmetry.symbol,
            "space_group_number": None if self.symmetry is None else self.symmetry.number,
            "n_operators": None if self.symmetry is None else len(self.symmetry),
            "centrosymmetric": None if self.symmetry is None else self.symmetry.centrosymmetric,
            "cell_lengths": cellpar[:3].tolist(), "cell_angles": cellpar[3:].tolist(),
            "volume": float(self.atoms.get_volume()), "volume_per_molecule": self.volume_per_molecule,
            "density_g_cm3": self.density,
        }


# ----------------------------------------------------------------------------- CIF parsing


def _blocks(path):
    """CIF blocks of path; anything before the first data_ line (some exporters write a banner) is skipped."""
    text = Path(path).read_text(errors="replace")
    match = re.search(r"^\s*data_", text, flags=re.MULTILINE | re.IGNORECASE)
    if match is None:
        raise ValueError(f"{path}: no data_ block found")
    body = text[match.start():].lstrip()
    lines = [("data_" + line.lstrip()[5:]) if line.lstrip().lower().startswith("data_") else line
             for line in body.splitlines()]
    return list(parse_cif(io.StringIO("\n".join(lines) + "\n")))


def _get(block, *tags, default=None):
    for tag in tags:
        value = block.get(tag.lower())
        if value is not None:
            return value
    return default


def _number(value, default=np.nan):
    """CIF numeric value -> float; strings like '0.123(4)', '?' and '.' are handled."""
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if text in ("?", "."):
        return default
    text = re.sub(r"\(\d+\)$", "", text)
    try:
        return float(text)
    except ValueError:
        return default


def _element(type_symbol, label):
    for raw in (type_symbol, label):
        if raw is None or str(raw).strip() in ("?", "."):
            continue
        match = re.match(r"([A-Za-z]{1,2})", str(raw).strip())
        if not match:
            continue
        letters = match.group(1)
        for candidate in (letters[:1].upper() + letters[1:].lower(), letters[:1].upper()):
            if candidate in atomic_numbers:
                return "H" if candidate == "D" else candidate
    raise ValueError(f"cannot read an element from atom site {label!r} (type {type_symbol!r})")


def _symmetry(block):
    ops = _get(block, *SYMOP_TAGS)
    symbol = _get(block, "_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m", "_space_group.name_h-m_alt")
    number = _get(block, "_space_group_it_number", "_symmetry_int_tables_number", "_space_group.it_number")
    symbol = None if symbol in (None, "?", ".") else str(symbol).strip()
    number = None if _number(number) != _number(number) else int(_number(number))
    if ops is not None:
        if isinstance(ops, str):
            ops = [ops]
        rotations, translations = parse_sitesym([str(op).replace(" ", "") for op in ops])
        return Symmetry(rotations.astype(int), np.mod(translations, 1.0), symbol, number)
    if symbol is None and number is None:
        warnings.warn("CIF lists no symmetry operators or space group; treating the sites as P1")
        return Symmetry.identity()
    from ase.spacegroup import Spacegroup
    group = Spacegroup(number if symbol is None else symbol)
    rotations, translations = zip(*group.get_symop())
    return Symmetry(np.array(rotations, dtype=int), np.mod(np.array(translations), 1.0), group.symbol, group.no)


def _sites(block, disorder, occupancy_cutoff):
    labels = _get(block, "_atom_site_label")
    xs = _get(block, "_atom_site_fract_x")
    if labels is None or xs is None:
        raise ValueError("the CIF block has no fractional atom sites (_atom_site_fract_x)")
    n = len(xs)
    types = _get(block, "_atom_site_type_symbol", default=[None] * n)
    frac = np.array([[_number(v) for v in _get(block, f"_atom_site_fract_{axis}")] for axis in "xyz"]).T
    occupancy = np.array([_number(v, 1.0) for v in _get(block, "_atom_site_occupancy", default=[1.0] * n)])
    groups = [str(g).strip() for g in _get(block, "_atom_site_disorder_group", default=["."] * n)]
    symbols = [_element(t, lab) for t, lab in zip(types, labels)]
    if np.isnan(frac).any():
        raise ValueError("some atom sites have no fractional coordinates")
    keep = np.ones(n, dtype=bool)
    partial = occupancy < 0.999
    has_groups = any(g not in (".", "?", "0") for g in groups)
    if disorder == "error" and (partial.any() or has_groups):
        raise ValueError(f"disordered sites present: {[l for l, p in zip(labels, partial) if p]}")
    if disorder == "major":
        if has_groups:
            keep &= np.array([g in (".", "?", "0", "1", "-1") for g in groups])
        else:
            keep &= occupancy >= occupancy_cutoff
    elif disorder not in ("all", "error"):
        raise ValueError("disorder must be 'major', 'all' or 'error'")
    dropped = [str(lab) for lab, k in zip(labels, keep) if not k]
    return [s for s, k in zip(symbols, keep) if k], frac[keep], [str(lab) for lab, k in zip(labels, keep) if k], dropped


def expand_sites(symbols, frac, cell, symmetry, *, merge_tol=0.3, labels=None):
    """Apply every operator to every site and merge copies closer than merge_tol (Angstrom).

    Returns an Atoms object (pbc=True) with an "asym_site" array (index of the CIF site
    each atom came from). Copies of different elements on one site raise ValueError.
    """
    frac = np.asarray(frac, dtype=float)
    images = np.einsum("oij,sj->soi", symmetry.rotations, frac) + symmetry.translations[None]
    images = np.mod(images, 1.0).reshape(-1, 3)
    site = np.repeat(np.arange(len(frac)), len(symmetry))
    positions = images @ np.asarray(cell, dtype=float)
    i, j, _, _ = periodic_pairs(positions, cell, np.ones(3, bool), merge_tol)
    parent = np.arange(len(positions))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(i.tolist(), j.tolist()):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(a) for a in range(len(positions))])
    keep = np.flatnonzero(roots == np.arange(len(positions)))
    for a, b in zip(i.tolist(), j.tolist()):
        if symbols[site[a]] != symbols[site[b]]:
            names = labels or [str(k) for k in range(len(symbols))]
            raise ValueError(f"sites {names[site[a]]} and {names[site[b]]} ({symbols[site[a]]}, {symbols[site[b]]}) "
                             f"overlap within {merge_tol} Angstrom; the structure is probably disordered "
                             "(try disorder='major')")
    atoms = Atoms([symbols[s] for s in site[keep]], positions=positions[keep], cell=cell, pbc=True)
    atoms.arrays["asym_site"] = site[keep]
    return atoms


# ----------------------------------------------------------------------------- whole molecules


def make_molecules_whole(atoms, *, bond_scale=1.3):
    """Copy of a periodic structure with each molecule unwrapped and its centre of mass in the cell.

    Atoms are reordered molecule by molecule (molecules ordered by their lowest
    original index) and the per-atom "mol_id" array is set. Per-atom arrays
    other than positions/numbers are carried along. Raises ValueError for a
    bonded network that does not close into finite molecules.
    """
    atoms = atoms.copy()
    for name in ("mol_id",):
        atoms.arrays.pop(name, None)
    graph = build_bond_graph(atoms, scale=bond_scale, periodic=True, index_array=None)
    components = connected_components(graph, atom_ids=range(len(atoms)))
    components = sorted((sorted(c) for c in components), key=lambda c: c[0])
    order, mol_id, positions = [], [], []
    masses = atoms.get_masses()
    cell = atoms.cell.array
    for m, component in enumerate(components):
        try:
            _, unwrapped = unwrap_component(atoms, graph, component, index_array=None, strict=True)
        except ValueError as error:
            raise ValueError(f"molecule {m} ({atoms[component].get_chemical_formula()}) is bonded to its own "
                             "periodic image: the structure is a framework or polymer, not a molecular crystal "
                             f"(or bond_scale={bond_scale} is too large)") from error
        com = masses[component] @ unwrapped / masses[component].sum()
        shift = np.floor(np.linalg.solve(cell.T, com) + 1e-9)
        positions.append(unwrapped - shift @ cell)
        order.extend(component)
        mol_id.extend([m] * len(component))
    whole = atoms[order]
    whole.positions = np.vstack(positions) if positions else np.empty((0, 3))
    whole.arrays["mol_id"] = np.asarray(mol_id, dtype=int)
    return whole


def normalize_hydrogen_positions(atoms, *, lengths=None, bond_scale=1.3):
    """Copy with each H moved along its bond to the standard X-H length (X_H_LENGTHS); returns (atoms, n_moved)."""
    lengths = dict(X_H_LENGTHS, **(lengths or {}))
    atoms = atoms.copy()
    graph = build_bond_graph(atoms, scale=bond_scale, periodic=True, index_array=None)
    symbols = atoms.get_chemical_symbols()
    cell = atoms.cell.array
    moved = 0
    for h, symbol in enumerate(symbols):
        if symbol != "H" or len(graph.get(h, [])) != 1:
            continue
        x, shift = graph[h][0]
        if symbols[x] not in lengths:
            continue
        parent = atoms.positions[x] + np.asarray(shift) @ cell
        bond = atoms.positions[h] - parent
        atoms.positions[h] = parent + bond / np.linalg.norm(bond) * lengths[symbols[x]]
        moved += 1
    return atoms, moved


def check_structure(atoms, *, min_distance=0.7):
    """Warnings for common CIF problems: missing hydrogens and atoms closer than min_distance."""
    problems = []
    symbols = set(atoms.get_chemical_symbols())
    if "C" in symbols and "H" not in symbols:
        problems.append("the structure has carbon but no hydrogen atoms (an X-ray CIF without H positions?); "
                        "add hydrogens before using an MLIP")
    i, j, _, d = periodic_pairs(atoms.positions, atoms.cell.array, atoms.pbc, min_distance)
    if len(d):
        problems.append(f"{len(d) // 2} atom pairs are closer than {min_distance} Angstrom "
                        f"(shortest {d.min():.3f}); check disorder handling")
    return problems


def read_cif(path, *, block=None, disorder="major", occupancy_cutoff=0.5, merge_tol=0.3, bond_scale=1.3,
             normalize_hydrogens=False, name=None, strict=True):
    """Read a CIF into a MolecularCrystal with whole molecules.

    block: data block name (without data_) or index; default the first block with
    atom sites. strict=True raises on the problems check_structure finds
    (overlapping atoms, carbon without hydrogen); strict=False only warns.
    """
    blocks = [b for b in _blocks(path) if _get(b, "_atom_site_fract_x") is not None]
    if not blocks:
        raise ValueError(f"{path}: no block with fractional atom sites")
    if block is None:
        chosen = blocks[0]
    elif isinstance(block, int):
        chosen = blocks[block]
    else:
        matches = [b for b in blocks if b.name.lower() == str(block).lower()]
        if not matches:
            raise ValueError(f"{path}: no block {block!r}; blocks: {[b.name for b in blocks]}")
        chosen = matches[0]
    cellpar = [_number(_get(chosen, f"_cell_{tag}")) for tag in
               ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")]
    if np.isnan(cellpar).any():
        raise ValueError(f"{path}: incomplete cell parameters {cellpar}")
    from ase.geometry import cellpar_to_cell
    cell = cellpar_to_cell(cellpar)
    symmetry = _symmetry(chosen)
    symbols, frac, labels, dropped = _sites(chosen, disorder, occupancy_cutoff)
    atoms = expand_sites(symbols, frac, cell, symmetry, merge_tol=merge_tol, labels=labels)
    moved = 0
    if normalize_hydrogens:
        atoms, moved = normalize_hydrogen_positions(atoms, bond_scale=bond_scale)
    for problem in check_structure(atoms):
        if strict:
            raise ValueError(f"{path}: {problem}")
        warnings.warn(problem)
    whole = make_molecules_whole(atoms, bond_scale=bond_scale)
    source = {
        "path": str(path), "block": chosen.name, "experimental_cellpar": list(map(float, cellpar)),
        "temperature_K": _number(_get(chosen, "_diffrn_ambient_temperature", "_cell_measurement_temperature"), None),
        "dropped_disorder_sites": dropped, "hydrogens_normalized": moved, "bond_scale": bond_scale,
        "z_cif": _number(_get(chosen, "_cell_formula_units_z"), None),
    }
    crystal = MolecularCrystal(whole, symmetry, name or chosen.name or Path(path).stem, source)
    if dropped:
        warnings.warn(f"{len(dropped)} minor-disorder sites dropped: {dropped[:10]}")
    return crystal


def crystal_from_atoms(atoms, *, symmetry=None, name="", bond_scale=1.3):
    """MolecularCrystal from any periodic Atoms (e.g. an MD frame or an ASE-built crystal)."""
    return MolecularCrystal(make_molecules_whole(atoms, bond_scale=bond_scale), symmetry, name,
                            {"bond_scale": bond_scale})


def write_cif(crystal, path):
    """Write the crystal's atoms (P1, whole molecules wrapped into the cell) to a CIF file."""
    from ase.io import write
    atoms = crystal.atoms.copy()
    atoms.wrap()
    write(path, atoms, format="cif")
