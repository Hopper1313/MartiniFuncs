"""Slabs of whole molecules cut from a MolecularCrystal along a Miller plane.

Geometry. For a primitive (h k l) of the crystal's cell, miller_basis finds an
integer basis (u, v, w) of the lattice with u, v in the plane (Gauss-reduced:
short and as orthogonal as the lattice allows) and h.w = 1, so the cell
(u, v, w) has the same volume as the bulk cell and stacks one interplanar
spacing d_hkl per w. The fractional coordinate along w of a point is its
height above the plane in units of d_hkl. u x v points along +G_hkl, so the top
face of a slab is (h k l) and the bottom face is (-h -k -l); the two are
equivalent when the crystal is centrosymmetric.

Molecules are never cut: a molecule belongs to a layer by its centre of mass.
A termination is a height c (fraction of d_hkl) of the cut plane; candidates
lie midway between consecutive molecular COM heights. terminations() ranks them
by the number of intermolecular atom contacts (shorter than contact_cutoff,
4 Angstrom) the plane crosses per nm^2, a calculator-free proxy for the
cleavage energy, and reports the COM gap at the cut.

A slab is n layers (n d_hkl of COM heights) thick, repeated in plane by a 2x2
integer matrix (rows: new in-plane vectors in units of u, v) and rotated so the
first in-plane vector is along x and the surface normal along z, with vacuum
along z (the c vector is perpendicular to the surface). Per-atom arrays:
"mol_id" (molecule), "layer" (0 = bottom), and the slab's metadata is in
atoms.info for Extended XYZ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations, product
from math import gcd

import numpy as np
from ase import Atoms

from .._checks import integer, positive
from ..structure.graph import periodic_pairs
from .cif import MolecularCrystal


# ----------------------------------------------------------------------------- lattice geometry


def primitive_miller(hkl):
    """(h, k, l) divided by their common divisor; (0, 0, 0) raises."""
    hkl = tuple(int(x) for x in hkl)
    if len(hkl) != 3 or not any(hkl):
        raise ValueError(f"Miller index must be three integers, not all zero: {hkl}")
    g = reduce(gcd, (abs(x) for x in hkl))
    return tuple(x // g for x in hkl)


def d_spacing(cell, hkl):
    """Interplanar spacing (Angstrom) of the (h k l) planes of cell (rows = lattice vectors)."""
    G = np.asarray(hkl, dtype=float) @ np.linalg.inv(np.asarray(cell, dtype=float)).T
    return float(1.0 / np.linalg.norm(G))


def reduce_2d(vectors):
    """Gauss (Lagrange) reduction of two lattice vectors (rows, any dimension).

    Returns (reduced, T) with reduced = T @ vectors, T integer and unimodular,
    |reduced[0]| <= |reduced[1]| and |reduced[0].reduced[1]| <= |reduced[0]|^2 / 2.
    The orientation (sign of the cross product) is kept.
    """
    a, b = np.asarray(vectors, dtype=float)
    T = np.eye(2, dtype=int)
    for _ in range(1000):
        if a @ a > b @ b:
            a, b = b, a
            T = T[[1, 0]]
        m = int(np.rint((a @ b) / (a @ a)))
        if m == 0:
            break
        b = b - m * a
        T[1] -= m * T[0]
    if np.linalg.det(T) < 0:                      # keep handedness: (a, b) -> (a, -b) is still reduced
        b = -b
        T[1] *= -1
    return np.array([a, b]), T


def miller_basis(cell, hkl):
    """Integer matrix P (rows u, v, w in units of the cell vectors) for the (h k l) surface.

    u, v span the plane (reduced), h.w = 1 and w's in-plane component is
    minimal; det P = +1 and (u x v) . G_hkl > 0.
    """
    hkl = np.array(primitive_miller(hkl))
    cell = np.asarray(cell, dtype=float)
    m = int(np.abs(hkl).max()) + 1
    grid = np.array(list(product(range(-m, m + 1), repeat=3)))
    in_plane = grid[(grid @ hkl == 0) & grid.any(axis=1)]
    in_plane = in_plane[np.argsort(np.linalg.norm(in_plane @ cell, axis=1), kind="stable")]
    basis = None
    for a, b in combinations(in_plane[:60], 2):
        cross = np.cross(a, b)
        if np.array_equal(np.abs(cross), np.abs(hkl)):
            basis = np.array([a, b])
            break
    if basis is None:
        raise ValueError(f"could not find an in-plane lattice basis for {tuple(hkl)}")
    _, T = reduce_2d(basis @ cell)
    u, v = T @ basis
    G = hkl @ np.linalg.inv(cell).T
    if np.cross(u @ cell, v @ cell) @ G < 0:
        u, v = v, u
    candidates = grid[grid @ hkl == 1]
    w = candidates[np.argmin(np.linalg.norm(candidates @ cell, axis=1))]
    # Remove whole multiples of u, v so that w is as close to the normal as possible.
    plane = np.array([u, v]) @ cell
    coef = np.linalg.lstsq(plane.T, w @ cell, rcond=None)[0]
    w = w - np.rint(coef).astype(int) @ np.array([u, v])
    P = np.array([u, v, w], dtype=int)
    if round(np.linalg.det(P)) != 1:
        raise RuntimeError(f"internal error: det P = {np.linalg.det(P)} for {tuple(hkl)}")
    return P


def wrap_fraction(x, tol=1e-9):
    """Fractions wrapped into [0, 1), with values within tol of an integer snapped to 0.

    np.mod(-1e-17, 1.0) is 1.0, which would put a molecule on a special position
    (an inversion centre at height 0) one whole layer too high.
    """
    x = np.asarray(x, dtype=float)
    wrapped = x - np.floor(x + tol)
    return np.where(np.abs(wrapped) < tol, 0.0, wrapped)


def _wrap_molecules(positions, mol_id, masses, cell, axes=(0, 1, 2)):
    """Translate each molecule by a lattice vector so its COM fraction is in [0, 1) on the given axes."""
    positions = positions.copy()
    cell = np.asarray(cell, dtype=float)
    inverse = np.linalg.inv(cell)
    for m in np.unique(mol_id):
        idx = mol_id == m
        com = masses[idx] @ positions[idx] / masses[idx].sum()
        frac = com @ inverse
        shift = np.zeros(3)
        shift[list(axes)] = np.floor(frac[list(axes)] + 1e-9)
        positions[idx] -= shift @ cell
    return positions


def _coms(positions, mol_id, masses):
    ids = np.unique(mol_id)
    return ids, np.array([masses[mol_id == m] @ positions[mol_id == m] / masses[mol_id == m].sum() for m in ids])


@dataclass
class OrientedBulk:
    """The bulk crystal in the (u, v, w) cell of one Miller plane; molecule COMs inside the cell."""

    atoms: Atoms
    hkl: tuple
    P: np.ndarray
    d_spacing: float
    com_heights: np.ndarray          # fraction of d_hkl, per molecule, in [0, 1)


def orient(crystal: MolecularCrystal, hkl):
    """OrientedBulk for (h k l) of crystal (Miller indices in the crystal's cell)."""
    hkl = primitive_miller(hkl)
    P = miller_basis(crystal.atoms.cell.array, hkl)
    cell = P @ crystal.atoms.cell.array
    atoms = crystal.atoms.copy()
    masses = atoms.get_masses()
    atoms.set_cell(cell, scale_atoms=False)
    atoms.positions = _wrap_molecules(atoms.positions, atoms.arrays["mol_id"], masses, cell)
    _, coms = _coms(atoms.positions, atoms.arrays["mol_id"], masses)
    heights = wrap_fraction((coms @ np.linalg.inv(cell))[:, 2])
    return OrientedBulk(atoms, hkl, P, d_spacing(crystal.atoms.cell.array, hkl), heights)


# ----------------------------------------------------------------------------- terminations


@dataclass(frozen=True)
class Termination:
    index: int
    shift: float                  # cut-plane height, fraction of d_hkl
    com_gap: float                # Angstrom between the COM planes on either side of the cut
    broken_contacts: int          # intermolecular atom pairs < contact_cutoff crossing one cut, per surface cell
    contacts_per_nm2: float
    n_molecule_layers: int        # distinct COM heights per d_hkl


def terminations(crystal_or_bulk, hkl=None, *, height_tol=0.25, contact_cutoff=4.0, max_terminations=None):
    """Candidate cut planes of (h k l), fewest broken intermolecular contacts first.

    height_tol (Angstrom): COM heights closer than this form one molecular layer
    (no cut between them).
    """
    bulk = crystal_or_bulk if isinstance(crystal_or_bulk, OrientedBulk) else orient(crystal_or_bulk, hkl)
    d = bulk.d_spacing
    heights = np.sort(bulk.com_heights)
    groups = [[heights[0]]]
    for h in heights[1:]:
        if (h - groups[-1][-1]) * d < height_tol:
            groups[-1].append(h)
        else:
            groups.append([h])
    if len(groups) > 1 and (groups[0][0] + 1 - groups[-1][-1]) * d < height_tol:
        groups[0] = [x - 1 for x in groups.pop()] + groups[0]
    centers_low = [g[0] for g in groups]
    centers_high = [g[-1] for g in groups]
    cuts = []
    for k in range(len(groups)):
        top = centers_high[k]
        bottom_next = centers_low[(k + 1) % len(groups)] + (1 if k + 1 == len(groups) else 0)
        cuts.append((float(wrap_fraction((top + bottom_next) / 2)), (bottom_next - top) * d))
    atoms = bulk.atoms
    mol_id = atoms.arrays["mol_id"]
    i, j, S, _ = periodic_pairs(atoms.positions, atoms.cell.array, np.ones(3, bool), contact_cutoff)
    inter = mol_id[i] != mol_id[j]
    i, j, S = i[inter], j[inter], S[inter]
    com_h = bulk.com_heights
    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    results = []
    for shift, gap in cuts:
        zi = com_h[mol_id[i]] - shift
        zj = com_h[mol_id[j]] + S[:, 2] - shift
        crossing = int(np.count_nonzero(np.floor(zi) != np.floor(zj)) // 2)
        results.append((crossing, -gap, shift, gap))
    results.sort()
    out = [Termination(k, float(shift), float(gap), crossing, crossing / area * 100.0, len(groups))
           for k, (crossing, _, shift, gap) in enumerate(results)]
    return out[:max_terminations] if max_terminations else out


# ----------------------------------------------------------------------------- in-plane supercells


def inplane_supercell(vectors, *, min_lengths=None, max_det=64, max_index=8, orthogonal=False,
                      angle_range=(60.0, 120.0)):
    """Smallest 2x2 integer matrix S whose rows S @ vectors are at least min_lengths long.

    vectors: the two in-plane lattice vectors (rows). min_lengths: a number or
    (L1, L2) in Angstrom; None gives the identity. The supercell angle must lie
    in angle_range; orthogonal=True prefers angles close to 90 degrees over a
    smaller area. Ties go to the more orthogonal and then the more compact cell.
    """
    vectors = np.asarray(vectors, dtype=float)
    if min_lengths is None and not orthogonal:
        return np.eye(2, dtype=int)
    if min_lengths is None:
        min_lengths = (0.0, 0.0)
    lengths = np.broadcast_to(np.asarray(min_lengths, dtype=float), (2,))
    r = np.arange(-max_index, max_index + 1)
    S = np.stack(np.meshgrid(r, r, r, r, indexing="ij"), axis=-1).reshape(-1, 2, 2)
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    S, det = S[(det > 0) & (det <= max_det)], det[(det > 0) & (det <= max_det)]
    A = S @ vectors
    la, lb = np.linalg.norm(A, axis=2).T
    angle = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", A[:, 0], A[:, 1]) / (la * lb), -1, 1)))
    ok = ((la >= lengths[0] - 1e-6) & (lb >= lengths[1] - 1e-6)
          & (angle >= angle_range[0] - 1e-6) & (angle <= angle_range[1] + 1e-6))
    if not ok.any():
        raise ValueError(f"no in-plane supercell with det <= {max_det} meets min_lengths={tuple(lengths)} "
                         f"and angle_range={angle_range}")
    skew = np.round(np.abs(angle - 90.0), 1)
    # np.lexsort sorts by the last key first: area (det) then skew, or skew then area when orthogonal.
    keys = (la + lb, det, skew) if orthogonal else (la + lb, skew, det)
    order = np.lexsort(tuple(k[ok] for k in keys))
    return S[ok][order[0]]


# ----------------------------------------------------------------------------- slabs


@dataclass
class Slab:
    atoms: Atoms
    hkl: tuple
    termination: Termination
    layers: int
    d_spacing: float
    P: np.ndarray                 # bulk cell -> (u, v, w)
    inplane: np.ndarray           # 2x2, rows of the slab's in-plane vectors in units of (u, v)
    vacuum: float
    composition: dict = field(default_factory=dict)
    bulk_molecules: int = 0       # molecules in the bulk cell the slab was cut from

    @property
    def area(self):
        """Surface cell area (Angstrom^2), one face."""
        return float(np.linalg.norm(np.cross(self.atoms.cell[0], self.atoms.cell[1])))

    @property
    def n_molecules(self):
        return int(self.atoms.arrays["mol_id"].max()) + 1

    @property
    def thickness(self):
        """n_layers * d_hkl: the COM-plane thickness of the slab (Angstrom)."""
        return self.layers * self.d_spacing

    @property
    def atomic_thickness(self):
        z = self.atoms.positions[:, 2]
        return float(z.max() - z.min())

    @property
    def bulk_units(self):
        """Number of bulk unit cells' worth of material in the slab (layers * det(inplane))."""
        return int(self.layers * round(np.linalg.det(self.inplane)))

    def molecules_in_layers(self, layers):
        """mol_ids of molecules whose layer index is in layers (0 = bottom)."""
        layer = self.atoms.arrays["layer"]
        mol = self.atoms.arrays["mol_id"]
        return np.unique(mol[np.isin(layer, list(layers))])

    def central_atoms(self, n_layers=1):
        """Indices of the atoms of the n_layers middle layers (to hold fixed in a relaxation)."""
        start = (self.layers - n_layers) // 2
        chosen = range(start, start + n_layers)
        return np.flatnonzero(np.isin(self.atoms.arrays["layer"], list(chosen)))

    def surface_vectors(self):
        """In-plane cell (2x2, Angstrom) in the slab's x-y frame (for lattice matching)."""
        return self.atoms.cell.array[:2, :2].copy()

    def info(self):
        return {
            "hkl": list(self.hkl), "termination": self.termination.index,
            "termination_shift": self.termination.shift, "layers": self.layers,
            "d_spacing": self.d_spacing, "thickness": self.thickness, "area": self.area,
            "n_molecules": self.n_molecules, "n_atoms": len(self.atoms), "vacuum": self.vacuum,
            "inplane": self.inplane.tolist(), "P": self.P.tolist(),
            "broken_contacts_per_nm2": self.termination.contacts_per_nm2,
        }


def _surface_frame(a1, a2):
    """Rotation (rows = new x, y, z axes) putting a1 on x and a1 x a2 on z."""
    x = a1 / np.linalg.norm(a1)
    z = np.cross(a1, a2)
    z /= np.linalg.norm(z)
    return np.array([x, np.cross(z, x), z])


def build_slab(crystal: MolecularCrystal, hkl, *, layers=None, min_thickness=None, termination=0,
               vacuum=15.0, supercell=None, min_lengths=None, orthogonal=False, max_atoms=None,
               max_det=64, height_tol=0.25, contact_cutoff=4.0):
    """Slab of whole molecules on the (h k l) plane of crystal.

    layers: number of d_hkl layers; or min_thickness (Angstrom): the fewest
    layers with layers*d_hkl >= min_thickness (default 3 layers).
    termination: index into terminations() (0 = fewest broken contacts) or a
    Termination. supercell: (m, n) or a 2x2 integer matrix on the reduced
    in-plane vectors; otherwise min_lengths/orthogonal choose one with
    inplane_supercell. max_atoms: raise if the slab would be larger.
    """
    bulk = orient(crystal, hkl)
    terms = terminations(bulk, height_tol=height_tol, contact_cutoff=contact_cutoff)
    if isinstance(termination, Termination):
        term = termination
    else:
        index = integer(termination, "termination", minimum=0)
        if index >= len(terms):
            raise ValueError(f"{bulk.hkl} has {len(terms)} terminations; termination={index} is out of range")
        term = terms[index]
    if layers is None:
        layers = 3 if min_thickness is None else max(1, int(np.ceil(positive(min_thickness, "min_thickness")
                                                                      / bulk.d_spacing - 1e-9)))
    layers = integer(layers, "layers")
    cell = bulk.atoms.cell.array
    uv = cell[:2]
    if supercell is not None:
        S = np.diag(supercell) if np.ndim(supercell) == 1 else np.asarray(supercell, dtype=int)
    else:
        S = inplane_supercell(uv, min_lengths=min_lengths, orthogonal=orthogonal, max_det=max_det)
    S = np.asarray(S, dtype=int)
    n_cells = int(round(np.linalg.det(S)))
    if n_cells <= 0:
        raise ValueError(f"supercell matrix must have positive determinant: {S.tolist()}")
    n_atoms = len(bulk.atoms) * n_cells * layers
    if max_atoms is not None and n_atoms > max_atoms:
        raise ValueError(f"slab {bulk.hkl} with {layers} layers and {n_cells} surface cells has {n_atoms} atoms "
                         f"> max_atoms={max_atoms}")

    atoms = bulk.atoms
    masses_cell = atoms.get_masses()
    mol_cell = atoms.arrays["mol_id"]
    n_mol_cell = int(mol_cell.max()) + 1
    A = S @ uv
    # lattice points of (u, v) inside the supercell A
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ S
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    inv_S = np.linalg.inv(S.astype(float))
    points = [(i, j) for i in range(lo[0], hi[0] + 1) for j in range(lo[1], hi[1] + 1)
              if np.all((np.array([i, j]) @ inv_S >= -1e-9) & (np.array([i, j]) @ inv_S < 1 - 1e-9))]
    if len(points) != n_cells:
        raise RuntimeError(f"internal error: {len(points)} lattice points for det {n_cells}")

    positions, numbers, mol_id, layer_of = [], [], [], []
    w = cell[2]
    count = 0
    for k in range(layers):
        for (i, j) in points:
            for m in range(n_mol_cell):
                idx = mol_cell == m
                # A molecule below the cut (h_m < c) is moved up by one w, so every COM of layer k
                # lies in [c + k, c + k + 1) d_hkl.
                base = np.floor(bulk.com_heights[m] - term.shift)
                translation = i * uv[0] + j * uv[1] + (k - base) * w
                positions.append(atoms.positions[idx] + translation)
                numbers.append(atoms.numbers[idx])
                mol_id.append(np.full(idx.sum(), count))
                layer_of.append(np.full(idx.sum(), k))
                count += 1
    positions = np.vstack(positions)
    numbers = np.concatenate(numbers)
    mol_id = np.concatenate(mol_id)
    layer_of = np.concatenate(layer_of)

    R = _surface_frame(A[0], A[1])
    positions = positions @ R.T
    A_rot = A @ R.T
    height = layers * bulk.d_spacing
    slab_cell = np.array([[A_rot[0, 0], 0.0, 0.0], [A_rot[1, 0], A_rot[1, 1], 0.0], [0.0, 0.0, 1.0]])
    masses = np.tile(masses_cell, layers * n_cells)
    positions = _wrap_molecules(positions, mol_id, masses, slab_cell, axes=(0, 1))
    z = positions[:, 2]
    c_length = (z.max() - z.min()) + positive(vacuum, "vacuum")
    positions[:, 2] += c_length / 2 - (z.max() + z.min()) / 2
    slab_cell[2, 2] = c_length
    slab_atoms = Atoms(numbers=numbers, positions=positions, cell=slab_cell, pbc=(True, True, False))
    slab_atoms.arrays["mol_id"] = mol_id
    slab_atoms.arrays["layer"] = layer_of
    composition = {}
    for m in range(count):
        f = slab_atoms[mol_id == m].get_chemical_formula(mode="hill")
        composition[f] = composition.get(f, 0) + 1
    slab = Slab(slab_atoms, bulk.hkl, term, layers, bulk.d_spacing, bulk.P, S, float(vacuum), composition,
                crystal.n_molecules)
    slab_atoms.info.update({k: v for k, v in slab.info().items() if k not in ("inplane", "P")})
    slab_atoms.info["slab_height_com"] = height
    return slab


def slab_supercell(slab: Slab, matrix):
    """The slab repeated in plane by a 2x2 integer matrix (rows in units of the slab's own in-plane vectors)."""
    M = np.diag(matrix) if np.ndim(matrix) == 1 else np.asarray(matrix, dtype=int)
    atoms = repeat_inplane(slab.atoms, M)
    new = Slab(atoms, slab.hkl, slab.termination, slab.layers, slab.d_spacing, slab.P, M @ slab.inplane,
               slab.vacuum, {k: v * int(round(np.linalg.det(M))) for k, v in slab.composition.items()},
               slab.bulk_molecules)
    atoms.info.update({k: v for k, v in new.info().items() if k not in ("inplane", "P")})
    return new


def flip_slab(slab: Slab):
    """The slab turned upside down (180 degrees about x): its (h k l) face now points down.

    The in-plane basis becomes (a1, -a2) so it stays right-handed; layer
    indices are reversed (0 is again the bottom layer).
    """
    atoms = slab.atoms.copy()
    atoms.positions[:, 1:] *= -1
    a1, a2, c = atoms.cell.array
    cell = np.array([a1, [-a2[0], a2[1], 0.0], [0.0, 0.0, c[2]]])
    atoms.positions[:, 2] += c[2]
    atoms.set_cell(cell, scale_atoms=False)
    atoms.positions = _wrap_molecules(atoms.positions, atoms.arrays["mol_id"], atoms.get_masses(), cell, axes=(0, 1))
    atoms.arrays["layer"] = slab.layers - 1 - atoms.arrays["layer"]
    inplane = slab.inplane * np.array([[1], [-1]])
    flipped = Slab(atoms, tuple(-x for x in slab.hkl), slab.termination, slab.layers, slab.d_spacing, slab.P,
                   inplane, slab.vacuum, dict(slab.composition), slab.bulk_molecules)
    atoms.info.update(hkl=list(flipped.hkl), flipped=True)
    return flipped


def repeat_inplane(atoms, M):
    """Copy of a slab-like Atoms (mol_id array, a1 on x, c along z) repeated by in-plane matrix M.

    The result is re-rotated so its first in-plane vector lies along x;
    molecules stay whole and are wrapped by COM.
    """
    M = np.asarray(M, dtype=int)
    n = int(round(np.linalg.det(M)))
    if n <= 0:
        raise ValueError(f"in-plane matrix must have positive determinant: {M.tolist()}")
    ab = atoms.cell.array[:2]
    A = M @ ab
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) @ M
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    inv = np.linalg.inv(M.astype(float))
    points = [(i, j) for i in range(lo[0], hi[0] + 1) for j in range(lo[1], hi[1] + 1)
              if np.all((np.array([i, j]) @ inv >= -1e-9) & (np.array([i, j]) @ inv < 1 - 1e-9))]
    mol = atoms.arrays["mol_id"]
    n_mol = int(mol.max()) + 1
    copies = []
    for k, (i, j) in enumerate(points):
        part = atoms.copy()
        part.positions += i * ab[0] + j * ab[1]
        part.arrays["mol_id"] = mol + k * n_mol
        copies.append(part)
    big = copies[0]
    for part in copies[1:]:
        big += part
    R = _surface_frame(A[0], A[1])
    big.positions = big.positions @ R.T
    A_rot = A @ R.T
    cell = np.array([[A_rot[0, 0], 0, 0], [A_rot[1, 0], A_rot[1, 1], 0], atoms.cell.array[2] @ R.T])
    big.set_cell(cell, scale_atoms=False)
    big.positions = _wrap_molecules(big.positions, big.arrays["mol_id"], big.get_masses(), cell, axes=(0, 1))
    big.info = dict(atoms.info)
    return big
