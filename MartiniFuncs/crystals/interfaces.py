"""Lattice matching and two-crystal interfaces (e.g. slabs to collide in MD).

match_lattices finds coincident supercells of two surface lattices with
pymatgen's Zur-McGill search (ZSLGenerator: superlattices up to max_area whose
lengths and angle agree within tolerance), then for each candidate computes
the in-plane deformation that maps B's supercell onto A's. ZSL compares
lengths and unsigned angles, so it also reports mirror-image matches; those
are dropped (ZSL enumerates the other superlattice bases, so a proper match is
reported separately if one exists). Strain is from the polar decomposition F = R U of the 2x2 in-plane
deformation gradient: principal strains are eig(U) - 1, area strain det F - 1.

build_interface repeats both slabs, rotates B in plane by R, and strains it
by U (strain_fraction=1: all strain in B; 0: all in A; 0.5: shared). With
rigid_molecules=True (default) only molecular centres of mass are strained
and each molecule keeps its internal geometry, so bond lengths are not
distorted, but the spacing of the molecules in plane is, by up to the
mismatch. strain_fraction=None strains neither slab: both keep their own
lattice spacing, the cell is A's, B is only rotated, and the mismatch collects
at the cell boundary as a seam (molecules across it are closer or farther
apart than in the crystal by info["seam_misfit_A"], one value per cell
vector). Two incommensurate lattices cannot share one periodic cell without
either strain or such a seam. B is stacked above A with the closest A-B atom pair separated by
gap along z; c has vacuum above and below. Per-atom arrays: "slab" (0 = A,
1 = B), "mol_id" (unique over both), "layer".

Size: the atom count is (in-plane area) x (each slab's thickness). match_lattices
lists the coincident cells (smaller cells need more strain); build_interface's
repeat=(n, m) enlarges any matched cell n x m times at the same strain; slab
thickness is set per crystal with build_slab(layers=... or min_thickness=...).
interface_options tabulates all of it for a pair of faces, filtered by atom
count and strain, and each option's build() makes that interface.

set_collision_velocities gives each slab a rigid velocity along z toward the
other (relative speed in m/s), optionally on top of a Maxwell-Boltzmann
thermal distribution with zero net momentum per slab.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import warnings

import numpy as np
from ase import Atoms, units
from scipy.linalg import fractional_matrix_power, polar

from .slabs import Slab, build_slab, flip_slab, repeat_inplane


@dataclass
class LatticeMatch:
    matrix_a: np.ndarray          # 2x2 integer: A supercell rows in units of A's in-plane vectors
    matrix_b: np.ndarray
    area: float                   # Angstrom^2 of A's supercell
    strain: np.ndarray            # 2x2 symmetric strain tensor (U - I) taking B onto A
    rotation_deg: float           # in-plane rotation of B
    n_atoms: int = 0              # atoms of both slabs in the matched cell

    @property
    def max_strain(self):
        return float(np.abs(np.linalg.eigvalsh(self.strain)).max())

    @property
    def area_strain(self):
        return float(np.linalg.det(self.strain + np.eye(2)) - 1)

    def repeated(self, repeat):
        """The same match on a larger cell: both supercells repeated n x m times (strain unchanged).

        repeat: an integer n (n x n), a pair (n, m), or a 2x2 integer matrix
        acting on the matched supercell vectors.
        """
        N = _repeat_matrix(repeat)
        k = round(abs(np.linalg.det(N)))
        return replace(self, matrix_a=N @ np.asarray(self.matrix_a), matrix_b=N @ np.asarray(self.matrix_b),
                       area=self.area * k, n_atoms=self.n_atoms * k)

    def summary(self):
        return {"matrix_a": self.matrix_a.tolist(), "matrix_b": self.matrix_b.tolist(), "area": self.area,
                "max_strain": self.max_strain, "area_strain": self.area_strain,
                "strain": self.strain.tolist(), "rotation_deg": self.rotation_deg, "n_atoms": self.n_atoms}


def _repeat_matrix(repeat):
    if np.ndim(repeat) == 0:
        N = np.diag([int(repeat)] * 2)
    elif np.ndim(repeat) == 1:
        N = np.diag([int(x) for x in repeat])
    else:
        N = np.asarray(repeat, dtype=int)
    if N.shape != (2, 2) or round(np.linalg.det(N)) < 1:
        raise ValueError(f"repeat must be a positive integer, a pair of them or a 2x2 matrix with det >= 1: {repeat}")
    return N


def _vectors(x):
    if isinstance(x, Slab):
        return x.surface_vectors()
    x = np.asarray(x.cell.array if isinstance(x, Atoms) else x, dtype=float)
    return x[:2, :2]


def deformation(cell_from, cell_to):
    """F (2x2, row-vector convention r' = r @ F) with cell_from @ F = cell_to, and its polar parts."""
    F = np.linalg.solve(np.asarray(cell_from, float), np.asarray(cell_to, float))
    R, U = polar(F.T)                      # column convention: F.T = R U
    return F, R, U


def match_lattices(a, b, *, max_area=400.0, max_length_tol=0.03, max_angle_tol=0.01, max_area_ratio_tol=0.09,
                   max_strain=None, max_matches=20, sort="area"):
    """Coincident supercells of surface lattices a and b (Slabs, slab Atoms or 2x2 in-plane cells).

    Returns LatticeMatch objects, smallest area first (sort="area") or smallest
    max principal strain first (sort="strain"). max_strain filters on the
    largest principal strain. n_atoms is filled in when a and b are Slabs.
    """
    from pymatgen.analysis.interfaces.zsl import ZSLGenerator
    for name, s in (("a", a), ("b", b)):
        if isinstance(s, Slab) and round(abs(np.linalg.det(s.inplane))) > 1:
            warnings.warn(f"slab {name} is already an in-plane supercell ({s.inplane.tolist()}); matches are "
                          "searched only among its multiples. Build slabs for matching without supercell/"
                          "min_lengths: build_interface makes the matched supercell.")
    va, vb = _vectors(a), _vectors(b)
    pad = lambda v: np.hstack([v, np.zeros((2, 1))])
    generator = ZSLGenerator(max_area_ratio_tol=max_area_ratio_tol, max_area=max_area,
                             max_length_tol=max_length_tol, max_angle_tol=max_angle_tol)
    found, seen = [], set()
    for match in generator(pad(vb), pad(va)):
        sa = np.asarray(match.substrate_sl_vectors)[:, :2]
        sb = np.asarray(match.film_sl_vectors)[:, :2]
        if np.linalg.det(sa) < 0:                     # same correspondence, right-handed order
            sa, sb = sa[[1, 0]], sb[[1, 0]]
        F, R, U = deformation(sb, sa)
        if np.linalg.det(F) <= 0:                     # mirror image: not reachable by a rotation
            continue
        Ma = np.rint(sa @ np.linalg.inv(va)).astype(int)
        Mb = np.rint(sb @ np.linalg.inv(vb)).astype(int)
        if not (np.allclose(Ma @ va, sa, atol=1e-6) and np.allclose(Mb @ vb, sb, atol=1e-6)):
            continue
        strain = U - np.eye(2)
        if max_strain is not None and np.abs(np.linalg.eigvalsh(strain)).max() > max_strain:
            continue
        key = (tuple(Ma.ravel()), tuple(Mb.ravel()))
        if key in seen:
            continue
        seen.add(key)
        n_atoms = 0
        if isinstance(a, Slab) and isinstance(b, Slab):
            n_atoms = int(len(a.atoms) * round(np.linalg.det(Ma)) + len(b.atoms) * round(np.linalg.det(Mb)))
        angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        found.append(LatticeMatch(Ma, Mb, float(abs(np.linalg.det(sa))), strain, angle, n_atoms))
    if sort == "strain":
        found.sort(key=lambda m: (round(m.max_strain, 4), m.area))
    else:
        found.sort(key=lambda m: (round(m.area, 3), m.max_strain))
    return found[:max_matches]


def _apply_inplane(atoms, M, rigid):
    """Apply in-plane linear map M (2x2, rows convention) to atoms; rigid keeps molecules undeformed."""
    out = atoms.copy()
    full = np.eye(3)
    full[:2, :2] = M
    if rigid:
        # r' = r @ M with M = U R^T (M.T = R U): molecules rotate by R, their COMs follow M.
        R, _ = polar(M.T)
        rot = np.eye(3)
        rot[:2, :2] = R
        masses = out.get_masses()
        mol = out.arrays["mol_id"]
        for m in np.unique(mol):
            idx = mol == m
            com = masses[idx] @ out.positions[idx] / masses[idx].sum()
            out.positions[idx] = (out.positions[idx] - com) @ rot.T + com @ full
    else:
        out.positions = out.positions @ full
    out.set_cell(atoms.cell.array @ full, scale_atoms=False)
    return out


def build_interface(slab_a: Slab, slab_b: Slab, match: LatticeMatch, *, gap=3.0, vacuum=20.0,
                    strain_fraction=1.0, rigid_molecules=True, repeat=1):
    """B stacked above A in the matched supercell, the closest A-B atoms gap (Angstrom) apart along z.

    B's bottom face (-h -k -l) meets A's top face (h k l). To bring B's top face
    down instead, match and build with slabs.flip_slab(slab_b): flipping mirrors
    the in-plane lattice, so it must happen before match_lattices. Returns an
    Atoms object with "slab", "mol_id" and "layer" arrays and the match and the
    strains applied to each slab in atoms.info.

    strain_fraction: 1 strains only B onto A's lattice, 0 only A onto B's, 0.5
    shares it; None strains neither (A's cell, B only rotated, the mismatch
    left as a seam at the cell boundary; see the module docstring).
    repeat: enlarge the matched cell, e.g. (2, 1) doubles it along its first
    vector; strain is unchanged, a seam misfit grows with the cell.
    """
    match = match.repeated(repeat)
    A = repeat_inplane(slab_a.atoms, match.matrix_a)
    B = repeat_inplane(slab_b.atoms, match.matrix_b)
    ca, cb = A.cell.array[:2, :2], B.cell.array[:2, :2]
    F, R, _ = deformation(cb, ca)
    seam = None
    if strain_fraction is None:
        target = ca
        Fa = np.eye(2)
        Fb = R.T                                    # rotation only (row-vector convention)
        seam = target - cb @ Fb                     # A's cell vectors minus B's own, rotated
    else:
        if not 0.0 <= float(strain_fraction) <= 1.0:
            raise ValueError("strain_fraction must be between 0 and 1, or None for no strain")
        target = cb @ np.real(fractional_matrix_power(F, strain_fraction))
        Fa = np.linalg.solve(ca, target)
        Fb = np.linalg.solve(cb, target)
    A = _apply_inplane(A, Fa, rigid_molecules)
    B = _apply_inplane(B, Fb, rigid_molecules)
    if seam is not None:
        # B keeps its lattice; put each B molecule's centre of mass inside the common cell
        from .slabs import _wrap_molecules
        common = np.eye(3)
        common[:2, :2] = target
        B.positions = _wrap_molecules(B.positions, B.arrays["mol_id"], B.get_masses(), common, axes=(0, 1))
    # stack along z
    za, zb = A.positions[:, 2], B.positions[:, 2]
    A.positions[:, 2] -= za.min()
    top_a = za.max() - za.min()
    B.positions[:, 2] += top_a + gap - zb.min()
    total = B.positions[:, 2].max()
    shift = vacuum / 2
    A.positions[:, 2] += shift
    B.positions[:, 2] += shift
    n_mol_a = int(A.arrays["mol_id"].max()) + 1
    B.arrays["mol_id"] = B.arrays["mol_id"] + n_mol_a
    A.arrays["slab"] = np.zeros(len(A), int)
    B.arrays["slab"] = np.ones(len(B), int)
    cell = np.zeros((3, 3))
    cell[:2, :2] = target
    cell[2, 2] = total + vacuum
    A.set_cell(cell, scale_atoms=False)
    B.set_cell(cell, scale_atoms=False)
    interface = A + B
    interface.pbc = (True, True, False)
    interface.info = {
        "hkl_a": list(slab_a.hkl), "hkl_b": list(slab_b.hkl), "gap": float(gap), "vacuum": float(vacuum),
        "strain_fraction_b": None if strain_fraction is None else float(strain_fraction), "match": match.summary(),
        "strained": strain_fraction is not None,
        "strain_a": (polar(Fa.T)[1] - np.eye(2)).tolist(), "strain_b": (polar(Fb.T)[1] - np.eye(2)).tolist(),
        "match_area": match.area, "area": float(abs(np.linalg.det(target))),
    }
    if seam is not None:
        interface.info["seam_misfit"] = seam.tolist()
        interface.info["seam_misfit_A"] = np.linalg.norm(seam, axis=1).tolist()
    return interface


def set_collision_velocities(atoms, speed_m_s, *, temperature_K=None, seed=None, moving="both"):
    """Velocities for a head-on collision of the two slabs of an interface ("slab" array 0 below, 1 above).

    speed_m_s: relative approach speed. moving="both" gives each slab half
    (A up, B down), "a" or "b" moves only that slab. temperature_K adds a
    Maxwell-Boltzmann thermal part with zero net momentum per slab.
    Velocities are set on atoms in ASE units and returned.
    """
    slab = atoms.arrays["slab"]
    v = np.zeros((len(atoms), 3))
    if temperature_K:
        rng = np.random.default_rng(seed)
        masses = atoms.get_masses()
        sigma = np.sqrt(units.kB * temperature_K / masses)[:, None]
        v = rng.normal(size=(len(atoms), 3)) * sigma
        for s in (0, 1):
            idx = slab == s
            v[idx] -= masses[idx] @ v[idx] / masses[idx].sum()
    speed = speed_m_s * 1e-5 / units.fs          # 1 m/s = 1e-5 Angstrom/fs; ASE velocity = (Angstrom/fs) / units.fs
    share = {"both": (0.5, 0.5), "a": (1.0, 0.0), "b": (0.0, 1.0)}[moving]
    v[slab == 0, 2] += share[0] * speed
    v[slab == 1, 2] -= share[1] * speed
    atoms.set_velocities(v)
    return v


def separation_curve(interface, calculator, gaps, *, area=None):
    """Rigid-slab binding curve: energy vs A-B gap (B shifted along z), and the work of separation.

    Returns dict with gaps, energies (eV) and work_of_separation_J_m2 =
    (E(largest gap) - min E) / area. Shifts are relative to the gap the
    interface was built with (atoms.info["gap"]).
    """
    gaps = np.asarray(gaps, dtype=float)
    base = interface.info.get("gap", 0.0)
    area = area or interface.info.get("area") or float(np.linalg.norm(np.cross(*interface.cell.array[:2])))
    energies = []
    upper = interface.arrays["slab"] == 1
    for g in gaps:
        trial = interface.copy()
        trial.positions[upper, 2] += g - base
        trial.cell[2, 2] += max(0.0, g - base)
        trial.calc = calculator
        energies.append(trial.get_potential_energy())
    energies = np.array(energies)
    work = (energies[np.argmax(gaps)] - energies.min()) / area * 16.02176634
    return {"gaps": gaps.tolist(), "energies": energies.tolist(), "work_of_separation_J_m2": float(work)}


@dataclass
class InterfaceOption:
    """One interface size: a coincident cell (possibly repeated) and a thickness for each slab."""

    match: LatticeMatch           # already repeated
    repeat: tuple
    layers_a: int
    layers_b: int
    thickness_a: float            # layers * d_hkl (Angstrom)
    thickness_b: float
    n_atoms: int
    cell_lengths: tuple           # in-plane cell vector lengths (Angstrom)
    seam_misfit: tuple            # Angstrom per cell vector if built with strain_fraction=None
    _source: dict = field(default=None, repr=False)

    @property
    def area(self):
        return self.match.area

    @property
    def max_strain(self):
        return self.match.max_strain

    def summary(self):
        return {"area": self.area, "cell_lengths": list(self.cell_lengths), "max_strain": self.max_strain,
                "repeat": list(self.repeat), "layers_a": self.layers_a, "layers_b": self.layers_b,
                "thickness_a": self.thickness_a, "thickness_b": self.thickness_b, "n_atoms": self.n_atoms,
                "seam_misfit": list(self.seam_misfit), "matrix_a": self.match.matrix_a.tolist(),
                "matrix_b": self.match.matrix_b.tolist()}

    def build(self, *, gap=3.0, vacuum=20.0, strain_fraction=0.5, rigid_molecules=True):
        """Build this interface (the slabs are cut again from the crystals given to interface_options)."""
        s = self._source
        a = build_slab(s["crystal_a"], s["hkl_a"], layers=self.layers_a, termination=s["termination_a"],
                       vacuum=vacuum)
        b = build_slab(s["crystal_b"], s["hkl_b"], layers=self.layers_b, termination=s["termination_b"],
                       vacuum=vacuum)
        if s["flip_b"]:
            b = flip_slab(b)
        return build_interface(a, b, self.match, gap=gap, vacuum=vacuum, strain_fraction=strain_fraction,
                               rigid_molecules=rigid_molecules)


def interface_options(crystal_a, hkl_a, crystal_b, hkl_b, *, termination_a=0, termination_b=0, flip_b=False,
                      thicknesses=(10.0, 15.0, 20.0, 30.0), repeats=((1, 1), (2, 1), (1, 2), (2, 2)),
                      min_atoms=None, max_atoms=None, max_strain=0.02, max_area=1500.0,
                      max_length_tol=0.04, max_angle_tol=0.04, all_cells=False):
    """Interface sizes for two faces: every (cell, repeat, thickness) within the limits, smallest first.

    thicknesses: slab thicknesses in Angstrom (each slab gets the fewest whole
    layers reaching it), or (thickness_a, thickness_b) pairs to size them
    separately. repeats: in-plane repeats of each matched cell. Coincident
    cells come from match_lattices; by default only cells on the size/strain
    front are kept (no smaller cell has less or equal strain), all_cells=True
    keeps every match. min_atoms/max_atoms/max_strain filter the result.
    Returns InterfaceOption objects sorted by atom count, then strain; call
    option.build(strain_fraction=...) to make one. Atom counts are exact.
    """
    from .slabs import terminations as _terminations
    ta = _terminations(crystal_a, hkl_a)[termination_a]
    tb = _terminations(crystal_b, hkl_b)[termination_b]
    a1 = build_slab(crystal_a, hkl_a, layers=1, termination=ta)
    b1 = build_slab(crystal_b, hkl_b, layers=1, termination=tb)
    if flip_b:
        b1 = flip_slab(b1)
    matches = match_lattices(a1, b1, max_area=max_area, max_length_tol=max_length_tol,
                             max_angle_tol=max_angle_tol, max_strain=max_strain, max_matches=10 ** 6, sort="area")
    if not all_cells:
        front, lowest = [], np.inf
        for m in sorted(matches, key=lambda m: (round(m.area, 3), m.max_strain)):
            if m.max_strain < lowest - 1e-9:
                front.append(m)
                lowest = m.max_strain
        matches = front
    pairs = [(t, t) if np.ndim(t) == 0 else tuple(t) for t in thicknesses]
    va, vb = a1.atoms.cell.array[:2, :2], b1.atoms.cell.array[:2, :2]
    source = {"crystal_a": crystal_a, "crystal_b": crystal_b, "hkl_a": tuple(hkl_a), "hkl_b": tuple(hkl_b),
              "termination_a": ta, "termination_b": tb, "flip_b": flip_b}
    options = []
    for m0 in matches:
        for rep in repeats:
            m = m0.repeated(rep)
            ca, cb = m.matrix_a @ va, m.matrix_b @ vb
            _, R, _ = deformation(cb, ca)
            seam = tuple(float(x) for x in np.linalg.norm(ca - cb @ R.T, axis=1))
            na, nb = round(abs(np.linalg.det(m.matrix_a))), round(abs(np.linalg.det(m.matrix_b)))
            for t_a, t_b in pairs:
                la = max(1, int(np.ceil(float(t_a) / a1.d_spacing - 1e-9)))
                lb = max(1, int(np.ceil(float(t_b) / b1.d_spacing - 1e-9)))
                n = la * na * len(a1.atoms) + lb * nb * len(b1.atoms)
                if (min_atoms and n < min_atoms) or (max_atoms and n > max_atoms):
                    continue
                options.append(InterfaceOption(m, tuple(int(x) for x in np.diag(_repeat_matrix(rep))), la, lb,
                                               la * a1.d_spacing, lb * b1.d_spacing, int(n),
                                               tuple(float(x) for x in np.linalg.norm(ca, axis=1)), seam, source))
    options.sort(key=lambda o: (o.n_atoms, o.max_strain))
    return options
