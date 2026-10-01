"""Small persistent-ID helpers: element maps, ID subsets, mass-weighted centers, point wrapping.
"""
from __future__ import annotations
import functools
import inspect
from collections.abc import Iterable, Sequence
import numpy as np
from ase import Atoms
from MartiniFuncs.structure import graph as mfg


def add_global_index(func):
    """Backward-compatible decorator ensuring ``atoms.global_index`` exists."""
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        atoms = bound.arguments.get("atoms")
        if atoms is not None:
            mfg.ensure_index_array(atoms, "global_index")
        return func(*args, **kwargs)

    return wrapper


@add_global_index
def get_element_map(atoms: Atoms, index_array: str = "global_index") -> dict[int, str]:
    """Return ``persistent_atom_id -> element symbol``."""
    ids, _ = mfg.get_index_maps(atoms, index_array=index_array)
    return {
        int(atom_id): symbol
        for atom_id, symbol in zip(ids, atoms.get_chemical_symbols())
    }


def _subset(atoms: Atoms, atom_ids: Iterable[int], index_array="global_index") -> Atoms:
    """Slice ``atoms`` using persistent IDs rather than local ASE indices."""
    local = mfg.local_indices(atoms, atom_ids, index_array=index_array)
    return atoms[local]


def _weighted_com(xyz: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return sum_i(masses[i]*xyz[i])/sum_i masses[i]; xyz is (N,3)
    in Angstrom and masses are positive atomic masses.
    """
    return np.average(np.asarray(xyz, float), axis=0, weights=np.asarray(masses, float))


def _site_center_of_mass(
    atoms: Atoms,
    graph,
    component: Sequence[int],
    site_ids: Iterable[int],
    index_array: str = "global_index",
) -> np.ndarray:
    """COM of a subset of atoms in the *unwrapped frame of its molecule*.

    Unwrap the component with lattice shifts n_i, then compute
    sum_{i in site_ids} m_i*(r_i+n_i @ cell)/sum_{i in site_ids} m_i.
    """
    site_ids = [int(i) for i in site_ids]
    shifts, _ = mfg.unwrap_component(
        atoms,
        graph,
        component,
        index_array=index_array,
    )
    local = mfg.local_indices(atoms, site_ids, index_array=index_array)
    shift_array = np.vstack([shifts[i] for i in site_ids])
    xyz = atoms.positions[local] + shift_array @ atoms.cell.array
    return _weighted_com(xyz, atoms.get_masses()[local])


def _wrap_points(points: np.ndarray, cell, pbc) -> tuple[np.ndarray, np.ndarray]:
    """Wrap arbitrary Cartesian points and return removed integer cell shifts.

    Returns ``wrapped, image`` with ``points = wrapped + image @ cell``.
    Only periodic dimensions are wrapped.

    For row-cell H, f=points @ inverse(H); image=floor(f) only on
    periodic axes and zero elsewhere. wrapped=points-image @ H.
    """
    points = np.asarray(points, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)

    frac = np.linalg.solve(cell.T, points.T).T
    image = np.zeros_like(frac, dtype=int)
    image[:, pbc] = np.floor(frac[:, pbc]).astype(int)
    wrapped = points - image @ cell
    return wrapped, image
