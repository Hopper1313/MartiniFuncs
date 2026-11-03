"""Which rows of an exported finite structure form the pair, its two members and each shell molecule."""
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from MartiniFuncs.structure import graph as mfg

SHELL_LABEL = "shell"


@dataclass(frozen=True)
class Fragments:
    """Row groups of one structure from pairs.extxyz or clusters.extxyz.

    first/second are the rows of the two pair members (pair_role equal to the
    pair model's role_labels, as written by the export); shell holds one row array
    per whole shell molecule (pair_role "shell"), found as connected components
    of the bond graph of the shell atoms. Rows index the structure as read.
    """
    first: np.ndarray
    second: np.ndarray
    shell: tuple
    role_labels: tuple

    @classmethod
    def from_structure(cls, atoms, *, role_labels=None, scale=1.3):
        """Fragments of a finite exported structure (needs the per-atom pair_role array).

        role_labels None reads them from the structure: the first two non-shell
        pair_role labels in row order, which exports write first member first.
        """
        if "pair_role" not in atoms.arrays:
            raise ValueError("Structure has no pair_role array; use pairs.extxyz / clusters.extxyz from extract")
        if np.any(atoms.pbc):
            raise ValueError("Expected a finite (nonperiodic) exported structure")
        roles = np.asarray(atoms.arrays["pair_role"]).astype(str)
        if role_labels is None:
            role_labels = tuple(dict.fromkeys(label for label in roles if label != SHELL_LABEL))[:2]
        role_labels = tuple(role_labels)
        unknown = sorted(set(roles) - {*role_labels, SHELL_LABEL})
        if unknown:
            raise ValueError(f"Unknown pair_role labels {unknown}; expected {role_labels} or {SHELL_LABEL!r}")
        first, second = (np.flatnonzero(roles == label) for label in role_labels)
        if not len(first) or not len(second):
            raise ValueError(f"Structure lacks atoms of pair member {role_labels[0] if not len(first) else role_labels[1]}")
        shell_rows = np.flatnonzero(roles == SHELL_LABEL)
        shell = ()
        if len(shell_rows):
            subset = Atoms(np.asarray(atoms.get_chemical_symbols())[shell_rows].tolist(),
                           positions=atoms.positions[shell_rows])
            bonds = mfg.build_bond_graph(subset, scale=scale, periodic=False, index_array=None)
            components = mfg.connected_components(bonds, atom_ids=range(len(subset)))
            shell = tuple(sorted((shell_rows[np.sort(c)] for c in components), key=lambda rows: rows[0]))
        return cls(first, second, shell, role_labels)

    @property
    def pair(self):
        """Rows of both pair members, first member then second."""
        return np.concatenate((self.first, self.second))

    def members(self):
        """[(role label, rows)] of the two pair members."""
        return [(self.role_labels[0], self.first), (self.role_labels[1], self.second)]

    def shell_molecules(self):
        """[(label, rows)] of the shell molecules, labelled shell_0, shell_1, ..."""
        return [(f"{SHELL_LABEL}_{k}", rows) for k, rows in enumerate(self.shell)]
