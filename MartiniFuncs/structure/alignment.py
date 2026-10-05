"""Rigid alignment of finite extracted clusters and their Cartesian vectors."""
from dataclasses import dataclass

import numpy as np


def kabsch_rotation(p, q):
    """Proper rotation R minimizing sum_i |p_i @ R - q_i|^2 for centered row-vector sets p, q.

    SVD(p.T @ q)=U S Vt gives R=U diag(1,1,det(U Vt)) Vt; the determinant
    correction excludes reflections.
    """
    u, _, vt = np.linalg.svd(np.asarray(p, dtype=float).T @ np.asarray(q, dtype=float))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


@dataclass(frozen=True)
class RigidTransform:
    """Row-vector convention: ``positions @ rotation + translation``."""

    rotation: np.ndarray
    translation: np.ndarray

    def positions(self, values):
        """Apply r_out=r_in @ rotation + translation to row-vector positions.
        """
        return np.asarray(values) @ self.rotation + self.translation

    def vectors(self, values):
        """Apply v_out=v_in @ rotation; force/velocity vectors have no translation.
        """
        return np.asarray(values) @ self.rotation


def align_geometry(mobile, reference, indices=None, *, id_array="global_index",
                   vector_arrays=("lammps_forces", "forces", "velocities", "momenta")):
    """Return ``(aligned_copy, rmsd_angstrom, transform)`` for finite clusters.

    ``indices`` selects mobile rows used in the fit. When both structures have
    ``id_array``, reference rows are matched by these unique persistent IDs;
    otherwise row correspondence is assumed. A missing ID array on only one
    structure is an error. No atoms are reordered. Source metadata is retained
    in its original frame. Calculator results are not transferred to the copy.

    Vector arrays listed above are rotated, never translated; pass additional
    Cartesian vector array names explicitly. Scalar arrays remain unchanged.
    Periodic structures must first be extracted/unwrapped as finite clusters.
    Collinear or single-point fits do not determine a unique rotation.

    For matched selected positions p,q, center with pc=mean(p), qc=mean(q).
    SVD((p-pc).T @ (q-qc))=U S Vt gives R=U diag(1,1,det(U Vt)) Vt,
    t=qc-pc @ R. Transform all mobile rows as p @ R+t and vectors as v @ R.
    RMSD=sqrt(mean_selected |p @ R+t-q|^2), in Angstrom, with equal atom
    weights. The determinant correction excludes reflections.
    """
    if np.any(mobile.pbc) or np.any(reference.pbc):
        raise ValueError("Extract finite, unwrapped clusters before alignment")
    selected = np.arange(len(mobile)) if indices is None else np.asarray(indices)
    if (selected.ndim != 1 or selected.dtype.kind not in "iu" or not len(selected)
            or np.any(selected < 0) or np.any(selected >= len(mobile))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("indices must be distinct, nonempty valid mobile row indices")
    reference_rows = selected
    if id_array is not None:
        present = [id_array in a.arrays for a in (mobile, reference)]
        if any(present) and not all(present):
            raise ValueError(f"Both structures must supply {id_array}")
        if all(present):
            mid, rid = (a.arrays[id_array] for a in (mobile, reference))
            for values, atoms in ((mid, mobile), (rid, reference)):
                if values.shape != (len(atoms),) or len(np.unique(values)) != len(values):
                    raise ValueError(f"{id_array} must contain unique scalar IDs")
            lookup = {value: i for i, value in enumerate(rid)}
            try:
                reference_rows = np.array([lookup[value] for value in mid[selected]])
            except KeyError as exc:
                raise ValueError(f"Reference is missing atom ID {exc.args[0]}") from exc
    if np.any(reference_rows >= len(reference)):
        raise ValueError("Reference does not contain all selected rows")
    if not np.array_equal(mobile.numbers[selected], reference.numbers[reference_rows]):
        raise ValueError("Corresponding atoms have different elements")
    p, q = mobile.positions[selected], reference.positions[reference_rows]
    if not np.isfinite(mobile.positions).all() or not np.isfinite(q).all():
        raise ValueError("Coordinates must be finite")
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    rotation = kabsch_rotation(p - pc, q - qc)
    transform = RigidTransform(rotation, qc - pc @ rotation)
    aligned = mobile.copy()
    aligned.positions = transform.positions(mobile.positions)
    # A finite cluster can still carry a display cell. Rotate its axes too.
    aligned.set_cell(mobile.cell.array @ rotation)
    for name in dict.fromkeys(vector_arrays):
        if name == "positions":
            raise ValueError("positions cannot be specified as a vector array")
        if name in mobile.arrays:
            values = mobile.arrays[name]
            if values.shape != (len(mobile), 3) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite (N, 3) Cartesian vector array")
            aligned.arrays[name] = transform.vectors(values).copy()
    rmsd = np.sqrt(np.mean(np.sum((aligned.positions[selected] - q) ** 2, axis=1)))
    return aligned, float(rmsd), transform
