"""Periodic-cell geometry: minimum images, plane spacings, point wrapping, affine steps, and TOR COM unwrapping.
"""
import numpy as np


def is_diagonal(cell):
    """True if the row-vector cell is exactly diagonal (orthorhombic, no tilt)."""
    return np.array_equal(cell, np.diag(np.diagonal(cell)))


def minimum_image(vectors, cell):
    """Shift Cartesian vectors (..., 3) by lattice vectors to fractional components in [-1/2, 1/2).

    d' = d - floor(d H^-1 + 1/2) H, with H the row-vector cell. For a diagonal
    cell this is the componentwise d - L floor(d/L + 1/2) used by TOR. It is
    the nearest image whenever |d| is below half the smallest plane spacing,
    which holds for per-frame molecular steps.
    """
    cell = np.asarray(cell, dtype=float)
    d = np.asarray(vectors, dtype=float)
    if is_diagonal(cell):
        lengths = np.diagonal(cell)
        return d - lengths * np.floor(d / lengths + .5)
    fractional = np.linalg.solve(cell.T, d.reshape(-1, 3).T).T.reshape(d.shape)
    return (fractional - np.floor(fractional + .5)) @ cell


def cell_plane_spacings(cell):
    """Distances between opposite cell faces, V/|a_j x a_k|, in Angstrom."""
    cell = np.asarray(cell, dtype=float)
    volume = abs(np.linalg.det(cell))
    return np.array([volume / np.linalg.norm(np.cross(cell[(i+1) % 3], cell[(i+2) % 3])) for i in range(3)])


def affine_step(previous_points, steps, previous_cell, cell, previous_origin, origin):
    """Displacement over one frame of points that ride with the cell deformation.

    A point with fixed fractional coordinates s, r = o + s H, moves by
    (o_t - o_(t-1)) + s (H_t - H_(t-1)). s is taken at the step midpoint,
    s = (s_(t-1) + s_t)/2 with s_t computed for the continuous (not rewrapped)
    end point r_(t-1) + step, which is second-order accurate in the frame
    spacing. Zero when cell and origin are unchanged. Row-vector cells.
    """
    p = np.asarray(previous_points, dtype=float).reshape(-1, 3)
    d = np.asarray(steps, dtype=float).reshape(-1, 3)
    previous_cell, cell = np.asarray(previous_cell, dtype=float), np.asarray(cell, dtype=float)
    previous_origin, origin = np.asarray(previous_origin, dtype=float), np.asarray(origin, dtype=float)
    if np.array_equal(previous_cell, cell) and np.array_equal(previous_origin, origin):
        return np.zeros(np.shape(steps))
    s_before = np.linalg.solve(previous_cell.T, (p - previous_origin).T).T
    s_after = np.linalg.solve(cell.T, (p + d - origin).T).T
    shift = (origin - previous_origin) + 0.5 * (s_before + s_after) @ (cell - previous_cell)
    return shift.reshape(np.shape(steps))


def wrap_points(points, cell, pbc):
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


def wrap_about_center(points, cell, center):
    """The image of each point inside the cell centered at center: c + mic(r - c)."""
    center = np.asarray(center, dtype=float)
    return center + minimum_image(np.asarray(points, dtype=float) - center, cell)


def unwrap_com_trajectory(com, cells, centers=None):
    """TOR temporal unwrapping of whole-molecule COMs for any periodic cell.

    Each frame's COMs are first wrapped into that frame's cell about its center
    c_t, W_t = c_t + mic(R_t - c_t, H_t) (centers=None uses c_t = 0, i.e. input
    already wrapped about the origin). Then u_0 = W_0 - c_0 and
    u_t = u_(t-1) + mic(W_t - W_(t-1), H_t), with mic the minimum image in the
    destination cell (Bullerjahn et al., JCTC 19 (2023) 3406, Eq. 2, for
    row-vector cells). For orthorhombic cells with a fixed center this is
    exactly the componentwise TOR update on centered wrapped COMs. The
    canonical centered wrap matters under barostat dilation: the image chosen
    for a COM changes its affine displacement by (image) x dL. A moving center
    (e.g. a growing shear tilt) contributes its true lab-frame motion.
    Displacements are lab-frame: an imposed shear flow or barostat dilation is
    not subtracted. Requires inter-sample steps below half the smallest plane
    spacing; that cannot be verified from wrapped data.
    """
    com = np.asarray(com, dtype=float)
    cells = np.asarray(cells, dtype=float)
    centers = np.zeros((len(com), 3)) if centers is None else np.asarray(centers, dtype=float)
    if (com.ndim != 3 or com.shape[-1] != 3 or len(com) < 1 or cells.shape != (len(com), 3, 3)
            or centers.shape != (len(com), 3) or not np.isfinite(com).all() or not np.isfinite(cells).all()):
        raise ValueError("Expected finite COMs (T, N, 3), cells (T, 3, 3) and centers (T, 3)")
    wrapped = np.stack([wrap_about_center(r, h, c) for r, h, c in zip(com, cells, centers)])
    steps = [minimum_image(wrapped[t] - wrapped[t-1], cells[t]) for t in range(1, len(com))]
    start = wrapped[:1] - centers[0]
    return np.concatenate((start, start + np.cumsum(steps, axis=0)), axis=0) if steps else start


def unwrap_orthorhombic_com(wrapped_com, box_lengths):
    """TOR temporal unwrapping for whole-molecule COMs in orthorhombic NPT cells.

    Input shape (frames, molecules, 3), Angstrom, with COMs wrapped consistently
    into origin-centered boxes; box_lengths has shape (frames, 3). Molecules
    must first be made whole. Requires inter-sample motion below half a box in
    every direction; that assumption cannot be established from wrapped data
    alone. Does not support shear, moving origins, or atomwise unwrapping.

    Implements Eq. 2 of Bullerjahn et al., JCTC 19 (2023), 3406-3417,
    doi:10.1021/acs.jctc.3c00308, using the destination box length.

    Componentwise, d_t=w_t-w_(t-1), delta_t=d_t-L_t*floor(d_t/L_t+1/2),
    u_0=w_0 and u_t=u_(t-1)+delta_t. w=wrapped_com and L=box_lengths;
    L_t is the destination frame's box, not the initial box. No drift or
    affine box-motion subtraction is added. This is unwrap_com_trajectory with
    diagonal cells; use that function directly for triclinic/sheared cells.
    """
    wrapped = np.asarray(wrapped_com, dtype=float)
    lengths = np.asarray(box_lengths, dtype=float)
    if (wrapped.ndim != 3 or wrapped.shape[-1] != 3 or len(wrapped) < 1
            or lengths.shape != (len(wrapped), 3) or not np.isfinite(wrapped).all()
            or not np.isfinite(lengths).all() or np.any(lengths <= 0)):
        raise ValueError("Expected finite COMs (T, N, 3) and positive lengths (T, 3)")
    if np.any(abs(wrapped) > lengths[:, None, :] / 2 + 1e-8):
        raise ValueError("COMs must be wrapped into origin-centered boxes")
    return unwrap_com_trajectory(wrapped, np.stack([np.diag(L) for L in lengths]), np.zeros((len(wrapped), 3)))
