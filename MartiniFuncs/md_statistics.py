"""MD statistics with explicit spatial scope and physical units.

ASE inputs use ASE masses/momenta. Positions are Angstrom, returned velocities
are Angstrom/ps, time arguments are ps, forces are eV/Angstrom. Array trajectory
inputs must have a fixed object ordering across time; no implicit unwrapping,
alignment, drift subtraction, or species averaging is performed.
"""
from dataclasses import dataclass
from itertools import zip_longest

import numpy as np
from ase import units


def _rows(atoms, atom_ids, id_array="global_index"):
    ids = np.asarray(atoms.arrays[id_array])
    requested = tuple(atom_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Select a nonempty set of distinct persistent atom IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("Persistent atom IDs must be unique")
    lookup = {value: i for i, value in enumerate(ids)}
    try:
        return np.array([lookup[value] for value in requested])
    except KeyError as exc:
        raise ValueError(f"Missing atom ID {exc.args[0]}") from exc


def _velocities(atoms):
    if "momenta" not in atoms.arrays:
        raise ValueError("ASE momenta/velocities are required")
    values = atoms.get_velocities()
    if not np.isfinite(values).all():
        raise ValueError("Velocities must be finite")
    return values


def _positions(atoms, positions_unwrapped):
    if np.any(atoms.pbc) and not positions_unwrapped:
        raise ValueError("Periodic positions must be explicitly unwrapped first")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Positions must be finite")
    return atoms.positions


def get_molecule_center_of_mass(atoms, atom_ids, *, positions_unwrapped=False):
    """One molecule's COM in Angstrom, selected by persistent global_index IDs.

    Unwrapping here means making this molecule whole, not temporal unwrapping.

    R = sum_i(m_i r_i) / sum_i(m_i), using masses and positions at the
    selected persistent IDs. The molecule must be whole; R is in Angstrom.
    """
    rows = _rows(atoms, atom_ids)
    return np.average(_positions(atoms, positions_unwrapped)[rows], axis=0,
                      weights=atoms.get_masses()[rows])


def get_molecule_com_velocity(atoms, atom_ids):
    """One molecule's mass-weighted translational velocity in Angstrom/ps.

    V = sum_i(m_i v_i) / sum_i(m_i). ASE velocities from atoms are
    multiplied by 1000*ase.units.fs to express V in Angstrom/ps.
    """
    rows = _rows(atoms, atom_ids)
    return np.average(_velocities(atoms)[rows], axis=0,
                      weights=atoms.get_masses()[rows]) * (1000 * units.fs)


def get_molecule_net_force(atoms, atom_ids, *, force_array="lammps_forces"):
    """Resultant source force in eV/Angstrom; never invokes a calculator.

    F_mol = sum_i F_i over selected IDs in atoms.arrays[force_array].
    No averaging or subtraction of intramolecular forces is performed.
    """
    return _forces(atoms, force_array)[_rows(atoms, atom_ids)].sum(axis=0)


def _forces(atoms, name):
    if atoms.info.get("force_units", "eV/Angstrom") != "eV/Angstrom":
        raise ValueError("Forces must be in eV/Angstrom")
    if name not in atoms.arrays:
        raise ValueError(f"Missing force array {name}")
    values = np.asarray(atoms.arrays[name], dtype=float)
    if values.shape != (len(atoms), 3) or not np.isfinite(values).all():
        raise ValueError("Forces must be finite with shape (N, 3)")
    return values


def get_frame_kinetic_energy(atoms, *, remove_com=False):
    """Whole-frame kinetic energy in eV, optionally subtracting COM motion.

    K = (1/2) sum_i m_i |v_i - V|^2. Here V is the frame mass-weighted
    COM velocity when remove_com=True, otherwise zero. ASE internal mass and
    velocity units give K in eV; use all atoms in the supplied frame.
    """
    velocities = _velocities(atoms).copy()
    masses = atoms.get_masses()
    if remove_com:
        velocities -= np.average(velocities, axis=0, weights=masses)
    return float(0.5 * np.sum(masses[:, None] * velocities ** 2))


def get_frame_temperature(atoms, *, degrees_of_freedom, remove_com=False):
    """Whole-frame kinetic temperature in K with explicitly supplied DOF.

    Caller accounts for constraints and any removed motion in the DOF. Removing
    COM velocity is not a correction for a shear/flow velocity profile.

    T = 2 K / (f k_B), where K is get_frame_kinetic_energy(atoms,
    remove_com=remove_com), f=degrees_of_freedom, and k_B is in eV/K.
    """
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive and finite")
    return 2 * get_frame_kinetic_energy(atoms, remove_com=remove_com) / (
        degrees_of_freedom * units.kB)


def get_frame_density(atoms):
    """Whole-cell mass density in g/cm³ (not an extracted-cluster density).

    rho = sum_i m_i * ase.units._amu / (|det(cell)| * 1e-30 * 1000).
    Masses are in amu, _amu is kg/amu, cell volume is Angstrom^3; the result
    is g/cm^3.
    """
    volume = atoms.get_volume()
    if volume <= 0:
        raise ValueError("A positive-volume simulation cell is required")
    return float(atoms.get_masses().sum() * units._amu / (volume * 1e-30) / 1000)


def get_frame_force_statistics(atoms, *, force_array="lammps_forces"):
    """Summarize atomic force magnitudes and whole-frame resultant, eV/Angstrom.

    For a_i=|F_i| from force_array, return mean(a_i),
    sqrt(mean(a_i^2)), max(a_i), and sum_i F_i. RMS magnitude averages over
    atoms; it is sqrt(3) times a Cartesian-component RMS.
    """
    forces = _forces(atoms, force_array)
    if not len(forces):
        raise ValueError("Frame must contain atoms")
    norms = np.linalg.norm(forces, axis=1)
    return {"mean_magnitude": float(norms.mean()),
            "rms_magnitude": float(np.sqrt(np.mean(norms ** 2))),
            "max_magnitude": float(norms.max()), "net_force": forces.sum(axis=0)}


def get_molecule_rmsd(first, second, atom_ids, *, align=True, positions_unwrapped=False):
    """Equal-weight RMSD in Angstrom, matching one molecule by persistent IDs.

    RMSD = sqrt(sum_i |a_i R + t - b_i|^2 / N), matching atom_ids in
    both inputs. If align=False, R=I and t=0. Otherwise center both sets,
    SVD(a_centered.T @ b_centered)=U S Vt, and set
    R=U diag(1,1,det(U Vt)) Vt and t=mean(b)-mean(a) R.
    Atoms have equal weights, independent of their masses.
    """
    atom_ids = tuple(atom_ids)
    a = _positions(first, positions_unwrapped)[_rows(first, atom_ids)]
    b = _positions(second, positions_unwrapped)[_rows(second, atom_ids)]
    if not np.array_equal(first.numbers[_rows(first, atom_ids)], second.numbers[_rows(second, atom_ids)]):
        raise ValueError("Matched atom IDs have different elements")
    if align:
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        u, _, vt = np.linalg.svd(a.T @ b)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        a = a @ (u @ correction @ vt)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def get_frame_rmsd(first, second, *, align=False, positions_unwrapped=False):
    """Whole-frame RMSD in Angstrom; collective rigid alignment is opt-in.

    RMSD = sqrt(mean_i |r_first,i R + t - r_second,i|^2) over all IDs.
    R,t are identity/zero unless align=True; then use the equal-weight
    proper-rotation fit documented in get_molecule_rmsd.
    """
    if set(first.arrays["global_index"]) != set(second.arrays["global_index"]):
        raise ValueError("Frames must contain the same persistent atom IDs")
    return get_molecule_rmsd(first, second, first.arrays["global_index"],
                             align=align, positions_unwrapped=positions_unwrapped)


def get_molecule_displacement(first, second, atom_ids, *, positions_unwrapped=False):
    """COM displacement in Angstrom; requires temporally unwrapped positions.

    Delta R = R_second - R_first, with each R=sum(m_i r_i)/sum(m_i)
    for atom_ids. Coordinates must share a temporally unwrapped reference frame.
    """
    atom_ids = tuple(atom_ids)
    return (get_molecule_center_of_mass(second, atom_ids, positions_unwrapped=positions_unwrapped)
            - get_molecule_center_of_mass(first, atom_ids, positions_unwrapped=positions_unwrapped))


def get_pair_relative_velocity(atoms, first_ids, second_ids):
    """Second molecule COM velocity minus first molecule COM velocity, Angstrom/ps.

    Delta V = V_second - V_first; V=sum(m_i v_i)/sum(m_i) for each
    ID group, converted from ASE velocity units to Angstrom/ps.
    """
    return get_molecule_com_velocity(atoms, second_ids) - get_molecule_com_velocity(atoms, first_ids)


def get_molecule_orientation(atoms, axis_atom_ids, *, positions_unwrapped=False):
    """Directed unit axis from the first to second explicitly chosen atom ID.

    This defines an axis, not a complete molecular orientation or plane normal.

    u = (r_second-r_first)/|r_second-r_first|, using axis_atom_ids
    in the given order and whole-molecule positions. u is dimensionless.
    """
    if len(axis_atom_ids) != 2:
        raise ValueError("Choose exactly two atom IDs to define the molecular axis")
    rows = _rows(atoms, axis_atom_ids)
    positions = _positions(atoms, positions_unwrapped)[rows]
    axis = positions[1] - positions[0]
    length = np.linalg.norm(axis)
    if length <= 1e-12:
        raise ValueError("Coincident atoms do not define an orientation")
    return axis / length


def get_pair_force_projections(atoms, first_ids, second_ids, *, direction,
                               force_array="lammps_forces"):
    """Signed molecular resultant projections along an explicit direction.

    Returns first, second, and second-minus-first projections in eV/Angstrom.
    This is not a generalized reaction-coordinate force or environment residual.

    u=direction/|direction|; p_A=(sum_A F_i) dot u and
    p_B=(sum_B F_i) dot u. difference=p_B-p_A, not a reduced-mass or
    reaction-coordinate projection.
    """
    axis = np.asarray(direction, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-12:
        raise ValueError("direction must be a finite nonzero 3-vector")
    axis = axis / np.linalg.norm(axis)
    first = float(get_molecule_net_force(atoms, first_ids, force_array=force_array) @ axis)
    second = float(get_molecule_net_force(atoms, second_ids, force_array=force_array) @ axis)
    return {"first": first, "second": second, "difference": second - first, "direction": axis}


def get_cell_deformation(first, second):
    """Row-vector affine map A satisfying first.cell @ A = second.cell.

    Returns the deformation map, not a strain tensor. Cells must be nonsingular;
    equivalent lattice-basis changes are not automatically identified.

    A = inverse(H_first) H_second, computed by np.linalg.solve rather
    than explicit inversion; H contains lattice vectors as rows.
    """
    a, b = first.cell.array, second.cell.array
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or abs(np.linalg.det(a)) <= 1e-12 or abs(np.linalg.det(b)) <= 1e-12):
        raise ValueError("Two finite nonsingular simulation cells are required")
    return np.linalg.solve(a, b)


def get_pair_geometry(atoms, pair, shift, *, scale=1.3):
    """Existing CPD/dienophile diagnostics, distances in Angstrom, angles in degrees.

    Delegates to pair_selection.measure_pair_geometry: bond distances
    are Euclidean in one placed image; angles are degrees(arccos(|a dot b| /
    (|a||b|))). See that function for the reactive axes and plane definitions.
    """
    from .pair_selection import measure_pair_geometry
    return measure_pair_geometry(atoms, pair, shift, scale=scale)


def get_pair_geometry_timeseries(atoms_by_frame, pairs_by_frame, shifts_by_frame,
                                 times_ps, *, pair, scale=1.3):
    """Track one pair's existing geometry diagnostics across synchronized frames.

    Pair/shift lists are the outputs of get_all_prereactive_pairs for each frame.
    Absence from the candidate list returns geometry=None; invalid geometry has
    an explicit error string. Absence is not evidence of molecular destruction.
    No threshold or orientation criterion is changed here.
    """
    key = pair_identity(pair)
    times = np.asarray(times_ps, dtype=float)
    _times(times, len(times))
    result, sentinel = [], object()
    for atoms, pairs, shifts, time in zip_longest(
            atoms_by_frame, pairs_by_frame, shifts_by_frame, times, fillvalue=sentinel):
        if any(value is sentinel for value in (atoms, pairs, shifts, time)):
            raise ValueError("Supply geometry, pairs, shifts, and time for every frame")
        if len(pairs) != len(shifts):
            raise ValueError("Each candidate pair requires one periodic shift")
        matches = [(p, s) for p, s in zip(pairs, shifts) if pair_identity(p) == key]
        if len(matches) > 1:
            raise ValueError("Duplicate pair identities in a frame")
        geometry, error = None, None
        if matches:
            try:
                geometry = get_pair_geometry(atoms, *matches[0], scale=scale)
            except ValueError as exc:
                error = str(exc)
        result.append({"time_ps": float(time), "pair": key, "present": bool(matches),
                       "geometry": geometry, "error": error})
    return result


def _times(times_ps, length, *, uniform=False):
    times = np.asarray(times_ps, dtype=float)
    if times.shape != (length,) or length < 2 or not np.isfinite(times).all():
        raise ValueError("Supply at least two finite timestamps, one per frame")
    steps = np.diff(times)
    if np.any(steps <= 0):
        raise ValueError("Timestamps must be strictly increasing")
    if uniform and not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-12):
        raise ValueError("Lag statistics require uniformly spaced timestamps")
    return times


def _series(values, times_ps):
    """Validate vectors of shape (T,N,3) and uniform timestamps; return vectors
    and lag[k]=k*(times_ps[1]-times_ps[0]), k=0,...,T-1, in ps.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite (frames, selected_objects, 3) array")
    times = _times(times_ps, len(values), uniform=True)
    return values, np.arange(len(values)) * (times[1] - times[0])


def get_molecular_msd(com_positions, times_ps, *, unwrapped, selection):
    """Time-origin/ensemble averaged MSD of selected molecular COMs, Angstrom².

    Positions must be temporally unwrapped in a declared lab or deformation-
    corrected frame. No rigid alignment or drift subtraction is applied.
    ``selection`` is a required label identifying the species/group measured.

    For positions R[t,i,:] of T frames and N molecules,
    MSD[k] = sum_{t=0}^{T-k-1} sum_i |R[t+k,i]-R[t,i]|^2 / ((T-k) N).
    lag[k]=k*(times_ps[1]-times_ps[0]); timestamps must be uniform.
    MSD[0]=0, and n_origins[k]=T-k. The average includes all selected molecules.
    """
    if unwrapped is not True:
        raise ValueError("MSD requires explicitly temporally unwrapped positions")
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError("Provide a species/group selection label")
    positions, lag = _series(com_positions, times_ps)
    msd = np.zeros(len(lag))
    for k in range(1, len(lag)):
        msd[k] = np.mean(np.sum((positions[k:] - positions[:-k]) ** 2, axis=-1))
    return {"lag_ps": lag, "msd_angstrom2": msd, "selection": selection,
            "n_objects": positions.shape[1], "n_origins": len(lag) - np.arange(len(lag))}


def get_atomic_msd(positions, times_ps, *, unwrapped, selection):
    """Atomic ensemble MSD; same units/conventions as get_molecular_msd.

    MSD[k]=mean_{t,i} |positions[t+k,i]-positions[t,i]|^2 over atoms i
    and all T-k origins; lag[k]=k*dt. Uses get_molecular_msd with atomic rows.
    """
    return get_molecular_msd(positions, times_ps, unwrapped=unwrapped, selection=selection)


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
    affine box-motion subtraction is added.
    """
    wrapped = np.asarray(wrapped_com, dtype=float)
    lengths = np.asarray(box_lengths, dtype=float)
    if (wrapped.ndim != 3 or wrapped.shape[-1] != 3 or len(wrapped) < 1
            or lengths.shape != (len(wrapped), 3) or not np.isfinite(wrapped).all()
            or not np.isfinite(lengths).all() or np.any(lengths <= 0)):
        raise ValueError("Expected finite COMs (T, N, 3) and positive lengths (T, 3)")
    if np.any(abs(wrapped) > lengths[:, None, :] / 2 + 1e-8):
        raise ValueError("COMs must be wrapped into origin-centered boxes")
    delta = np.diff(wrapped, axis=0)
    box = lengths[1:, None, :]
    delta -= box * np.floor(delta / box + .5)
    return np.concatenate((wrapped[:1], wrapped[:1] + np.cumsum(delta, axis=0)), axis=0)


def get_molecular_diffusion_coefficient(msd_result, *, fit_start_ps, fit_end_ps):
    """Fit 3D ensemble MSD = intercept + 6 D t in an explicit lag interval.

    Returns D in Angstrom²/ps and cm²/s plus slope, intercept, and fit residual.
    This fit does not establish diffusive behavior or statistical uncertainty.
    It is not appropriate for confined/lower-dimensional diffusion.

    Select inclusive lags fit_start_ps <= t <= fit_end_ps. Unweighted
    least squares minimizes sum_j (MSD_j-a-b*t_j)^2 with a free intercept;
    D=b/6, and D[cm^2/s]=1e-4*D[Angstrom^2/ps]. residual_rms is
    sqrt(mean_j (MSD_j-a-b*t_j)^2). Negative slopes are reported, not clipped.
    """
    if not (np.isfinite(fit_start_ps) and np.isfinite(fit_end_ps) and 0 <= fit_start_ps < fit_end_ps):
        raise ValueError("Specify a finite increasing nonnegative fit interval")
    lag = np.asarray(msd_result["lag_ps"])
    values = np.asarray(msd_result["msd_angstrom2"])
    if values.shape != lag.shape or not np.isfinite(values).all():
        raise ValueError("Invalid MSD values")
    _times(lag, len(lag))
    mask = (lag >= fit_start_ps) & (lag <= fit_end_ps)
    if mask.sum() < 3:
        raise ValueError("Fit interval must contain at least three lag samples")
    slope, intercept = np.polyfit(lag[mask], values[mask], 1)
    return {"selection": msd_result["selection"], "diffusion_angstrom2_per_ps": float(slope / 6),
            "diffusion_cm2_per_s": float(slope / 6 * 1e-4),
            "slope_angstrom2_per_ps": float(slope), "intercept_angstrom2": float(intercept),
            "fit_interval_ps": (float(lag[mask][0]), float(lag[mask][-1])),
            "residual_rms_angstrom2": float(np.sqrt(np.mean((values[mask] - (slope * lag[mask] + intercept)) ** 2)))}


def _vector_correlation(values, times_ps, *, selection, subtract_mean=False, normalize=False):
    """For vectors x[t,i] with T frames and N objects, compute
    C[k]=sum_{t=0}^{T-k-1} sum_i x[t,i] dot x[t+k,i] / ((T-k) N).
    subtract_mean removes each object's temporal vector mean first.
    normalize returns C[k]/C[0]; lags are k*(times_ps[1]-times_ps[0]).
    """
    if not isinstance(selection, str) or not selection.strip():
        raise ValueError("Provide a species/group selection label")
    vectors, lag = _series(values, times_ps)
    if subtract_mean:
        vectors = vectors - vectors.mean(axis=0, keepdims=True)
    corr = np.array([np.mean(np.sum(vectors[:len(vectors)-k] * vectors[k:], axis=-1))
                     for k in range(len(vectors))])
    if normalize:
        if corr[0] <= 0:
            raise ValueError("Cannot normalize a zero-variance correlation")
        corr /= corr[0]
    return {"lag_ps": lag, "correlation": corr, "selection": selection,
            "normalized": normalize, "subtract_mean": subtract_mean,
            "n_origins": len(lag) - np.arange(len(lag)), "n_objects": vectors.shape[1]}


def get_molecular_velocity_autocorrelation(velocities, times_ps, *, selection, subtract_mean=False, normalize=False):
    """Dot-product COM VACF, averaged over molecules/time origins, (Angstrom/ps)².

    Input velocities must be Angstrom/ps. Mean subtraction, if requested, is
    the temporal mean of each molecule. Normalized output is dimensionless.

    C_v[k]=mean_{t,i} V[t,i] dot V[t+k,i], over T-k origins and all
    selected molecules. If subtract_mean, replace each V[t,i] by
    V[t,i]-mean_t V[t,i]. If normalize, divide by C_v[0]. Lag=k*dt.
    """
    return _vector_correlation(velocities, times_ps, selection=selection,
                               subtract_mean=subtract_mean, normalize=normalize)


def get_atomic_velocity_autocorrelation(velocities, times_ps, **kwargs):
    """Atomic VACF with the same conventions as the molecular COM VACF.

    C_v[k]=mean_{t,i} v[t,i] dot v[t+k,i] over atoms and T-k origins.
    kwargs control per-atom temporal-mean subtraction and division by C_v[0].
    """
    return get_molecular_velocity_autocorrelation(velocities, times_ps, **kwargs)


def get_atomic_force_autocorrelation(forces, times_ps, *, selection, subtract_mean=False, normalize=False):
    """Atomic force dot-product correlation, (eV/Angstrom)² or dimensionless.

    C_F[k]=mean_{t,i} F[t,i] dot F[t+k,i], over atoms and T-k origins.
    subtract_mean removes each atom's temporal vector mean; normalize divides
    by C_F[0]. Input forces must be eV/Angstrom; lag=k*dt in ps.
    """
    return _vector_correlation(forces, times_ps, selection=selection,
                               subtract_mean=subtract_mean, normalize=normalize)


def get_molecular_force_autocorrelation(net_forces, times_ps, **kwargs):
    """Correlation of molecular resultant forces, not individual atomic forces.

    Use F_mol[t,i]=sum_{a in molecule i} F_atom[t,a] as net_forces;
    C[k]=mean_{t,i} F_mol[t,i] dot F_mol[t+k,i]. This includes cross terms
    between atoms within each molecule. kwargs follow the atomic correlation.
    """
    return get_atomic_force_autocorrelation(net_forces, times_ps, **kwargs)


def pair_identity(pair):
    """Canonical unordered pair of molecule atom-ID tuples; independent of rows."""
    if len(pair) != 2:
        raise ValueError("A pair must contain two molecule ID collections")
    first, second = (tuple(sorted(group)) for group in pair)
    if (not first or not second or len(set(first)) != len(first)
            or len(set(second)) != len(second) or set(first) & set(second)):
        raise ValueError("Pair molecules need nonempty, distinct, disjoint atom IDs")
    return tuple(sorted((first, second)))


@dataclass(frozen=True)
class PairLifetime:
    pair: tuple
    first_frame: int
    last_frame: int
    first_time_ps: float
    last_time_ps: float
    observed_span_ps: float
    detected_frames: int
    left_censored: bool
    right_censored: bool


def get_pair_lifetime(pairs_by_frame, times_ps, *, max_gap_frames=0):
    """Track encounter episodes from per-frame lists of molecule-ID pairs.

    Directly accepts pair lists from get_all_prereactive_pairs, or filtered lists.
    Use one fixed selection criterion throughout. Molecule IDs here are atom-ID
    sets, not per-frame candidate numbers. Pair order/atom order is immaterial.

    Default episodes are continuous detections. max_gap_frames bridges that many
    missing samples (intermittent definition). Span = last detection minus first
    detection: a single-sample event has span zero, not a known zero lifetime.
    Entry/exit occur between samples and are not interpolated. Boundary episodes
    are censored; a bridged episode still awaiting termination at the final frame
    is right-censored. Do not average these spans as unbiased physical lifetimes.
    The iterable of pair lists is consumed once; trajectory coordinates are not
    retained. Timestamps may be irregular. Returned frame indices are local to
    the supplied trajectory/window. Missing frames are not absent detections.

    observed_span_ps=times_ps[last_frame]-times_ps[first_frame]. Two
    detections at indices i<j remain in one encounter when j-i-1 <=
    max_gap_frames; detected_frames counts observations, not bridged gaps.
    """
    if not isinstance(max_gap_frames, (int, np.integer)) or max_gap_frames < 0:
        raise ValueError("max_gap_frames must be a nonnegative integer")
    times = np.asarray(times_ps, dtype=float)
    if times.ndim != 1:
        raise ValueError("Timestamps must be one-dimensional")
    _times(times, len(times))
    active, result = {}, []

    def finish(key, state, censored):
        start, end, count = state
        result.append(PairLifetime(key, start, end, float(times[start]), float(times[end]),
                                   float(times[end] - times[start]), count, start == 0, censored))

    sentinel = object()
    for index, (pairs, time) in enumerate(zip_longest(pairs_by_frame, times, fillvalue=sentinel)):
        if pairs is sentinel or time is sentinel:
            raise ValueError("One pair list is required for every timestamp")
        current = {pair_identity(pair) for pair in pairs}
        for key in list(active):
            # On a detected frame the gap excludes this frame; on an absent
            # frame it includes it. Close only when the allowed gap is exceeded.
            gap = index - active[key][1] - (1 if key in current else 0)
            if gap > max_gap_frames:
                finish(key, active.pop(key), False)
        for key in sorted(current):
            if key in active:
                active[key][1] = index
                active[key][2] += 1
            else:
                active[key] = [index, index, 1]
    for key, state in active.items():
        finish(key, state, True)
    return sorted(result, key=lambda episode: (episode.first_frame, episode.pair))
