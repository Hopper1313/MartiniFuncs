"""Indexed, bounded process scheduling for independent trajectory frames.

Workers seek contiguous byte ranges in an uncompressed text dump. No temporary
per-frame files are created. At most twice the worker count of chunks is queued;
results are consumed in source order, including errors. Temporal reductions such
as TOR remain in the parent and therefore include every chunk boundary.
"""
from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .lammpsio import iter_lammps_frames


class FrameSummary(NamedTuple):
    """Per-frame molecular summary returned by prepare_frame (fixed molecular order).

    com: whole-molecule COMs, Angstrom (any periodic image). cell: row-vector
    cell (3, 3); pbc: periodic flags; origin: cell origin (LAMMPS lo corner).
    pairs/shifts: retained-frame pair candidates or None. scalars: density,
    volume and optional temperature. velocity: COM velocity Angstrom/ps or None.
    force: net molecular force eV/Angstrom or None.
    """
    com: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    origin: np.ndarray
    pairs: list | None
    shifts: list | None
    scalars: dict
    velocity: np.ndarray | None
    force: np.ndarray | None


@dataclass
class PreparedFrame:
    """Small process payload: source identity and computed molecular summaries."""
    index: int
    timestep: int
    prepared: FrameSummary
    atoms: None = None


def _require_equal(actual, expected, label, frame, *, approximate=False):
    """Raise ValueError (not AssertionError) when a fixed-identity check fails."""
    same = (np.allclose(actual, expected, rtol=1e-7, atol=0) if approximate
            else np.array_equal(actual, expected))
    if np.shape(actual) != np.shape(expected) or not same:
        raise ValueError(f"Frame {frame}: atom {label} differ from the first frame; "
                         "split the trajectory or rerun with matching files")


_SCAN_BLOCK = 1 << 26   # bytes per frame_offsets read


def frame_offsets(path):
    """Return uint64 frame starts plus EOF; scan bytes without parsing atoms.

    Memory is eight bytes per frame boundary. Offsets refer to the original
    uncompressed file, so workers do not copy the trajectory or scan its prefix.
    """
    marker = b'ITEM: TIMESTEP'
    pattern = b'\n' + marker
    offsets = array('Q')
    with open(path, 'rb') as handle:
        head = handle.read(1 << 16)
        body = head.lstrip(b'\r\n')
        if not body.strip():
            raise ValueError(f'{path}: no frames')
        if not (body.startswith(marker) and body[len(marker):len(marker)+1] in (b'\n', b'\r', b'')):
            raise ValueError(f'{path}: expected ITEM: TIMESTEP')
        if len(body) == len(head):
            offsets.append(0)   # later markers (and one after blank lines) follow a newline
        handle.seek(0)
        # bytes.find over 64 MiB blocks (C speed). A match starting at i is only
        # accepted once the byte after the marker is available; later starts are
        # carried into the next block, so no boundary match is lost or repeated.
        base, data = 0, b''
        while True:
            block = handle.read(_SCAN_BLOCK)
            eof = not block
            data += block
            limit = len(data) if eof else len(data) - len(pattern)
            hit = data.find(pattern)
            while 0 <= hit < limit:
                if data[hit+len(pattern):hit+len(pattern)+1] in (b'\n', b'\r', b''):
                    offsets.append(base + hit + 1)
                hit = data.find(pattern, hit + 1)
            if eof:
                break
            base, data = base + limit, data[limit:]
        offsets.append(base + len(data))
    if len(offsets) < 2:
        raise ValueError(f'{path}: no frames')
    return np.asarray(offsets, dtype=np.uint64)


def ordered_tasks(function, tasks, workers, initializer=None, initargs=()):
    """Yield ordered results with bounded submission and spawned CPU workers."""
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers == 1:
        if initializer:
            initializer(*initargs)
        for task in tasks:
            yield function(task)
        return
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initializer, initargs=initargs) as pool:
        tasks = iter(tasks)
        pending = deque()
        try:
            for _ in range(2 * workers):
                task = next(tasks, None)
                if task is None:
                    break
                pending.append(pool.submit(function, task))
            while pending:
                yield pending.popleft().result()
                task = next(tasks, None)
                if task is not None:
                    pending.append(pool.submit(function, task))
        finally:
            for future in pending:
                future.cancel()


def prepare_frame(frame, context):
    """Validate and reconstruct one frame with shared first-frame topology.

    Whole molecular coordinates follow parent-child minimum-image displacements.
    COM is sum(m_i*r_i)/sum(m_i), COM velocity is sum(m_i*v_i)/sum(m_i), and
    net molecular force is sum(F_i). Velocities convert from ASE units to A/ps;
    source forces remain eV/A. Density and temperature delegate to md_statistics.
    This is spatial reconstruction only; temporal unwrapping is deliberately
    left to the ordered consumer.
    """
    from ase.geometry import find_mic
    from ase import units as ase_units
    from . import md_statistics as md
    from .reaction_detection import find_possible_products, PossibleReactionError
    atoms = frame.atoms
    # analyze_run screens the first frame before building its topology.
    if context['reaction_atom_counts'] is not None and frame.index != context.get('prescreened_frame'):
        hits = find_possible_products(atoms, reactant_atom_counts=context['reaction_atom_counts'],
                                      bond_scale=context['bond_scale'])
        if hits:
            raise PossibleReactionError([dict(frame=frame.index, timestep=frame.timestep, **h) for h in hits])
    _require_equal(atoms.arrays['global_index'], context['ids'], 'IDs', frame.index)
    _require_equal(atoms.numbers, context['numbers'], 'elements', frame.index)
    _require_equal(atoms.get_masses(), context['masses'], 'masses', frame.index, approximate=True)
    if context['has_velocities'] != ('momenta' in atoms.arrays) or context['has_forces'] != ('lammps_forces' in atoms.arrays):
        raise ValueError('Velocity/force availability changes within the dump')
    length = atoms.cell.array.diagonal()
    orthogonal = atoms.pbc.all() and np.allclose(atoms.cell.array, np.diag(length), atol=1e-10) and np.all(length > 0)
    whole = atoms.positions.copy()
    for level in context['levels']:
        parent, child = level.T
        delta = atoms.positions[child] - atoms.positions[parent]
        if orthogonal:
            delta -= length * np.floor(delta / length + .5)
        else:
            delta = find_mic(delta, atoms.cell, atoms.pbc)[0]
        whole[child] = whole[parent] + delta
    edges = context['edges']
    if len(edges) and np.any(np.linalg.norm(whole[edges[:, 1]] - whole[edges[:, 0]], axis=1) > 1.25 * context['cutoffs']):
        raise ValueError(f'Frame {frame.index}: fixed molecular topology no longer reconstructs consistently')
    com = np.column_stack([np.bincount(context['groups'], weights=context['masses'] * whole[:, axis]) /
                           context['molecular_masses'] for axis in range(3)])
    pairs = shifts = None
    if context['compute_pairs'] and frame.index % context['frame_stride'] == 0:
        if context['topology'] is not None:
            pairs, shifts, _ = context['topology'].evaluate(whole, atoms.cell.array, atoms.pbc,
                                                         criteria=context['pair_criteria'])
        else:
            from .geom_utils import get_all_prereactive_pairs
            pairs, shifts = get_all_prereactive_pairs(atoms, scale=context['bond_scale'], criteria=context['pair_criteria'])
    scalars = dict(density_g_cm3=md.get_frame_density(atoms), volume_angstrom3=atoms.get_volume())
    velocity = force = None
    if context['has_velocities']:
        if context['degrees_of_freedom'] is not None:
            scalars['temperature_K'] = md.get_frame_temperature(atoms, degrees_of_freedom=context['degrees_of_freedom'])
        v = atoms.get_velocities()*1000*ase_units.fs
        velocity = np.column_stack([np.bincount(context['groups'], weights=context['masses']*v[:, axis]) /
                                    context['molecular_masses'] for axis in range(3)])
    if context['has_forces']:
        f = atoms.arrays['lammps_forces']
        force = np.column_stack([np.bincount(context['groups'], weights=f[:, axis]) for axis in range(3)])
    return FrameSummary(com, atoms.cell.array.copy(), atoms.pbc.copy(), np.asarray(atoms.get_celldisp()).ravel().copy(),
                        pairs, shifts, scalars, velocity, force)


def _initialize_analysis(path, options, context):
    """Install immutable source and topology once per process."""
    global _analysis_source
    _analysis_source = path, options, context


def _analysis_chunk(task):
    """Parse and prepare a contiguous range; return bounded ordered frames."""
    path, options, context = _analysis_source
    start, lower, upper = task
    result = []
    for frame in iter_lammps_frames(path, byte_range=(lower, upper), frame_offset=start, **options):
        # Send only molecular summaries; full atomic frames stay in the worker.
        result.append(PreparedFrame(frame.index, frame.timestep, prepare_frame(frame, context)))
    return result


def analysis_frames(path, options, context, *, workers, batch_size, max_frames, progress):
    """Yield frame zero for topology setup, then prepare indexed chunks in parallel.

    The caller fills context after receiving frame zero. This preserves exactly
    the serial first-frame molecular identities in every worker.
    """
    initial = iter_lammps_frames(path, **options)
    try:
        first = next(initial, None)
        if first is None:
            return
        yield first
    finally:
        initial.close()
    if progress:
        progress('Indexing source frame byte offsets for parallel analysis...')
    offsets = frame_offsets(path)
    stop = len(offsets)-1 if max_frames is None else min(max_frames, len(offsets)-1)
    tasks = ((start, int(offsets[start]), int(offsets[min(start+batch_size, stop)]))
             for start in range(1, stop, batch_size))
    if progress:
        progress(f'Parallel frame analysis: {stop} source frames, {workers} workers, {batch_size} frames/chunk')
    yield from (frame for chunk in ordered_tasks(_analysis_chunk, tasks, workers,
                _initialize_analysis, (str(Path(path).resolve()), options, context)) for frame in chunk)
