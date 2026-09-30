"""Audit a metal-unit, orthorhombic dump independently of the package reader.

Run from the repository root with ``python -m tools.verify_molecular_tracking
RUN_DIR --output REPORT_DIR --timestep-ps 0.0005``. No UMA model is needed.
The raw reader deliberately supports only the x/y/z + ix/iy/iz dump convention.
It fails for tilted/nonperiodic cells. Outputs include independently reconstructed
COM trajectories and a JSON comparison with analyze_run. This checks tracking,
not equilibration, chemical identity throughout the run, or diffusion convergence.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from scipy.sparse.csgraph import connected_components

from MartiniFuncs.trajectory_analysis import analyze_run, resolve_run_files


def raw_frames(path):
    """Read and ID-sort raw dump fields without ASE or MartiniFuncs parsing."""
    with path.open() as stream:
        while line := stream.readline():
            assert line.strip() == 'ITEM: TIMESTEP'
            step = int(stream.readline())
            assert stream.readline().strip() == 'ITEM: NUMBER OF ATOMS'
            count = int(stream.readline())
            header = stream.readline().split()
            assert header[-3:] == ['pp', 'pp', 'pp']
            bounds = np.array([list(map(float, stream.readline().split())) for _ in range(3)])
            assert bounds.shape[1] == 2 or np.all(bounds[:, 2] == 0), 'Tilted cell unsupported'
            columns = stream.readline().split()[2:]
            rows = np.array([stream.readline().split() for _ in range(count)])
            rows = rows[np.argsort(rows[:, columns.index('id')].astype(int))]
            def get(names, dtype=float):
                return rows[:, [columns.index(n) for n in names.split()]].astype(dtype)
            yield (step, bounds[:, :2], get('id', int).ravel(), get('element', str).ravel(),
                   get('mass').ravel(), get('x y z'), get('ix iy iz', int))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timestep-ps', required=True, type=float)
    args = parser.parse_args()
    files = resolve_run_files(args.run_dir)
    print('Computing package trajectories (pairs disabled)...', flush=True)
    result = analyze_run(args.run_dir, units='metal', timestep_ps=args.timestep_ps,
                         compute_pairs=False, progress=lambda s: print(s, flush=True))
    package_groups = {frozenset(int(i)+1 for i in group): n
                      for n, group in enumerate(result['molecule_atom_ids'])}
    steps, boxes, centers, wrapped, image_com = [], [], [], [], []
    max_geometry_error = 0.
    max_span_fraction = 0.
    graph_checks = []
    for frame, (step, bounds, ids, elements, mass, xyz, images) in enumerate(raw_frames(files['dump'])):
        length = bounds[:, 1]-bounds[:, 0]
        center = bounds.mean(axis=1)
        if frame == 0:
            first_ids, first_elements, first_mass = ids.copy(), elements.copy(), mass.copy()
        np.testing.assert_array_equal(ids, first_ids)
        np.testing.assert_array_equal(elements, first_elements)
        np.testing.assert_array_equal(mass, first_mass)
        if frame % 500 == 0:
            # Independent all-pairs graph, using the same declared radius scale.
            delta = xyz[:, None, :]-xyz[None, :, :]
            delta -= np.rint(delta/length)*length
            radii = covalent_radii[[atomic_numbers[e] for e in elements]]
            adjacency = np.linalg.norm(delta, axis=-1) < 1.3*(radii[:, None]+radii[None, :])
            count, labels = connected_components(adjacency, directed=False)
            memberships = {frozenset(ids[labels == k]) for k in range(count)}
            assert memberships == set(package_groups), f'Molecule membership mismatch at frame {frame}'
            graph_checks.append(frame)
        if frame == 0:
            # Align independent graph labels to the package's persistent ID groups.
            groups = np.array([package_groups[frozenset(ids[labels == k])] for k in labels])
            roots = np.array([np.flatnonzero(groups == k)[0] for k in range(count)])
            molecular_mass = np.bincount(groups, weights=mass)
        # Independent spatial reconstruction relative to ONE atom, not bond traversal.
        relative = xyz-xyz[roots[groups]]
        relative -= np.rint(relative/length)*length
        whole = xyz[roots[groups]]+relative
        max_span_fraction = max(max_span_fraction, float(np.max(np.abs(relative/length))))
        if frame == 0:
            # Restart image flags may differ by fixed atomic lattice offsets.
            image_offset = np.rint((whole-xyz)/length).astype(int)-images
        image_whole = xyz+(images+image_offset)*length
        relative_image = image_whole-image_whole[roots[groups]]
        max_geometry_error = max(max_geometry_error, float(np.max(np.abs(relative_image-relative))))
        def com(positions):
            return np.column_stack([np.bincount(groups, weights=mass*positions[:, a])/molecular_mass
                                    for a in range(3)])
        w = com(whole)-center
        w -= np.rint(w/length)*length
        steps.append(step)
        boxes.append(length)
        centers.append(center)
        wrapped.append(w)
        image_com.append(com(image_whole))
        if frame % 500 == 0:
            print(f'Independent raw reconstruction: frame {frame}', flush=True)
    steps, boxes, centers, wrapped, image_com = map(np.asarray, (steps, boxes, centers, wrapped, image_com))
    assert result['com_angstrom'] is not None, result['notices']
    np.testing.assert_allclose(centers, np.broadcast_to(centers[0], centers.shape), atol=1e-7, rtol=0)
    delta = np.diff(wrapped, axis=0)
    crossing = np.rint(delta/boxes[1:, None, :]).astype(int)
    increments = delta-crossing*boxes[1:, None, :]
    tor = np.concatenate([wrapped[:1], wrapped[:1]+np.cumsum(increments, axis=0)])
    # Independent temporal check using accumulated LAMMPS atomic image flags.
    # q_t = w_t + I_t L_t. TOR increment = delta(q) - I_(t-1)*delta(L).
    q = image_com-centers[:, None, :]
    branches = np.rint((q-wrapped)/boxes[:, None, :]).astype(int)
    image_increments = np.diff(q, axis=0)-branches[:-1]*np.diff(boxes, axis=0)[:, None, :]
    image_tor = np.concatenate([wrapped[:1], wrapped[:1]+np.cumsum(image_increments, axis=0)])
    rows = result['frames']
    np.testing.assert_array_equal(steps, [r['timestep'] for r in rows])
    times = steps*args.timestep_ps
    np.testing.assert_allclose(times, [r['time_ps'] for r in rows], atol=1e-10, rtol=0)
    thermo_errors = []
    for path in sorted(args.run_dir.glob('*.out')):
        header = None
        for line in path.read_text(errors='replace').splitlines():
            fields = line.split()
            if fields and fields[0] == 'Step':
                header = fields
            elif header and 'Time' in header and len(fields) == len(header):
                try:
                    step = int(fields[0])
                    time = float(fields[header.index('Time')])
                except ValueError:
                    continue
                if step in steps:
                    thermo_errors.append(abs(time-step*args.timestep_ps))
    summary = dict(source=str(files['dump']), frames=len(steps), atoms=len(ids),
        species_counts=dict(Counter(result['species'])), graph_checked_frames=graph_checks,
        timestep_range=[int(steps[0]), int(steps[-1])], time_range_ps=[float(times[0]), float(times[-1])],
        saved_step_spacings=np.unique(np.diff(steps)).tolist(), saved_time_spacings_ps=np.unique(np.diff(times)).tolist(),
        thermo_samples_checked=len(thermo_errors), thermo_max_time_error_ps=max(thermo_errors, default=None),
        max_image_vs_anchor_geometry_error_angstrom=max_geometry_error,
        max_anchor_component_fraction_of_box=max_span_fraction,
        max_wrapped_com_error_angstrom=float(np.max(np.abs(wrapped-result['wrapped_com_angstrom']))),
        max_tor_com_error_angstrom=float(np.max(np.abs(tor-result['com_angstrom']))),
        max_image_tor_error_angstrom=float(np.max(np.abs(image_tor-tor))),
        max_image_branch_residual_angstrom=float(np.max(np.abs(q-wrapped-branches*boxes[:, None, :]))),
        com_boundary_crossings=int(np.count_nonzero(crossing)),
        max_saved_com_increment_angstrom=float(np.linalg.norm(increments, axis=-1).max()),
        max_saved_com_increment_box_fraction=float(np.abs(increments/boxes[1:, None, :]).max()))
    # Independently reproduce the 75--100 ps fit discussed in the notebooks.
    # This is a numerical cross-check, not a claim of diffusive behavior.
    spacing = np.diff(times)
    if np.allclose(spacing, spacing[0]) and times[-1]-times[0] >= 100:
        lag = np.arange(min(len(times), int(np.floor(100/spacing[0]))+1))*spacing[0]
        mask = (lag >= 75) & (lag <= 100)
        if mask.sum() >= 3:
            summary['independent_75_100_ps_fits'] = {}
            for species in sorted(set(result['species'])):
                coordinates = tor[:, result['species'] == species]
                msd = np.array([0.] + [np.mean(np.sum((coordinates[k:]-coordinates[:-k])**2, axis=-1))
                                      for k in range(1, len(lag))])
                slope, intercept = np.polyfit(lag[mask], msd[mask], 1)
                summary['independent_75_100_ps_fits'][species] = dict(
                    diffusion_cm2_per_s=float(slope/6*1e-4), intercept_angstrom2=float(intercept),
                    residual_rms_angstrom2=float(np.sqrt(np.mean((msd[mask]-slope*lag[mask]-intercept)**2))))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/'com_comparison.npz', time_ps=times, timestep=steps,
        species=result['species'], package_com=result['com_angstrom'], independent_tor_com=tor,
        image_tor_com=image_tor, image_com=image_com, wrapped_com=wrapped, box_lengths=boxes,
        atom_ids=ids, atom_molecule_index=groups)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    assert max_geometry_error < 1e-7, 'Image flags disagree with spatial reconstruction'
    for key in ['max_wrapped_com_error_angstrom', 'max_tor_com_error_angstrom',
                'max_image_tor_error_angstrom', 'max_image_branch_residual_angstrom']:
        assert summary[key] < 1e-7, (key, summary[key])
    assert thermo_errors and max(thermo_errors) < 1e-8, 'No matching thermo clock or clock mismatch'


if __name__ == '__main__':
    main()
