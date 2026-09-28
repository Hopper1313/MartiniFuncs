"""Opt-in integration check: python -m tests.validate_npt_statistics.

Streams the full NPT fixture. Writes compact reports; does not load UMA or
change trajectory data. The long trajectory is not part of default unit tests.
"""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import time

import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs import md_statistics as md
from MartiniFuncs import graph
from MartiniFuncs.geom_utils import get_all_prereactive_pairs
from MartiniFuncs.lammpsio import iter_lammps_frames, read_lammps_input_settings


def check_episode_counts(rows, episodes):
    """Independent reconstruction for continuous episodes (no bridged gaps)."""
    counts = np.zeros(len(rows), dtype=int)
    for event in episodes:
        first, last = event["first_frame"], event["last_frame"]
        assert event["detected_frames"] == last - first + 1
        assert_allclose(event["observed_span_ps"], rows[last]["time_ps"] - rows[first]["time_ps"])
        counts[first:last+1] += 1
    for row, count in zip(rows, counts):
        if not row["pair_error"]:
            assert count == int(row["candidate_pairs"]), f"Episode count mismatch at frame {row['frame']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixtures = Path(__file__).parent / "lammpsio_tests"
    parser.add_argument("--dump", type=Path, default=fixtures / "equilibrate_2p5.lammpstrj")
    parser.add_argument("--input", type=Path, default=fixtures / "infile.in")
    parser.add_argument("--output", type=Path, default=Path("reports/npt_statistics"))
    args = parser.parse_args()
    settings = read_lammps_input_settings(args.input)
    if settings["units"] != "metal":
        raise ValueError("This reference check assumes metal dump units")
    start = time.perf_counter()
    rows, pairs_by_frame, failures = [], [], []
    molecule_velocities = {"CPD": [], "dienophile": []}
    molecule_forces = {"CPD": [], "dienophile": []}
    atomic_positions, atomic_velocities, atomic_forces = [], [], []
    max_errors = dict(temperature_K=0., density_g_cm3=0., velocity_angstrom_ps=0.,
                      force_magnitude_eV_angstrom=0.)
    first = None
    for frame in iter_lammps_frames(args.dump, units="metal"):
        atoms, raw = frame.atoms, frame.columns
        if first is None:
            first = atoms.copy()
            expected_ids = atoms.arrays["global_index"].copy()
            first_graph = graph.build_bond_graph(atoms, scale=1.3)
            components = graph.connected_components(first_graph, atom_ids=expected_ids)
            components_by_species = {"CPD": [c for c in components if len(c) == 11],
                                     "dienophile": [c for c in components if len(c) == 23]}
            if sum(map(len, components_by_species["CPD"] + components_by_species["dienophile"])) != len(atoms):
                raise ValueError("Initial frame is not fully assigned to CPD/dienophile")
            molecule_rows = {name: [np.array([np.flatnonzero(expected_ids == i)[0] for i in c])
                                    for c in comps] for name, comps in components_by_species.items()}
        np.testing.assert_array_equal(atoms.arrays["global_index"], expected_ids)
        np.testing.assert_array_equal(atoms.numbers, first.numbers)
        assert_allclose(atoms.get_masses(), first.get_masses())
        velocities = np.column_stack([raw[name] for name in ("vx", "vy", "vz")])
        forces = np.column_stack([raw[name] for name in ("fx", "fy", "fz")])
        # Independent SI reference: metal velocity Angstrom/ps = 100 m/s.
        kinetic_joule = .5 * np.sum(raw["mass"][:, None] * 1.66053906660e-27 * (velocities * 100) ** 2)
        dof = 3 * len(atoms) - 3  # unconstrained system, standard COM DOF correction
        temperature_ref = 2 * kinetic_joule / (dof * 1.380649e-23)
        temperature = md.get_frame_temperature(atoms, degrees_of_freedom=dof)
        assert_allclose(temperature, temperature_ref, rtol=2e-6)
        kinetic_ev = md.get_frame_kinetic_energy(atoms)
        assert_allclose(kinetic_ev, kinetic_joule / 1.602176634e-19, rtol=2e-6)
        density = md.get_frame_density(atoms)
        density_ref = raw["mass"].sum() * 1.66053906660 / np.linalg.det(atoms.cell.array)
        assert_allclose(density, density_ref, rtol=2e-6)
        magnitudes = np.linalg.norm(forces, axis=1)
        assert_allclose(magnitudes, raw["v_netforce"], rtol=2e-5, atol=2e-5)
        stats = md.get_frame_force_statistics(atoms)
        assert_allclose(stats["net_force"], forces.sum(axis=0))
        assert_allclose(stats["rms_magnitude"], np.sqrt(np.mean(magnitudes ** 2)))
        max_errors["temperature_K"] = max(max_errors["temperature_K"], abs(temperature-temperature_ref))
        max_errors["density_g_cm3"] = max(max_errors["density_g_cm3"], abs(density-density_ref))
        max_errors["force_magnitude_eV_angstrom"] = max(max_errors["force_magnitude_eV_angstrom"], float(np.max(abs(magnitudes-raw["v_netforce"]))))
        for name, comps in components_by_species.items():
            vs, fs = [], []
            for comp, indices in zip(comps, molecule_rows[name]):
                velocity = md.get_molecule_com_velocity(atoms, comp)
                reference = np.average(velocities[indices], axis=0, weights=raw["mass"][indices])
                assert_allclose(velocity, reference, rtol=1e-10, atol=1e-10)
                max_errors["velocity_angstrom_ps"] = max(max_errors["velocity_angstrom_ps"], float(np.max(abs(velocity-reference))))
                force = md.get_molecule_net_force(atoms, comp)
                assert_allclose(force, forces[indices].sum(axis=0))
                vs.append(velocity)
                fs.append(force)
            molecule_velocities[name].append(vs)
            molecule_forces[name].append(fs)
        try:
            pairs, shifts = get_all_prereactive_pairs(atoms, scale=1.3)
            pairs_by_frame.append(pairs)
            error = ""
        except ValueError as exc:
            pairs_by_frame.append(None)
            error = str(exc)
            failures.append({"frame": frame.index, "timestep": frame.timestep, "error": error})
        rows.append(dict(frame=frame.index, timestep=frame.timestep,
                         time_ps=frame.timestep * settings["timestep"], temperature_K=temperature,
                         kinetic_energy_eV=kinetic_ev, density_g_cm3=density, volume_angstrom3=atoms.get_volume(),
                         rms_force_eV_angstrom=stats["rms_magnitude"],
                         candidate_pairs=len(pairs) if not error else None, pair_error=error))
        # Bounded late-time window, first 16 persistent atoms: numerical MSD/correlation
        # check only. Image-flag unwrapping with a changing NPT cell is not a
        # validated diffusion convention; no physical diffusion coefficient reported.
        if len(atomic_positions) == 201:
            atomic_positions.pop(0)
            atomic_velocities.pop(0)
            atomic_forces.pop(0)
        atomic_positions.append((atoms.positions + atoms.arrays["image_flags"] @ atoms.cell.array)[:16].copy())
        atomic_velocities.append(velocities[:16].copy())
        atomic_forces.append(forces[:16].copy())
        if frame.index % 100 == 0:
            print(f"frame {frame.index}: {temperature:.2f} K, pair failures {len(failures)}, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    if len(rows) < 3:
        raise ValueError("Need at least three frames")
    times = np.array([r["time_ps"] for r in rows])
    correlations = {}
    for name in molecule_velocities:
        correlations[name] = {}
        for quantity, values, function in (
            ("velocity", molecule_velocities[name], md.get_molecular_velocity_autocorrelation),
            ("net_force", molecule_forces[name], md.get_molecular_force_autocorrelation),
        ):
            result = function(values, times, selection=name)
            values = np.asarray(values)
            for lag in (0, 1, len(times)//2, len(times)-1):
                reference = sum(float(np.dot(a, b)) for a, b in
                                zip(values[:len(times)-lag].reshape(-1, 3), values[lag:].reshape(-1, 3))) / ((len(times)-lag)*values.shape[1])
                assert_allclose(result["correlation"][lag], reference, atol=1e-9)
            correlations[name][quantity] = {"lag_ps": result["lag_ps"].tolist(), "correlation": result["correlation"].tolist()}
    window_times = times[-len(atomic_positions):]
    msd = md.get_atomic_msd(atomic_positions, window_times, unwrapped=True, selection="first 16 atom IDs; image-flag numerical check")
    positions = np.asarray(atomic_positions)
    assert_allclose(msd["msd_angstrom2"][-1], np.mean(np.sum((positions[-1]-positions[0])**2, axis=1)))
    for function, values in ((md.get_atomic_velocity_autocorrelation, atomic_velocities),
                              (md.get_atomic_force_autocorrelation, atomic_forces)):
        result = function(values, window_times, selection="first 16 atom IDs")
        assert_allclose(result["correlation"][0], np.mean(np.sum(np.asarray(values)**2, axis=-1)))
    # Never bridge a frame whose pair detection failed. Censor each valid segment.
    episodes = []
    segment_start = 0
    for stop in range(len(rows)+1):
        if stop == len(rows) or pairs_by_frame[stop] is None:
            if stop - segment_start == 1:
                # A one-frame valid window is censored at both ends.
                for pair in {md.pair_identity(p) for p in pairs_by_frame[segment_start]}:
                    episodes.append(asdict(md.PairLifetime(
                        pair, segment_start, segment_start, float(times[segment_start]),
                        float(times[segment_start]), 0., 1, True, True)))
            if stop-segment_start >= 2:
                for event in md.get_pair_lifetime(pairs_by_frame[segment_start:stop], times[segment_start:stop]):
                    data = asdict(event)
                    data["first_frame"] += segment_start
                    data["last_frame"] += segment_start
                    episodes.append(data)
            segment_start = stop+1
    check_episode_counts(rows, episodes)
    summary = dict(dump=str(args.dump.resolve()), input=str(args.input.resolve()), frames=len(rows), atoms=len(first),
                   time_range_ps=[float(times[0]), float(times[-1])], sample_interval_ps=float(times[1]-times[0]),
                   molecules={name:len(comps) for name,comps in components_by_species.items()},
                   temperature_dof=3*len(first)-3, raw_column_checks_passed=True,
                   max_reference_errors=max_errors, pair_detection_failures=failures, pair_episode_count=len(episodes),
                   completed_episode_count=sum(not e["left_censored"] and not e["right_censored"] for e in episodes),
                   pair_episode_count_reconstruction_passed=True,
                   single_sample_episodes=sum(e["detected_frames"] == 1 for e in episodes),
                   left_censored_episodes=sum(e["left_censored"] for e in episodes),
                   right_censored_episodes=sum(e["right_censored"] for e in episodes),
                   max_observed_pair_span_ps=max((e["observed_span_ps"] for e in episodes),default=0),
                   correlations=correlations, atomic_msd_window_ps=[float(window_times[0]),float(window_times[-1])],
                   atomic_msd_angstrom2=msd["msd_angstrom2"].tolist(),
                   diffusion_status="Not physically estimated: changing NPT cell and unvalidated unwrapping/fit regime",
                   elapsed_seconds=time.perf_counter()-start)
    for field in ("temperature_K", "density_g_cm3", "candidate_pairs"):
        vals = np.array([r[field] for r in rows if r[field] is not None])
        summary[field] = dict(mean=float(vals.mean()), min=float(vals.min()), max=float(vals.max())) if len(vals) else None
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "frames.csv").open("w",newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output / "pair_lifetimes.json").write_text(json.dumps(episodes,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("correlations","atomic_msd_angstrom2","pair_detection_failures")},indent=2),flush=True)
    print(f"Reports: {args.output.resolve()}",flush=True)


if __name__ == "__main__":
    main()
