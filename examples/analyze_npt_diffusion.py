"""Species COM diffusion estimate from the NPT fixture; run as a module.

python -m examples.analyze_npt_diffusion
Retains only COM trajectories; assumes fixed molecules and orthorhombic boxes.
"""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from ase.data import covalent_radii
from MartiniFuncs import graph, md_statistics as md
from MartiniFuncs.lammpsio import iter_lammps_frames, read_lammps_input_settings


def main():
    """Generate the historical fixture-specific COM diffusion report.

    For each fixed molecule R=sum(m_i*r_i)/sum(m_i), after bondwise
    minimum-image reconstruction d <- d-L*floor(d/L+1/2). Center/wrap R,
    then delegate temporal TOR unwrapping, MSD[k]=mean(|R[t+k]-R[t]|^2),
    and the free-intercept fit D=slope/6 to md_statistics. Units are
    Angstrom, ps, and Angstrom^2/ps. For new runs use trajectory_analysis.
    """
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests/lammpsio_tests"
    output = root / "reports/npt_statistics/diffusion"
    output.mkdir(parents=True, exist_ok=True)
    settings = read_lammps_input_settings(fixtures / "infile.in")
    if settings["units"] != "metal":
        raise ValueError("This analysis expects metal units")
    wrapped, lengths, centers, times = [], [], [], []
    start = time.perf_counter()
    first = None
    max_bond_ratio = 0.
    for frame in iter_lammps_frames(fixtures / "equilibrate_2p5.lammpstrj", units="metal"):
        atoms = frame.atoms
        length = atoms.cell.array.diagonal()
        if not atoms.pbc.all() or not np.allclose(atoms.cell.array, np.diag(length), atol=1e-10):
            raise ValueError("Only fully periodic orthorhombic cells are supported")
        if first is None:
            first = atoms.copy()
            ids = atoms.arrays["global_index"]
            index = {int(value): i for i, value in enumerate(ids)}
            g = graph.build_bond_graph(atoms, scale=1.3)
            components = graph.connected_components(g, atom_ids=ids)
            groups = np.empty(len(atoms), dtype=int)
            species, levels = [], {}
            for group, component in enumerate(components):
                formula = atoms[[index[i] for i in component]].get_chemical_formula()
                if formula not in {"C5H6", "C11H9NO2"}:
                    raise ValueError(f"Unrecognized molecule {formula}")
                species.append("CPD" if formula == "C5H6" else "dienophile")
                groups[[index[i] for i in component]] = group
                queue, seen = deque([(component[0], 0)]), {component[0]}
                while queue:
                    parent, depth = queue.popleft()
                    for child, _ in g[parent]:
                        if child not in seen:
                            seen.add(child)
                            levels.setdefault(depth+1, []).append((index[parent], index[child]))
                            queue.append((child, depth+1))
            levels = [np.asarray(levels[d]) for d in sorted(levels)]
            edges = np.array([(index[i], index[j]) for i in g for j, _ in g[i] if i < j])
            bond_cutoffs = 1.3 * (covalent_radii[atoms.numbers[edges[:,0]]] + covalent_radii[atoms.numbers[edges[:,1]]])
            masses = atoms.get_masses()
            group_mass = np.bincount(groups, weights=masses)
        np.testing.assert_array_equal(atoms.arrays["global_index"], first.arrays["global_index"])
        np.testing.assert_array_equal(atoms.numbers, first.numbers)
        np.testing.assert_allclose(atoms.get_masses(), masses)
        # Reconstruct whole molecules along fixed initial spanning trees.
        whole = atoms.positions.copy()
        for level in levels:
            parent, child = level.T
            delta = atoms.positions[child] - atoms.positions[parent]
            delta -= length * np.floor(delta/length + .5)
            whole[child] = whole[parent] + delta
        bond_delta = whole[edges[:,1]] - whole[edges[:,0]]
        ratio = np.linalg.norm(bond_delta, axis=1) / bond_cutoffs
        max_bond_ratio = max(max_bond_ratio, float(ratio.max()))
        if np.any(ratio > 1.25):
            raise ValueError("Fixed-topology reconstruction has stretched/inconsistent bonds")
        com = np.column_stack([np.bincount(groups, weights=masses*whole[:,axis]) / group_mass for axis in range(3)])
        center = np.asarray(atoms.get_celldisp()).ravel() + length/2
        com -= center
        com -= length * np.floor(com/length + .5)
        wrapped.append(com)
        lengths.append(length.copy())
        centers.append(center)
        times.append(frame.timestep * settings["timestep"])
        if frame.index % 500 == 0:
            print(f"frame {frame.index}, elapsed {time.perf_counter()-start:.1f}s", flush=True)
    centers = np.asarray(centers)
    if np.max(np.ptp(centers, axis=0)) > 1e-7:
        raise ValueError("Box center moved; choose an explicit reference-frame convention")
    times, lengths, species = np.asarray(times), np.asarray(lengths), np.asarray(species)
    unwrapped = md.unwrap_orthorhombic_com(wrapped, lengths)
    step_fraction = np.max(abs(np.diff(unwrapped, axis=0))/lengths[1:,None,:])
    np.savez_compressed(output / "com_trajectory.npz", time_ps=times, com_angstrom=unwrapped,
                        wrapped_com_angstrom=np.asarray(wrapped),
                        species=species, box_lengths_angstrom=lengths)
    report = {"method": "Whole-molecule COM then TOR, orthorhombic NPT; no drift subtraction",
              "reference": "https://doi.org/10.1021/acs.jctc.3c00308", "discard_before_ps": 100.,
              "max_saved_step_fraction_of_box": float(step_fraction), "max_initial_bond_cutoff_ratio": max_bond_ratio,
              "finite_size_correction": "none", "species": {}}
    for name in sorted(set(species)):
        position = unwrapped[:,species == name,:]
        keep = times >= 100.
        result = md.get_molecular_msd(position[keep], times[keep], unwrapped=True, selection=name)
        fits = [md.get_molecular_diffusion_coefficient(result, fit_start_ps=a, fit_end_ps=b)
                for a,b in [(20,100),(50,200),(100,300),(200,400)]]
        temporal = []
        for a,b in [(100,550),(550,1000)]:
            mask = (times >= a) & (times <= b)
            sub = md.get_molecular_msd(position[mask], times[mask], unwrapped=True, selection=name)
            f = md.get_molecular_diffusion_coefficient(sub, fit_start_ps=50, fit_end_ps=200)
            temporal.append({"trajectory_window_ps": [a,b], **f})
        entry = {"molecules": int((species == name).sum()), "fits": fits, "temporal_halves": temporal,
                 "lag_ps": result["lag_ps"].tolist(), "msd_angstrom2": result["msd_angstrom2"].tolist()}
        # Independent single-origin check of the maximum lag MSD.
        np.testing.assert_allclose(result["msd_angstrom2"][-1], np.mean(np.sum((position[-1]-position[np.flatnonzero(keep)[0]])**2, axis=1)))
        report["species"][name] = entry
        print(name, json.dumps({"fits":fits,"temporal_halves":temporal},indent=2),flush=True)
    report["elapsed_seconds"] = time.perf_counter()-start
    (output / "diffusion.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
