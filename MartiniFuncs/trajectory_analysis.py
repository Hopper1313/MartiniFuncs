"""Directory-driven molecular trajectory analysis without precomputed reports."""
from collections import deque
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
import re

import numpy as np
from ase import units as ase_units
from ase.data import covalent_radii
from ase.geometry import find_mic

from . import graph, md_statistics as md
from .lammpsio import iter_lammps_frames, read_lammps_input_settings


def resolve_run_files(directory, *, input_file=None, dump_file=None, data_file=None, require_input=True):
    """Find a run's input, dump and optional data file without reading them.

    For each kind, prefer matches directly in directory; search recursively
    only when none exist there. Multiple matches always require an override,
    including optional files. Overrides are relative to directory or absolute.
    Returns absolute Paths under keys 'input', 'dump', 'data'; absent optional
    files are None. Set require_input=False when using explicit physics settings.
    Discovery does not verify that the files describe the same simulation.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Simulation directory does not exist: {directory}")

    def select(override, patterns, required, argument):
        if override is not None:
            path = Path(override).expanduser()
            path = path if path.is_absolute() else directory/path
            if not path.is_file():
                raise FileNotFoundError(path)
            return path.resolve()
        choices = sorted({p for pattern in patterns for p in directory.glob(pattern) if p.is_file()})
        if not choices:
            choices = sorted({p for pattern in patterns for p in directory.rglob(pattern) if p.is_file()})
        if len(choices) == 1:
            return choices[0]
        if not choices and not required:
            return None
        names = ", ".join(str(p.relative_to(directory)) for p in choices) or "none"
        raise ValueError(f"Select an explicit {argument} for {patterns} in {directory}; found: {names}")

    return {"input": select(input_file, ("*.in", "in.*"), require_input, "input_file"),
            "dump": select(dump_file, ("*.lammpstrj", "*.dump"), True, "dump_file"),
            "data": select(data_file, ("*.data",), False, "data_file")}


def analyze_run(directory, *, input_file=None, dump_file=None, data_file=None,
                frame_stride=1, max_frames=None, bond_scale=1.3,
                degrees_of_freedom=None, compute_pairs=False, pair_criteria=None,
                pair_backend="cached", pair_workers=1, pair_batch_size=32,
                units=None, timestep_ps=None, units_style=None, type_symbols=None, progress=None):
    """Stream a text dump; retain scalar histories and all molecular COM vectors.

    max_frames caps source frames read, not retained frames. Molecules are the
    first frame's distance-graph components, with fixed membership thereafter;
    species are grouped by chemical formula (isomers are not distinguished).
    Existing bonds are checked each source frame, but new bond formation is
    not detected: this analysis assumes a nonreactive trajectory.

    Uses metal units and constant timestep from the input. For restart inputs,
    units may come from a LAMMPS write_data header or explicit units.
    Explicit units and timestep_ps take precedence. When both are given, input
    parsing and data-header inference are skipped entirely; no input file is
    required. timestep_ps is always in picoseconds, never dump step spacing.
    units_style is retained as an alias for older notebooks. type_symbols is
    an optional {LAMMPS type: element} mapping when the dump lacks elements.
    Missing velocities
    or forces disable only those metrics. Temperature requires explicit DOF.
    COMs are reconstructed at every source frame before optional downsampling.
    TOR diffusion requires periodic orthorhombic boxes with fixed box centers;
    other cells still support scalar/velocity/force/pair analyses.

    Returns both com_angstrom (TOR, for MSD) and wrapped_com_angstrom (actual
    instantaneous centered-box geometry, for RDFs), with box_lengths_angstrom.
    The COM arrays are None when the supported cell convention fails. Vector
    arrays share retained-frame and fixed molecular order; they do not share
    the atomic row axis of the source Atoms.

    Time is dump_step*timestep_ps. For fixed molecular groups,
    R_m=sum_i(m_i*r_whole,i)/sum_i m_i, V_m=sum_i(m_i*v_i)/sum_i m_i,
    and F_m=sum_i F_i. V_m is converted to Angstrom/ps. Whole positions follow
    minimum-image bond vectors along the initial graph's spanning tree.
    For centered wrapped COM w_t, accumulate u_t=u_(t-1)+d_t-L_t*floor(d_t/L_t+1/2),
    d_t=w_t-w_(t-1). This TOR update uses every source frame before striding.
    Temperature is 2*K/(f*k_B) without COM subtraction; '3N-3' specifies f only.

    When compute_pairs=True, pair_backend='cached' reuses initial molecule/site
    identities and the whole coordinates already reconstructed above. It assumes
    unchanged connectivity; existing checks detect broken reconstruction, not
    every new bond. 'reference' rebuilds the pair graph each retained frame.
    pair_workers=1 avoids process overhead; >1 enables spawned processes with
    pair_batch_size retained frames per batch and at most 2*pair_workers batches
    in flight. Only pair calculations are parallel; reading and TOR are sequential.
    Standalone multiprocessing scripts need an if __name__ == '__main__' guard.
    Returned pairs and pair_shifts_by_frame stay in retained-frame order.
    Cached pairs retain initial component atom order; the reference graph's
    traversal order can change per frame. Molecular identity is the atom-ID set,
    not that traversal order. Candidate pair order and image conventions agree.
    """
    if pair_backend not in {"cached", "reference"}:
        raise ValueError("pair_backend must be 'cached' or 'reference'")
    for name, value in (("pair_workers", pair_workers), ("pair_batch_size", pair_batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if compute_pairs and pair_backend == "reference" and pair_workers != 1:
        raise ValueError("Parallel pair workers require pair_backend='cached'")
    for name, value in (("frame_stride", frame_stride), ("max_frames", max_frames)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if units is not None and units_style is not None and units != units_style:
        raise ValueError("units and units_style must agree when both are provided")
    explicit_units = units if units is not None else units_style
    explicit_settings = explicit_units is not None and timestep_ps is not None
    files = resolve_run_files(directory, input_file=input_file, dump_file=dump_file,
                              data_file=data_file, require_input=not explicit_settings)
    settings = (read_lammps_input_settings(files["input"]) if not explicit_settings
                else {"dump_elements": {}})
    unit_sources = {}
    if explicit_units is None and "units" in settings:
        unit_sources["input"] = settings["units"]
    if explicit_units is None and files["data"] is not None:
        with files["data"].open(encoding="utf-8") as handle:
            header = handle.readline().strip()
        # Only recognize LAMMPS's structured write_data header, not free comments.
        if header.startswith("LAMMPS data file via write_data,"):
            match = re.search(r"\bunits\s*=\s*([a-z]+)\b", header)
            if match:
                unit_sources["LAMMPS data header"] = match.group(1)
    if explicit_units is not None:
        unit_sources["explicit units"] = explicit_units
    if len(set(unit_sources.values())) > 1:
        raise ValueError(f"Conflicting units metadata: {unit_sources}")
    if not unit_sources:
        raise ValueError("Units are absent from input/data metadata (common for read_restart). "
                         "Set units='metal' explicitly if that is the simulation's unit style.")
    settings["units"] = next(iter(unit_sources.values()))
    settings["units_sources"] = unit_sources
    if settings["units"] != "metal":
        raise ValueError("This molecular analysis currently supports only units metal")
    if timestep_ps is not None:
        if not np.isfinite(timestep_ps) or timestep_ps <= 0:
            raise ValueError("timestep_ps must be finite and positive")
        settings["timestep"] = float(timestep_ps)
    settings["timestep_source"] = "explicit timestep_ps" if timestep_ps is not None else "input"
    if not np.isfinite(settings.get("timestep",np.nan)) or settings.get("timestep",0) <= 0:
        raise ValueError("Timestep missing from input: supply timestep_ps from the original run/restart settings")
    mappings = list(settings["dump_elements"].values())
    mapping = type_symbols if type_symbols is not None else (mappings[0] if mappings and all(m == mappings[0] for m in mappings) else None)
    frames = iter_lammps_frames(files["dump"], units="metal", input_datafile=files["data"], type_symbols=mapping)
    rows, coms, velocities, forces, pairs_by_frame = [], [], [], [], []
    pair_shifts_by_frame = []
    pair_runner = None
    wrapped_coms, box_lengths = [], []
    notices = ["Molecular membership fixed from first-frame bond graph; species grouped by formula.",
               "No potential energy or pressure computed from atomic forces; thermo logs are not parsed."]
    diffusion_reason = None
    first = None
    previous_wrapped = previous_box_center = running_com = None
    frames_read = 0
    previous_step = None
    with ExitStack() as stack:
        for frame in islice(frames, max_frames):
            if previous_step is not None and frame.timestep <= previous_step:
                raise ValueError("Timesteps must increase; split restarted/concatenated trajectories explicitly")
            previous_step = frame.timestep
            frames_read += 1
            atoms = frame.atoms
            if first is None:
                first = atoms.copy()
                if degrees_of_freedom == "3N-3":
                    degrees_of_freedom = 3*len(atoms)-3
                ids = atoms.arrays["global_index"]
                lookup = {int(i): row for row, i in enumerate(ids)}
                bonded = graph.build_bond_graph(atoms, scale=bond_scale)
                components = graph.connected_components(bonded, atom_ids=ids)
                groups = np.empty(len(atoms), dtype=int)
                species, levels = [], {}
                for group, component in enumerate(components):
                    indices = [lookup[i] for i in component]
                    groups[indices] = group
                    species.append(atoms[indices].get_chemical_formula())
                    queue, seen = deque([(component[0],0)]), {component[0]}
                    while queue:
                        parent, depth = queue.popleft()
                        for child, _ in bonded[parent]:
                            if child not in seen:
                                seen.add(child)
                                levels.setdefault(depth+1, []).append((lookup[parent],lookup[child]))
                                queue.append((child,depth+1))
                levels = [np.asarray(levels[d]) for d in sorted(levels)]
                edges = np.array([(lookup[i],lookup[j]) for i in bonded for j,_ in bonded[i] if i<j], dtype=int).reshape(-1,2)
                cutoffs = bond_scale*(covalent_radii[atoms.numbers[edges[:,0]]] + covalent_radii[atoms.numbers[edges[:,1]]])
                masses = atoms.get_masses()
                molecular_masses = np.bincount(groups, weights=masses)
                if compute_pairs:
                    if set(species) != {"C5H6", "C11H9NO2"}:
                        raise ValueError("Pair lifetime screening is currently specific to the CPD/dienophile mixture")
                    if pair_backend == "cached":
                        from .trajectory_pairs import FixedPairTopology, PairBatchRunner
                        topology = FixedPairTopology.from_frame(atoms, bonded, components)
                        pair_runner = stack.enter_context(PairBatchRunner(
                            topology, pair_criteria, workers=pair_workers, batch_size=pair_batch_size))
                        notices.append("Pair sites/topology cached from the first frame; new bonds are not detected. "
                                       "Use pair_backend='reference' to rebuild pair topology each retained frame.")
                has_velocities = "momenta" in atoms.arrays
                has_forces = "lammps_forces" in atoms.arrays
                if degrees_of_freedom is None:
                    notices.append("Temperature omitted: set degrees_of_freedom for this simulation.")
                if not has_velocities:
                    notices.append("No velocities: temperature and molecular VACF unavailable.")
                if not has_forces:
                    notices.append("No forces: molecular force correlation unavailable.")
            np.testing.assert_array_equal(atoms.arrays["global_index"], ids)
            np.testing.assert_array_equal(atoms.numbers, first.numbers)
            np.testing.assert_allclose(atoms.get_masses(), masses)
            if has_velocities != ("momenta" in atoms.arrays) or has_forces != ("lammps_forces" in atoms.arrays):
                raise ValueError("Velocity/force availability changes within the dump")
            length = atoms.cell.array.diagonal()
            orthogonal = atoms.pbc.all() and np.allclose(atoms.cell.array,np.diag(length),atol=1e-10) and np.all(length>0)
            whole = atoms.positions.copy()
            for level in levels:
                parent, child = level.T
                delta = atoms.positions[child]-atoms.positions[parent]
                if orthogonal:
                    delta -= length*np.floor(delta/length+.5)
                else:
                    delta = find_mic(delta, atoms.cell, atoms.pbc)[0]
                whole[child] = whole[parent]+delta
            if len(edges) and np.any(np.linalg.norm(whole[edges[:,1]]-whole[edges[:,0]],axis=1) > 1.25*cutoffs):
                raise ValueError(f"Frame {frame.index}: fixed molecular topology no longer reconstructs consistently")
            com = np.column_stack([np.bincount(groups,weights=masses*whole[:,axis])/molecular_masses for axis in range(3)])
            if not orthogonal:
                diffusion_reason = "Molecular MSD disabled: TOR currently requires fully periodic orthorhombic cells."
            elif diffusion_reason is None:
                center = np.asarray(atoms.get_celldisp()).ravel()+length/2
                if previous_box_center is not None and not np.allclose(center,previous_box_center,rtol=0,atol=1e-7):
                    diffusion_reason = "Molecular MSD disabled: box center moves; an explicit reference-frame convention is needed."
                wrapped = com-center
                wrapped -= length*np.floor(wrapped/length+.5)
                if previous_wrapped is None:
                    running_com = wrapped.copy()
                else:
                    delta = wrapped-previous_wrapped
                    delta -= length*np.floor(delta/length+.5)
                    running_com += delta
                previous_wrapped, previous_box_center = wrapped, center
            if frame.index % frame_stride:
                continue
            row = {"frame": frame.index, "time_ps": frame.timestep*settings["timestep"],
                   "density_g_cm3": md.get_frame_density(atoms), "volume_angstrom3": atoms.get_volume()}
            if has_velocities:
                if degrees_of_freedom is not None:
                    row["temperature_K"] = md.get_frame_temperature(atoms,degrees_of_freedom=degrees_of_freedom)
                v = atoms.get_velocities()*1000*ase_units.fs
                velocities.append(np.column_stack([np.bincount(groups,weights=masses*v[:,axis])/molecular_masses for axis in range(3)]))
            if has_forces:
                f = atoms.arrays["lammps_forces"]
                forces.append(np.column_stack([np.bincount(groups,weights=f[:,axis]) for axis in range(3)]))
            if compute_pairs:
                if pair_runner is not None:
                    pair_runner.add(frame.index, whole, atoms.cell.array, atoms.pbc)
                else:
                    from .geom_utils import get_all_prereactive_pairs
                    pairs, shifts = get_all_prereactive_pairs(atoms, scale=bond_scale, criteria=pair_criteria)
                    pairs_by_frame.append(pairs)
                    pair_shifts_by_frame.append(shifts)
                    row["candidate_pairs"] = len(pairs)
            rows.append(row)
            wrapped_coms.append(wrapped.copy() if diffusion_reason is None else np.full_like(com,np.nan))
            box_lengths.append(length.copy())
            coms.append(running_com.copy() if diffusion_reason is None else np.full_like(com,np.nan))
            if progress is not None and (len(rows)==1 or len(rows)%100==0):
                progress(f"Read {frames_read} frames; retained {len(rows)}; time {row['time_ps']:g} ps")
        if pair_runner is not None:
            for row, (pairs, shifts) in zip(rows, pair_runner.finish()):
                pairs_by_frame.append(pairs)
                pair_shifts_by_frame.append(shifts)
                row["candidate_pairs"] = len(pairs)
            if progress is not None:
                progress(f"Pair analysis complete for {len(rows)} retained frames")
    if len(rows)<2:
        raise ValueError("Need at least two retained frames")
    times = np.array([row["time_ps"] for row in rows])
    if np.any(np.diff(times)<=0):
        raise ValueError("Timesteps must increase; split restarted/concatenated trajectories explicitly")
    if diffusion_reason:
        notices.append(diffusion_reason)
    return {"files":files,"settings":settings,"frames":rows,"frames_read":frames_read,
            "source_frame_limit":max_frames,"species":np.array(species),
            "molecule_atom_ids":components,"com_angstrom":None if diffusion_reason else np.asarray(coms),
            "wrapped_com_angstrom":None if diffusion_reason else np.asarray(wrapped_coms),
            "box_lengths_angstrom":np.asarray(box_lengths),
            "com_velocity_angstrom_ps":np.asarray(velocities) if has_velocities else None,
            "net_force_eV_angstrom":np.asarray(forces) if has_forces else None,
            "pairs_by_frame":pairs_by_frame if compute_pairs else None,
            "pair_shifts_by_frame":pair_shifts_by_frame if compute_pairs else None,
            "pair_analysis": {"backend": pair_backend, "workers": pair_workers,
                              "batch_size": pair_batch_size} if compute_pairs else None,
            "notices":notices}


def select_frame_window(result, *, startframe=0, endframe=None):
    """Return a retained-frame mask for source frames in [startframe, endframe).

    Indices are zero-based positions in the original dump, not LAMMPS timesteps
    or row numbers after striding. For frame_stride=2, selecting [3, 9) keeps
    source frames 4, 6, 8. Defaults select all cached frames. endframe=None means
    frames_read (including source frames omitted by striding). Bounds outside
    the cached source range, reversed bounds, and empty selections are errors.
    """
    frames = np.asarray([r["frame"] for r in result["frames"]])
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu" or np.any(np.diff(frames) <= 0):
        raise ValueError("Require increasing integer source frame indices")
    total = result.get("frames_read", int(frames[-1]) + 1)
    endframe = total if endframe is None else endframe
    for name, value in (("startframe", startframe), ("endframe", endframe)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer source frame index")
    if not 0 <= startframe < endframe <= total:
        raise ValueError(f"Require 0 <= startframe < endframe <= {total} (cached source frames)")
    keep = (frames >= startframe) & (frames < endframe)
    if not keep.any():
        raise ValueError("No retained frames in this range; check frame_stride and source frame indices")
    return keep


def molecular_window_statistics(result, *, startframe=0, endframe=None, max_lag_ps=100.):
    """All-molecule, all-time-origin MSD and dot-product correlations by formula.

    Select zero-based source frames in [startframe, endframe), before interpreting
    the stride. Defaults use the entire cached trajectory. See select_frame_window
    for bounds. Bounded lag range controls work; correlations are raw (no temporal
    mean subtraction). At least three retained, uniformly spaced frames are needed.

    Within the selected T frames and each species' N molecules,
    MSD[k]=sum_{t,i}|R[t+k,i]-R[t,i]|^2/((T-k)*N);
    C[k]=sum_{t,i}x[t,i] dot x[t+k,i]/((T-k)*N), using molecular
    COM velocities for VACF or summed molecular forces for force_acf.
    lag[k]=k*dt<=max_lag_ps and n_origins=T-k. Timestamps still supply dt in ps;
    selecting frames does not change the physical time units of the statistics.
    """
    times = np.array([r["time_ps"] for r in result["frames"]])
    if not np.isfinite(max_lag_ps) or max_lag_ps<=0:
        raise ValueError("Require positive finite max_lag_ps")
    keep = select_frame_window(result, startframe=startframe, endframe=endframe)
    selected_times = times[keep]
    if len(selected_times)<3:
        raise ValueError("Select at least three uniformly spaced frames for lag statistics")
    md._times(selected_times, len(selected_times), uniform=True)
    dt = selected_times[1]-selected_times[0]
    nlag = min(len(selected_times),int(np.floor(max_lag_ps/dt))+1)
    statistics = {}
    for name in sorted(set(result["species"])):
        mask = result["species"]==name
        entry = {"lag_ps":np.arange(nlag)*dt,"n_origins":len(selected_times)-np.arange(nlag),
                 "n_molecules":int(mask.sum()),"selection":name}
        for source, key in (("com_angstrom","msd_angstrom2"),
                            ("com_velocity_angstrom_ps","vacf"),("net_force_eV_angstrom","force_acf")):
            if result[source] is None:
                continue
            values = result[source][keep][:,mask]
            out = []
            for lag in range(nlag):
                left = values[:len(values)-lag]
                right = values[lag:]
                product = (right-left)**2 if key=="msd_angstrom2" else left*right
                out.append(np.mean(np.sum(product,axis=-1)))
            entry[key] = np.array(out)
        statistics[name] = entry
    return {"time_window_ps":[float(selected_times[0]),float(selected_times[-1])],
            "frame_window":[int(startframe), int(endframe if endframe is not None else result.get("frames_read", result["frames"][-1]["frame"]+1))],
            "frame_mask":keep,"species":statistics}
