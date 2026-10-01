"""Directory-driven molecular trajectory analysis without precomputed reports."""
from collections import deque
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
import re

import numpy as np
from ase.data import covalent_radii

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
                units=None, timestep_ps=None, units_style=None, type_symbols=None, progress=None,
                reaction_atom_counts=None):
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
    TOR diffusion requires fully periodic cells: orthorhombic, triclinic and
    sheared (fix deform) cells are supported, and the box center may move.
    Displacements are lab-frame: imposed shear flow or barostat dilation is not
    subtracted. Non-periodic directions disable COM tracking; scalar/velocity/
    force/pair analyses remain available.

    Returns both com_angstrom (TOR, for MSD) and wrapped_com_angstrom (actual
    instantaneous geometry wrapped about the cell center, for RDFs), with
    cell_angstrom (frames, 3, 3) row-vector cells and box_lengths_angstrom (the
    cell diagonal, equal to the box lengths for orthorhombic cells). The COM
    arrays are None when COM tracking is unsupported. max_com_step_cell_fraction
    is the largest per-source-frame COM step as a fraction of the cell (in
    fractional coordinates); TOR is ambiguous at 0.5 and a notice is added above
    0.25. Vector arrays share retained-frame and fixed molecular order; they do
    not share the atomic row axis of the source Atoms.

    Time is dump_step*timestep_ps. For fixed molecular groups,
    R_m=sum_i(m_i*r_whole,i)/sum_i m_i, V_m=sum_i(m_i*v_i)/sum_i m_i,
    and F_m=sum_i F_i. V_m is converted to Angstrom/ps. Whole positions follow
    minimum-image bond vectors along the initial graph's spanning tree.
    With R_t the whole-molecule COM and H_t the cell, accumulate
    u_t=u_(t-1)+mic(R_t-R_(t-1), H_t), u_0 the COM wrapped about the first cell
    center; mic is md_statistics.minimum_image, which for orthorhombic cells is
    d-L*floor(d/L+1/2). This TOR update uses every source frame before striding.
    Temperature is 2*K/(f*k_B) without COM subtraction; '3N-3' specifies f only.

    reaction_atom_counts=(11, 23) enables an exact-size possible-product screen
    for the current CPD/MAL system. It rebuilds connectivity on EVERY source frame,
    before striding, and raises PossibleReactionError with atom IDs at the first
    candidate. This costs an additional graph calculation per source frame. None
    skips screening; fixed-connectivity analysis then assumes no new products.

    When compute_pairs=True, pair_backend='cached' reuses initial molecule/site
    identities and the whole coordinates already reconstructed above. It assumes
    unchanged connectivity; existing checks detect broken reconstruction, not
    every new bond. 'reference' rebuilds the pair graph each retained frame.
    pair_workers=1 avoids process overhead; >1 uses indexed frame chunks with
    pair_batch_size source frames per task and at most 2*pair_workers tasks in
    flight. Parsing, reconstruction, reaction and pair screening are parallel.
    TOR accumulation remains ordered in the parent across all source frames.
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
    from .frame_parallel import analysis_frames, prepare_frame
    preparation = {}
    reader_options = dict(units="metal", input_datafile=files["data"], type_symbols=mapping)
    frames = (analysis_frames(files["dump"], reader_options, preparation,
                             workers=pair_workers, batch_size=pair_batch_size,
                             max_frames=max_frames, progress=progress)
              if pair_workers > 1 else iter_lammps_frames(files["dump"], **reader_options))
    rows, coms, velocities, forces, pairs_by_frame = [], [], [], [], []
    pair_shifts_by_frame = []
    wrapped_coms, box_lengths, cells = [], [], []
    max_step_fraction = 0.
    notices = ["Molecular membership fixed from first-frame bond graph; species grouped by formula.",
               "No potential energy or pressure computed from atomic forces; thermo logs are not parsed."]
    diffusion_reason = None
    first = None
    previous_com = running_com = None
    frames_read = 0
    previous_step = None
    with ExitStack() as stack:
        if hasattr(frames, 'close'):
            stack.callback(frames.close)
        for frame in islice(frames, max_frames):
            if previous_step is not None and frame.timestep <= previous_step:
                raise ValueError("Timesteps must increase; split restarted/concatenated trajectories explicitly")
            previous_step = frame.timestep
            frames_read += 1
            atoms = getattr(frame, 'atoms', None)
            if first is None and reaction_atom_counts is not None:
                from .reaction_detection import find_possible_products, PossibleReactionError
                hits = find_possible_products(atoms, reactant_atom_counts=reaction_atom_counts,
                                              bond_scale=bond_scale)
                if hits:
                    raise PossibleReactionError([dict(frame=int(frame.index), timestep=int(frame.timestep), **hit)
                                                 for hit in hits])
                preparation["prescreened_frame"] = frame.index
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
                topology = None
                if compute_pairs:
                    if set(species) != {"C5H6", "C11H9NO2"}:
                        raise ValueError("Pair lifetime screening is currently specific to the CPD/dienophile mixture")
                    if pair_backend == "cached":
                        from .trajectory_pairs import FixedPairTopology
                        topology = FixedPairTopology.from_frame(atoms, bonded, components)
                        notices.append("Pair sites/topology cached from the first frame. The optional product-size "
                                       "screen does not track all new bonds. Use pair_backend='reference' "
                                       "to rebuild pair topology each retained frame.")
                has_velocities = "momenta" in atoms.arrays
                has_forces = "lammps_forces" in atoms.arrays
                if degrees_of_freedom is None:
                    notices.append("Temperature omitted: set degrees_of_freedom for this simulation.")
                if not has_velocities:
                    notices.append("No velocities: temperature and molecular VACF unavailable.")
                if not has_forces:
                    notices.append("No forces: molecular force correlation unavailable.")
                preparation.update(ids=ids, numbers=first.numbers, masses=masses,
                    has_velocities=has_velocities, has_forces=has_forces,
                    levels=levels, edges=edges, cutoffs=cutoffs, groups=groups,
                    molecular_masses=molecular_masses, topology=topology,
                    compute_pairs=compute_pairs, pair_criteria=pair_criteria,
                    frame_stride=frame_stride, bond_scale=bond_scale,
                    degrees_of_freedom=degrees_of_freedom,
                    reaction_atom_counts=reaction_atom_counts)
            prepared = getattr(frame, "prepared", None)
            if prepared is None:
                prepared = prepare_frame(frame, preparation)
            com, cell, pbc = prepared.com, prepared.cell, prepared.pbc
            pairs, shifts, scalars = prepared.pairs, prepared.shifts, prepared.scalars
            velocity, force = prepared.velocity, prepared.force
            if not pbc.all():
                diffusion_reason = "Molecular MSD disabled: COM tracking requires a fully periodic cell."
            elif diffusion_reason is None:
                center = prepared.origin + cell.sum(axis=0)/2
                # Canonical image inside the current cell (TOR convention), then
                # destination-cell minimum-image steps; see md.unwrap_com_trajectory.
                in_cell = md.wrap_about_center(com, cell, center)
                wrapped = in_cell - center
                if previous_com is None:
                    running_com = wrapped.copy()
                else:
                    step = md.minimum_image(in_cell-previous_com, cell)
                    running_com += step
                    fraction = np.abs(np.linalg.solve(cell.T, step.reshape(-1, 3).T)).max()
                    max_step_fraction = max(max_step_fraction, float(fraction))
                previous_com = in_cell
            if frame.index % frame_stride:
                continue
            row = {"frame": frame.index, "timestep": frame.timestep, "time_ps": frame.timestep*settings["timestep"],
                   **scalars}
            if has_velocities:
                velocities.append(velocity)
            if has_forces:
                forces.append(force)
            if compute_pairs:
                pairs_by_frame.append(pairs)
                pair_shifts_by_frame.append(shifts)
                row["candidate_pairs"] = len(pairs)
            rows.append(row)
            wrapped_coms.append(wrapped.copy() if diffusion_reason is None else np.full_like(com,np.nan))
            box_lengths.append(cell.diagonal().copy())
            cells.append(cell.copy())
            coms.append(running_com.copy() if diffusion_reason is None else np.full_like(com,np.nan))
            if progress is not None and (len(rows)==1 or len(rows)%100==0):
                progress(f"Processed {frames_read} frames; retained {len(rows)}; time {row['time_ps']:g} ps")
    if len(rows)<2:
        raise ValueError("Need at least two retained frames")
    times = np.array([row["time_ps"] for row in rows])
    if np.any(np.diff(times)<=0):
        raise ValueError("Timesteps must increase; split restarted/concatenated trajectories explicitly")
    if diffusion_reason:
        notices.append(diffusion_reason)
    elif max_step_fraction > .25:
        notices.append(f"Largest per-frame COM step is {max_step_fraction:.2f} of the cell; TOR unwrapping "
                       "is ambiguous at 0.5. Save frames more often.")
    if not diffusion_reason and len(cells) and not all(md._is_diagonal(c) for c in cells):
        notices.append("Non-orthorhombic cell: COM displacements are lab-frame and include any imposed shear flow.")
    return {"files":files,"settings":settings,"frames":rows,"frames_read":frames_read,
            "reaction_screen": {"reactant_atom_counts": list(reaction_atom_counts),
                                "bond_scale": bond_scale, "frames_checked": frames_read} if reaction_atom_counts is not None else None,
            "source_frame_limit":max_frames,"species":np.array(species),
            "molecule_atom_ids":components,"com_angstrom":None if diffusion_reason else np.asarray(coms),
            "wrapped_com_angstrom":None if diffusion_reason else np.asarray(wrapped_coms),
            "box_lengths_angstrom":np.asarray(box_lengths),
            "cell_angstrom":np.asarray(cells),
            "max_com_step_cell_fraction":None if diffusion_reason else max_step_fraction,
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


def molecular_window_statistics(result, *, startframe=0, endframe=None, max_lag_ps=100.,
                                lag_statistics=None, progress=None):
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
    lag_statistics selects msd, vacf and/or force-acf; None keeps all three.
    An empty sequence selects metadata only and does not allocate lag arrays.
    """
    requested = {'msd', 'vacf', 'force-acf'} if lag_statistics is None else set(lag_statistics)
    if requested - {'msd', 'vacf', 'force-acf'}:
        raise ValueError('Unknown lag statistic')
    times = np.array([r["time_ps"] for r in result["frames"]])
    if not np.isfinite(max_lag_ps) or max_lag_ps<=0:
        raise ValueError("Require positive finite max_lag_ps")
    keep = select_frame_window(result, startframe=startframe, endframe=endframe)
    selected_times = times[keep]
    if requested and len(selected_times)<3:
        raise ValueError("Select at least three uniformly spaced frames for lag statistics")
    if requested:
        md._times(selected_times, len(selected_times), uniform=True)
        dt = selected_times[1]-selected_times[0]
        nlag = md.lag_count(max_lag_ps, dt, len(selected_times))
    statistics = {}
    for name in sorted(set(result["species"])) if requested else []:
        mask = result["species"]==name
        entry = {"lag_ps":np.arange(nlag)*dt,"n_origins":len(selected_times)-np.arange(nlag),
                 "n_molecules":int(mask.sum()),"selection":name}
        for source, key in (("com_angstrom","msd_angstrom2"),
                            ("com_velocity_angstrom_ps","vacf"),("net_force_eV_angstrom","force_acf")):
            option = {'msd_angstrom2': 'msd', 'vacf': 'vacf', 'force_acf': 'force-acf'}[key]
            if option not in requested:
                continue
            if result[source] is None:
                if progress:
                    progress(f'{name}: {option} unavailable; source data/cell convention does not support it')
                continue
            if progress:
                progress(f'{name}: starting {option}, {len(selected_times)} frames, {nlag} lags')
            values = np.asarray(result[source][keep][:,mask], dtype=float)
            # FFT evaluation of the same all-origin estimator; O(T log T) for any lag range.
            entry[key] = md.lag_msd(values, nlag) if key == "msd_angstrom2" else md.lag_correlation(values, nlag)
            if progress:
                progress(f'{name}: {option} complete ({nlag} lags)')
        statistics[name] = entry
    return {"time_window_ps":[float(selected_times[0]),float(selected_times[-1])],
            "frame_window":[int(startframe), int(endframe if endframe is not None else result.get("frames_read", result["frames"][-1]["frame"]+1))],
            "frame_mask":keep,"species":statistics}
