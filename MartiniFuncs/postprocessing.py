"""Persistent trajectory analysis, reusable reports, and streaming pair exports.

The cache stores molecular arrays, not an atomic trajectory. Reports can run
without the original simulation files; structure exports require the source dump.
cache_analysis forwards pair_workers and pair_batch_size to analyze_run's existing
process pool. Reading/unwrapping, report calculations, and exports are sequential.
"""
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
import json
from pathlib import Path
import platform

import numpy as np

from . import md_statistics as md
from .trajectory_analysis import analyze_run, molecular_window_statistics, resolve_run_files, select_frame_window


ARRAY_KEYS = ("species", "com_angstrom", "wrapped_com_angstrom", "box_lengths_angstrom",
              "com_velocity_angstrom_ps", "net_force_eV_angstrom")
SCHEMA_VERSION = 1


def _plain(value):
    """Convert scientific metadata to JSON types without pickle or NaN literals."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json(path, value):
    """Write JSON through a sibling temporary file; manifest appears last."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv(path, rows, fields=None):
    """Write scalar records; structured cells use JSON rather than Python repr."""
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            plain = _plain(row)
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in plain.items()})


def _software():
    """Record installed version and source digest, including editable installs."""
    try:
        package_version = version("MartiniFuncs")
    except PackageNotFoundError:
        package_version = "uninstalled"
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {"MartiniFuncs": package_version, "source_sha256": digest.hexdigest(),
            "python": platform.python_version(), "numpy": np.__version__}


def _fingerprints(files):
    """Record absolute path, byte size and modification time; no full dump hash."""
    fingerprints = {}
    for name, path in files.items():
        if path is None:
            fingerprints[name] = None
        else:
            path = Path(path).resolve()
            stat = path.stat()
            fingerprints[name] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return fingerprints


def cache_analysis(directory, cache_dir, *, units, timestep_ps, pair_criteria=None,
                   compute_pairs=True, pair_workers=1, pair_batch_size=32,
                   progress=print, **options):
    """Analyze once or reuse an exactly matching completed cache.

    Units/timestep are explicit. All analyze_run options except progress are
    recorded. Changing worker/batch count permits reuse; changing selection,
    source size/mtime/path, or package source code requires a new cache directory.
    Initial metadata is fixed; partial directories never count as completed.

    pair_workers > 1 uses analyze_run's existing process pool for pair screening
    with the default cached backend. pair_batch_size is retained frames per task.
    Reading and temporal unwrapping stay ordered in the parent process. These
    options do not parallelize write_report or export_pairs. For Python scripts,
    call this function inside an if __name__ == "__main__" guard when using workers.
    A reused cache records the worker settings of its original computation.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    files = resolve_run_files(directory, input_file=options.get("input_file"),
                              dump_file=options.get("dump_file"), data_file=options.get("data_file"),
                              require_input=False)
    # Resolution is captured separately; filenames need not match spelling on reuse.
    config = {k: v for k, v in options.items() if k not in {"input_file", "dump_file", "data_file"}}
    config.update(units=units, timestep_ps=timestep_ps, compute_pairs=compute_pairs,
                  pair_criteria=asdict(pair_criteria) if pair_criteria is not None else None)
    config = _plain(config)
    software = _software()
    fingerprints = _fingerprints(files)
    if cache_dir.exists():
        result, manifest = load_analysis_cache(cache_dir)
        if (manifest["configuration"] != config or manifest["sources"] != fingerprints
                or manifest["software"]["source_sha256"] != software["source_sha256"]):
            raise ValueError("Cache does not match source files, analysis settings, or package code; use a new --output directory")
        if progress:
            progress(f"Reusing completed cache: {cache_dir}")
        return result, manifest
    result = analyze_run(directory, units=units, timestep_ps=timestep_ps, compute_pairs=compute_pairs,
                         pair_criteria=pair_criteria, pair_workers=pair_workers,
                         pair_batch_size=pair_batch_size, progress=progress, **options)
    if _fingerprints(files) != fingerprints:
        raise ValueError("Simulation files changed during analysis; cache is incomplete. Wait for MD to finish.")
    cache_dir.mkdir(parents=True, exist_ok=False)
    arrays = []
    for key in ARRAY_KEYS:
        if result[key] is not None:
            np.save(cache_dir / f"{key}.npy", result[key], allow_pickle=False)
            arrays.append(key)
    metadata = {k: v for k, v in result.items() if k not in ARRAY_KEYS}
    _json(cache_dir / "analysis.json", metadata)
    _csv(cache_dir / "frames.csv", result["frames"])
    _json(cache_dir / "molecules.json", [
        {"molecule": i, "formula": str(result["species"][i]), "atom_ids": group}
        for i, group in enumerate(result["molecule_atom_ids"])])
    manifest = {"schema_version": SCHEMA_VERSION, "configuration": config, "sources": fingerprints,
                "software": software, "arrays": arrays,
                "units": {"positions": "Angstrom", "time": "ps", "velocity": "Angstrom/ps",
                          "force": "eV/Angstrom"}}
    _json(cache_dir / "manifest.json", manifest)
    return result, manifest


def load_analysis_cache(cache_dir):
    """Load metadata and memory-map molecular NPY arrays; never unpickle objects.

    Returned result follows analyze_run's dictionary API, except paths are JSON
    strings and arrays are read-only memory maps. The source dump is not opened.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not (cache_dir / "manifest.json").is_file():
        raise ValueError(f"Incomplete or missing analysis cache: {cache_dir}; use a fresh output directory")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or not set(manifest["arrays"]) <= set(ARRAY_KEYS):
        raise ValueError("Unsupported analysis cache schema")
    result = json.loads((cache_dir / "analysis.json").read_text(encoding="utf-8"))
    for key in ARRAY_KEYS:
        result[key] = np.load(cache_dir / f"{key}.npy", mmap_mode="r", allow_pickle=False) if key in manifest["arrays"] else None
    return result, manifest


def _begin_product(directory, configuration):
    """Create a new product or recognize an identical completed one; never overwrite."""
    directory = Path(directory).expanduser().resolve()
    if directory.exists():
        marker = directory / "manifest.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["configuration"] == _plain(configuration):
            return directory, False
        raise ValueError(f"Output exists with different settings or is incomplete: {directory}; choose a new name")
    directory.mkdir(parents=True, exist_ok=False)
    return directory, True


def write_report(cache_dir, output_dir, *, startframe=0, endframe=None,
                 max_lag_ps=100., fit_interval_ps=None, max_gap_frames=0,
                 stability=False, rdf_bins=None):
    """Compute a source-frame-window report from cached molecular data.

    startframe is inclusive and endframe exclusive, both zero-based source dump
    indices (not retained-row indices or timesteps). Defaults select the full
    cache. An unchanged cache can produce multiple reports in separate folders.

    MSD[k]=mean_{t,i}|R[t+k,i]-R[t,i]|^2 and correlations are mean vector dot
    products over all species molecules and valid origins. D=slope/6 is fitted
    only with an explicit fit_interval_ps. Lifetimes are recomputed within the
    window, so boundary censoring reflects that window. RDF uses instantaneous
    wrapped COMs; optional scalar ACF/batch errors delegate to stability.
    """
    result, source = load_analysis_cache(cache_dir)
    if isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int) or max_gap_frames < 0:
        raise ValueError("max_gap_frames must be a nonnegative integer")
    if rdf_bins is not None and (isinstance(rdf_bins, bool) or not isinstance(rdf_bins, int) or rdf_bins < 1):
        raise ValueError("rdf_bins must be a positive integer")
    config = dict(cache=str(Path(cache_dir).resolve()), source=source, startframe=startframe, endframe=endframe,
                  max_lag_ps=max_lag_ps, fit_interval_ps=fit_interval_ps,
                  max_gap_frames=max_gap_frames, stability=stability, rdf_bins=rdf_bins, software=_software())
    # Check completed reports before performing potentially expensive statistics.
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        return _begin_product(output, config)[0]
    window = molecular_window_statistics(result, startframe=startframe, endframe=endframe,
                                         max_lag_ps=max_lag_ps)
    rows = [r for r, keep in zip(result["frames"], window["frame_mask"]) if keep]
    times = [r["time_ps"] for r in rows]
    fits = []
    if fit_interval_ps is not None:
        for entry in window["species"].values():
            if "msd_angstrom2" not in entry:
                raise ValueError("Diffusion fit requested but molecular MSD is unavailable")
            fits.append(md.get_molecular_diffusion_coefficient(entry, fit_start_ps=fit_interval_ps[0],
                                                               fit_end_ps=fit_interval_ps[1]))
    if rdf_bins is not None and result["wrapped_com_angstrom"] is None:
        raise ValueError("RDF unavailable for this cell/reference-frame convention")
    output, _ = _begin_product(output, config)
    _csv(output / "frames.csv", rows)
    species_files = {}
    for index, (name, entry) in enumerate(window["species"].items()):
        filename = f"species_{index:03d}_lags.csv"
        keys = [key for key, value in entry.items() if isinstance(value, np.ndarray)]
        _csv(output / filename, [dict(zip(keys, values)) for values in zip(*(entry[k] for k in keys))])
        species_files[name] = {"file": filename, "n_molecules": entry["n_molecules"]}
    if fits:
        _csv(output / "diffusion.csv", fits)
    if result["pairs_by_frame"] is not None:
        pairs = [p for p, keep in zip(result["pairs_by_frame"], window["frame_mask"]) if keep]
        events = md.get_pair_lifetime(pairs, times, max_gap_frames=max_gap_frames)
        _csv(output / "pair_lifetimes.csv", [asdict(e) for e in events],
             fields=list(md.PairLifetime.__dataclass_fields__))
    if stability:
        from .stability import get_series_statistics
        summary = {key: get_series_statistics([r[key] for r in rows], sample_interval_ps=times[1]-times[0])
                   for key in rows[0] if key not in {"frame", "time_ps"}}
        _json(output / "scalar_stability.json", summary)
    if rdf_bins is not None:
        from itertools import combinations_with_replacement
        from .stability import get_molecular_com_rdf
        box = result["box_lengths_angstrom"][window["frame_mask"]]
        positions = result["wrapped_com_angstrom"][window["frame_mask"]]
        edges = np.linspace(0, .999*box.min()/2, rdf_bins+1)
        rdf_files = []
        for a, b in combinations_with_replacement(window["species"], 2):
            if a == b and window["species"][a]["n_molecules"] < 2:
                continue
            rdf = get_molecular_com_rdf(positions, box, result["species"], species_a=a, species_b=b, edges=edges)
            filename = f"rdf_{len(rdf_files):03d}.csv"
            _csv(output / filename, [dict(r_angstrom=r, g_r=g) for r, g in zip(rdf["r_angstrom"], rdf["g_r"])])
            rdf_files.append(dict(species_a=a, species_b=b, file=filename))
        _json(output / "rdf_species.json", rdf_files)
    _json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "configuration": config,
                                     "time_window_ps": window["time_window_ps"], "frame_window": window["frame_window"],
                                     "species": species_files,
                                     "notices": result["notices"]})
    return output


def export_pairs(cache_dir, output_dir, *, shell_radius=None, startframe=0, endframe=None, progress=print):
    """Stream selected cached encounters into pair/shell Extended XYZ and an index.

    Select source frames in [startframe, endframe), defaulting to the entire
    cache. Bounds are zero-based dump indices, including when frames were strided.
    One structure per accepted pair per retained frame, not one per lifetime.
    Re-read source frames once; translations r'=r+image @ cell leave forces
    unchanged. Forces remain eV/Angstrom, positions Angstrom, stored velocities
    ASE units. Shell selection uses atom-to-pair distance then completes molecules.
    Source fingerprints must match the cache; no UMA model is evaluated.
    """
    from . import graph
    from .geom_utils import (get_pair_positions_and_forces, get_pair_shell_clusters,
                             get_cluster_positions_and_forces, pair_data_to_extxyz)
    from .lammpsio import iter_lammps_frames
    from .pair_selection import PairCriteria, assess_prereactive_pairs

    result, source = load_analysis_cache(cache_dir)
    if result["pairs_by_frame"] is None:
        raise ValueError("Cache contains no pair analysis; rerun analysis with pairs enabled")
    if _fingerprints(result["files"]) != source["sources"]:
        raise ValueError("Source files changed since caching; refuse to export mismatched structures/forces")
    if shell_radius is not None and (not np.isfinite(shell_radius) or shell_radius <= 0):
        raise ValueError("shell_radius must be positive and finite")
    keep = select_frame_window(result, startframe=startframe, endframe=endframe)
    config = dict(cache=str(Path(cache_dir).resolve()), source=source, shell_radius=shell_radius,
                  startframe=startframe, endframe=endframe, software=_software())
    output, create = _begin_product(output_dir, config)
    if not create:
        return output
    selected = {r["frame"]: (r, p, s)
                for r, p, s, retained in zip(result["frames"], result["pairs_by_frame"],
                                            result["pair_shifts_by_frame"], keep) if retained}
    options = source["configuration"]
    mapping = options.get("type_symbols")
    if mapping is not None:
        mapping = {int(k): v for k, v in mapping.items()}
    scale = options.get("bond_scale", 1.3)
    last_frame = max(selected) if selected else -1
    count = 0
    with (output / "pairs.extxyz").open("w", encoding="utf-8") as pair_stream, \
         (output / "index.jsonl").open("w", encoding="utf-8") as index_stream:
        from contextlib import ExitStack
        with ExitStack() as stack:
            cluster_stream = stack.enter_context((output / "clusters.extxyz").open("w", encoding="utf-8")) if shell_radius is not None else None
            if selected:
                for frame in iter_lammps_frames(result["files"]["dump"], units=options["units"],
                                                input_datafile=result["files"]["data"], type_symbols=mapping):
                    if frame.index > last_frame:
                        break
                    if frame.index not in selected:
                        continue
                    row, pairs, shifts = selected[frame.index]
                    if not pairs:
                        continue
                    if "lammps_forces" not in frame.atoms.arrays:
                        raise ValueError("Pair structure export requires source force columns")
                    bonded = graph.build_bond_graph(frame.atoms, scale=scale)
                    components = graph.connected_components(bonded, atom_ids=graph.atom_ids(frame.atoms))
                    memberships = {frozenset(c) for c in components}
                    if any(frozenset(c) not in memberships for pair in pairs for c in pair):
                        raise ValueError(f"Frame {frame.index}: cached pair membership differs from current graph; use reference analysis")
                    records = get_pair_positions_and_forces(frame.atoms, frame, pairs, shifts, bond_scale=scale, graph=bonded)
                    buffer = StringIO()
                    pair_data_to_extxyz(records, buffer)
                    pair_stream.write(buffer.getvalue())
                    assessments = assess_prereactive_pairs(frame.atoms, pairs, shifts, criteria=PairCriteria(), scale=scale, graph=bonded)
                    if cluster_stream is not None:
                        clusters = get_pair_shell_clusters(frame.atoms, pairs, shifts, shell_radius=shell_radius,
                                                           scale=scale, graph=bonded, components=components)
                        buffer = StringIO()
                        pair_data_to_extxyz(get_cluster_positions_and_forces(frame.atoms, frame, clusters), buffer)
                        cluster_stream.write(buffer.getvalue())
                    for candidate, (pair, shift, assessment) in enumerate(zip(pairs, shifts, assessments)):
                        record = dict(structure_index=count, frame=frame.index, timestep=frame.timestep,
                                      time_ps=row["time_ps"], candidate=candidate, pair_atom_ids=pair,
                                      cpd_shift=shift, geometry=asdict(assessment.geometry) if assessment.geometry else None,
                                      geometry_errors=assessment.reasons)
                        index_stream.write(json.dumps(_plain(record), allow_nan=False) + "\n")
                        count += 1
                    if progress and frame.index % 100 == 0:
                        progress(f"Exported {count} pair structures through frame {frame.index}")
    if _fingerprints(result["files"]) != source["sources"]:
        raise ValueError("Source files changed during export; output is incomplete")
    _json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "configuration": config,
                                     "structures": count, "pair_file": "pairs.extxyz",
                                     "cluster_file": "clusters.extxyz" if shell_radius is not None else None,
                                     "index": "index.jsonl"})
    return output
