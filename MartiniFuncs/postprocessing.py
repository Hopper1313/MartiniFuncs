"""Persistent trajectory analysis, reusable reports, and streaming pair exports.

The cache stores molecular arrays, not an atomic trajectory. Reports can run
without the original simulation files; structure exports require the source dump.
cache_analysis forwards pair_workers and pair_batch_size to indexed frame workers.
Frame parsing, reconstruction and screening run in workers; export_pairs can also
parallelize parsing and shell construction. TOR and final writes remain ordered.
"""
from collections.abc import Sequence
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


ARRAY_KEYS = ("species", "com_angstrom", "wrapped_com_angstrom", "box_lengths_angstrom", "cell_angstrom",
              "com_velocity_angstrom_ps", "net_force_eV_angstrom")
SCHEMA_VERSION = 2
SUPPORTED_SCHEMAS = {1, 2}


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


def _manifest(cache_dir):
    """Read a completed cache manifest without loading any analysis data."""
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Incomplete or missing analysis cache: {cache_dir}; use a fresh output directory")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_SCHEMAS or not set(manifest["arrays"]) <= set(ARRAY_KEYS):
        raise ValueError("Unsupported analysis cache schema")
    return manifest


def cache_analysis(directory, cache_dir, *, units, timestep_ps, pair_criteria=None,
                   compute_pairs=True, pair_workers=1, pair_batch_size=32,
                   progress=print, **options):
    """Analyze once or reuse an exactly matching completed cache.

    Units/timestep are explicit. All analyze_run options except progress are
    recorded. Changing worker/batch count permits reuse; changing selection,
    source size/mtime/path, or package source code requires a new cache directory.
    Initial metadata is fixed; partial directories never count as completed.
    On reuse only manifest.json is read before deciding; data load afterwards.

    pair_workers > 1 processes contiguous indexed chunks in spawned workers.
    pair_batch_size is source frames per task. Temporal unwrapping stays ordered
    in the parent process. write_report remains serial; export_pairs has separate
    workers/batch_size arguments. For Python scripts,
    call this function inside an if __name__ == "__main__" guard when using workers.
    A reused cache records the worker settings of its original computation.
    The returned (result, manifest) can be passed to write_report/export_pairs
    as preloaded=... to avoid reading the cache again.
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
        manifest = _manifest(cache_dir)
        if (manifest["configuration"] != config or manifest["sources"] != fingerprints
                or manifest["software"]["source_sha256"] != software["source_sha256"]):
            raise ValueError("Cache does not match source files, analysis settings, or package code; use a new --output directory")
        if progress:
            progress(f"Reusing completed cache: {cache_dir}")
        return load_analysis_cache(cache_dir)
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
    np.savez(cache_dir / "frames.npz", **_frame_columns(result["frames"]))
    pair_storage = _save_pairs(cache_dir, result)
    metadata = {k: v for k, v in result.items()
                if k not in ARRAY_KEYS and k not in {"frames", "pairs_by_frame", "pair_shifts_by_frame"}}
    metadata["pair_storage"] = pair_storage
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


def _frame_columns(rows):
    """Per-frame records as one numeric array per key."""
    keys = list(dict.fromkeys(k for row in rows for k in row))
    columns = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows])
        if values.dtype.kind not in "iufb":
            raise ValueError(f"Frame column {key} is not numeric; cannot store it compactly")
        columns[key] = values
    return columns


def _save_pairs(cache_dir, result):
    """Store pairs as molecule indices: frame offsets, (CPD, dienophile) index, shift.

    Every cached pair molecule is one of molecule_atom_ids (fixed membership),
    so 34 atom IDs per pair reduce to two indices. If a pair molecule is not an
    initial molecule (possible with the reference backend after a reaction),
    pairs are kept as compact JSON instead. Returns the storage label.
    """
    pairs, shifts = result.get("pairs_by_frame"), result.get("pair_shifts_by_frame")
    if pairs is None:
        return None
    lookup = {frozenset(map(int, m)): i for i, m in enumerate(result["molecule_atom_ids"])}
    counts = np.array([len(p) for p in pairs], dtype=np.int64)
    molecules = np.empty((int(counts.sum()), 2), dtype=np.int32)
    images = np.empty((int(counts.sum()), 3), dtype=np.int32)
    k = 0
    try:
        for frame_pairs, frame_shifts in zip(pairs, shifts):
            for (first, second), shift in zip(frame_pairs, frame_shifts):
                molecules[k] = lookup[frozenset(map(int, first))], lookup[frozenset(map(int, second))]
                images[k] = np.asarray(shift, dtype=int).ravel()
                k += 1
    except KeyError:
        text = json.dumps(_plain({"pairs_by_frame": pairs, "pair_shifts_by_frame": shifts}),
                          separators=(",", ":"), allow_nan=False)
        (cache_dir / "pairs.json").write_text(text + "\n", encoding="utf-8")
        return "json"
    np.save(cache_dir / "pair_frame_offsets.npy", np.concatenate(([0], np.cumsum(counts))))
    np.save(cache_dir / "pair_molecules.npy", molecules)
    np.save(cache_dir / "pair_shifts.npy", images)
    return "molecule_index"


class _CachedPairFrames(Sequence):
    """Read-only per-frame view of compactly cached pairs.

    Item f is that frame's list of [CPD atom IDs, dienophile atom IDs] pairs
    (atom order of molecule_atom_ids), or of [sx, sy, sz] shifts when
    shifts=True. Rows are materialized on access; nothing is held per frame.
    """

    def __init__(self, offsets, molecules, images, molecule_atom_ids, *, shifts=False):
        self._offsets, self._molecules, self._images = offsets, molecules, images
        self._ids, self._shifts = [list(map(int, m)) for m in molecule_atom_ids], shifts

    def __len__(self):
        return len(self._offsets) - 1

    def _frame(self, index):
        lo, hi = int(self._offsets[index]), int(self._offsets[index + 1])
        if self._shifts:
            return self._images[lo:hi].tolist()
        return [[self._ids[a], self._ids[b]] for a, b in self._molecules[lo:hi].tolist()]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._frame(i) for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._frame(index)

    def __iter__(self):
        return (self._frame(i) for i in range(len(self)))


def load_analysis_cache(cache_dir):
    """Load metadata and memory-map molecular NPY arrays; never unpickle objects.

    Returned result follows analyze_run's dictionary API, except paths are JSON
    strings and arrays are read-only memory maps. The source dump is not opened.
    Schema 2 caches return pairs_by_frame / pair_shifts_by_frame as read-only
    sequences materialized per frame; schema 1 caches (one large analysis.json)
    still load unchanged.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest = _manifest(cache_dir)
    result = json.loads((cache_dir / "analysis.json").read_text(encoding="utf-8"))
    for key in ARRAY_KEYS:
        result[key] = np.load(cache_dir / f"{key}.npy", mmap_mode="r", allow_pickle=False) if key in manifest["arrays"] else None
    if manifest["schema_version"] >= 2:
        with np.load(cache_dir / "frames.npz", allow_pickle=False) as columns:
            names = list(columns.files)
            values = [columns[name].tolist() for name in names]
        result["frames"] = [dict(zip(names, row)) for row in zip(*values)]
        storage = result.pop("pair_storage", None)
        if storage == "molecule_index":
            offsets = np.load(cache_dir / "pair_frame_offsets.npy", allow_pickle=False)
            # Small (14 bytes per pair); read fully so no file handle stays open.
            molecules = np.load(cache_dir / "pair_molecules.npy", allow_pickle=False)
            images = np.load(cache_dir / "pair_shifts.npy", allow_pickle=False)
            result["pairs_by_frame"] = _CachedPairFrames(offsets, molecules, images, result["molecule_atom_ids"])
            result["pair_shifts_by_frame"] = _CachedPairFrames(offsets, molecules, images, result["molecule_atom_ids"],
                                                               shifts=True)
        elif storage == "json":
            result.update(json.loads((cache_dir / "pairs.json").read_text(encoding="utf-8")))
        else:
            result["pairs_by_frame"] = result["pair_shifts_by_frame"] = None
    return result, manifest


def cached_frame_offsets(cache_dir, result):
    """Dump frame byte offsets, computed once and saved in the cache.

    frame_offsets.npy is valid for the dump fingerprint recorded in the
    manifest; callers already refuse to export when the dump has changed.
    The file is written atomically, so a partial write is never used.
    """
    from .frame_parallel import frame_offsets
    path = Path(cache_dir) / "frame_offsets.npy"
    if path.is_file():
        return np.load(path, allow_pickle=False)
    offsets = frame_offsets(result["files"]["dump"])
    temporary = path.with_name("frame_offsets.tmp.npy")
    np.save(temporary, offsets, allow_pickle=False)
    temporary.replace(path)
    return offsets


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
                 stability=False, rdf_bins=None, lag_statistics=None,
                 pair_lifetimes=True, frame_summary=True, progress=None, preloaded=None):
    """Compute a source-frame-window report from cached molecular data.

    startframe is inclusive and endframe exclusive, both zero-based source dump
    indices (not retained-row indices or timesteps). Defaults select the full
    cache. An unchanged cache can produce multiple reports in separate folders.

    MSD[k]=mean_{t,i}|R[t+k,i]-R[t,i]|^2 and correlations are mean vector dot
    products over all species molecules and valid origins. D=slope/6 is fitted
    only with an explicit fit_interval_ps. Lifetimes are recomputed within the
    window, so boundary censoring reflects that window. RDF uses instantaneous
    wrapped COMs; optional scalar ACF/batch errors delegate to stability.
    lag_statistics=() skips all lag calculations. Otherwise select msd, vacf,
    force-acf (None means all). pair_lifetimes and frame_summary independently
    control their files. These options change reports, not the underlying cache.
    preloaded=(result, manifest) from cache_analysis/load_analysis_cache for this
    cache_dir skips reading it again.
    """
    requested = ['msd', 'vacf', 'force-acf'] if lag_statistics is None else sorted(set(lag_statistics))
    if set(requested) - {'msd', 'vacf', 'force-acf'}:
        raise ValueError('Unknown lag statistic')
    if fit_interval_ps is not None and 'msd' not in requested:
        raise ValueError('Diffusion fitting requires msd in lag_statistics')
    if progress and preloaded is None:
        progress('Loading report cache')
    result, source = preloaded if preloaded is not None else load_analysis_cache(cache_dir)
    if isinstance(max_gap_frames, bool) or not isinstance(max_gap_frames, int) or max_gap_frames < 0:
        raise ValueError("max_gap_frames must be a nonnegative integer")
    if rdf_bins is not None and (isinstance(rdf_bins, bool) or not isinstance(rdf_bins, int) or rdf_bins < 1):
        raise ValueError("rdf_bins must be a positive integer")
    config = dict(cache=str(Path(cache_dir).resolve()), source=source, startframe=startframe, endframe=endframe,
                  max_lag_ps=max_lag_ps, fit_interval_ps=fit_interval_ps,
                  max_gap_frames=max_gap_frames, stability=stability, rdf_bins=rdf_bins, software=_software())
    config.update(lag_statistics=requested, pair_lifetimes=pair_lifetimes, frame_summary=frame_summary)
    # Check completed reports before performing potentially expensive statistics.
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        completed = _begin_product(output, config)[0]
        if progress:
            progress(f'Reusing completed report: {completed}')
        return completed
    if progress:
        progress(f"Report selection: lag statistics={requested or 'none'}, pair lifetimes={pair_lifetimes}, frame summary={frame_summary}")
    window = molecular_window_statistics(result, startframe=startframe, endframe=endframe,
                                         max_lag_ps=max_lag_ps, lag_statistics=requested, progress=progress)
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
    if frame_summary:
        if progress:
            progress(f'Writing frame summary: {len(rows)} frames')
        _csv(output / "frames.csv", rows)
    species_files = {}
    for index, (name, entry) in enumerate(window["species"].items()):
        filename = f"species_{index:03d}_lags.csv"
        keys = [key for key, value in entry.items() if isinstance(value, np.ndarray)]
        _csv(output / filename, [dict(zip(keys, values)) for values in zip(*(entry[k] for k in keys))])
        species_files[name] = {"file": filename, "n_molecules": entry["n_molecules"]}
    if fits:
        _csv(output / "diffusion.csv", fits)
    if pair_lifetimes and result["pairs_by_frame"] is not None:
        if progress:
            progress(f'Computing pair lifetimes over {len(rows)} frames')
        pairs = [p for p, keep in zip(result["pairs_by_frame"], window["frame_mask"]) if keep]
        events = md.get_pair_lifetime(pairs, times, max_gap_frames=max_gap_frames, progress=progress)
        # first_frame/last_frame count retained rows inside this window; the
        # source columns use the dump-frame numbering of every other output.
        source_frames = [r["frame"] for r in rows]
        _csv(output / "pair_lifetimes.csv",
             [dict(asdict(e), first_source_frame=source_frames[e.first_frame],
                   last_source_frame=source_frames[e.last_frame]) for e in events],
             fields=list(md.PairLifetime.__dataclass_fields__) + ["first_source_frame", "last_source_frame"])
        if progress:
            progress(f'Wrote {len(events)} pair lifetime records')
    elif pair_lifetimes and progress:
        progress('Pair lifetimes unavailable: cache has no pair analysis')
    if stability:
        from .stability import get_series_statistics
        if len(times) < 3:
            raise ValueError('Scalar stability requires at least three frames')
        summary = {}
        for key in rows[0]:
            if key not in {'frame', 'time_ps', 'timestep'}:
                if progress:
                    progress(f'Computing scalar stability: {key}')
                summary[key] = get_series_statistics([r[key] for r in rows], sample_interval_ps=times[1]-times[0])
        _json(output / "scalar_stability.json", summary)
    if rdf_bins is not None:
        from itertools import combinations_with_replacement
        from .stability import get_molecular_com_rdf
        cells = result.get("cell_angstrom")
        box = np.asarray(cells if cells is not None else result["box_lengths_angstrom"])[window["frame_mask"]]
        positions = result["wrapped_com_angstrom"][window["frame_mask"]]
        spacing = (min(md.cell_plane_spacings(c).min() for c in box) if box.ndim == 3 else box.min())
        edges = np.linspace(0, .999*spacing/2, rdf_bins+1)
        rdf_files = []
        for a, b in combinations_with_replacement(sorted(set(result['species'])), 2):
            if a == b and np.count_nonzero(result['species'] == a) < 2:
                continue
            if progress:
                progress(f'Computing COM RDF: {a} / {b}')
            rdf = get_molecular_com_rdf(positions, box, result["species"], species_a=a, species_b=b, edges=edges)
            filename = f"rdf_{len(rdf_files):03d}.csv"
            _csv(output / filename, [dict(r_angstrom=r, g_r=g) for r, g in zip(rdf["r_angstrom"], rdf["g_r"])])
            rdf_files.append(dict(species_a=a, species_b=b, file=filename))
        _json(output / "rdf_species.json", rdf_files)
    _json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "configuration": config,
                                     "time_window_ps": window["time_window_ps"], "frame_window": window["frame_window"],
                                     "species": species_files,
                                     "notices": result["notices"]})
    if progress:
        progress(f'Report complete: {output}')
    return output


def _export_identity(records, frame, first_structure, time_ps):
    """Attach original LAMMPS IDs, source rows and globally numbered structures.

    Frame-local indices are zero-based rows in the ID-sorted source ASE frame;
    local_index remains the exported structure's row. Neither replaces persistent
    global_index. Pair and cluster records receive the same structure_index.
    """
    ids = frame.atoms.arrays["global_index"]
    lookup = {int(atom_id): row for row, atom_id in enumerate(ids)}
    for candidate, record in records.items():
        rows = np.array([lookup[int(i)] for i in record["atom_ids"]], dtype=int)
        record.update(frame_local_index=rows,
                      lammps_id=frame.atoms.arrays["lammps_id"][rows],
                      structure_index=first_structure+candidate, candidate_index=candidate,
                      time_ps=time_ps)


def export_pairs(cache_dir, output_dir, *, shell_radius=None, startframe=0, endframe=None, progress=print,
                 workers=1, batch_size=32, preloaded=None):
    """Stream selected cached encounters into pair/shell Extended XYZ and an index.

    Select source frames in [startframe, endframe), defaulting to the entire
    cache. Bounds are zero-based dump indices, including when frames were strided.
    One structure per accepted pair per retained frame, not one per lifetime.
    Re-read source frames once; translations r'=r+image @ cell leave forces
    unchanged. Forces remain eV/Angstrom, positions Angstrom, stored velocities
    ASE units. Shell selection uses atom-to-pair distance then completes molecules.
    Source fingerprints must match the cache; no UMA model is evaluated.
    workers parallelizes indexed parsing, graph validation, shell construction
    and serialization; batch_size bounds selected frames per task. Final writes
    are ordered in the parent. Scientific output is independent of task order.
    Entries store forces in lammps_forces[:,0:3] (fx,fy,fz), local_index,
    global_index, lammps_id, frame_local_index and pair_role. structure_index
    links pairs.extxyz, clusters.extxyz and index.jsonl in identical order.
    frames.csv includes selected frames with zero accepted pairs.
    Only selected frames are parsed (each from its own byte range), using
    frame offsets saved in the cache after the first export.
    preloaded=(result, manifest) skips reading the cache again.
    """
    for name, value in (("workers", workers), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    result, source = preloaded if preloaded is not None else load_analysis_cache(cache_dir)
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
    count = 0
    frame_counts = []
    with (output / "pairs.extxyz").open("w", encoding="utf-8") as pair_stream, \
         (output / "index.jsonl").open("w", encoding="utf-8") as index_stream:
        from contextlib import ExitStack
        with ExitStack() as stack:
            cluster_stream = stack.enter_context((output / "clusters.extxyz").open("w", encoding="utf-8")) if shell_radius is not None else None
            from .frame_parallel import ordered_tasks
            if selected:
                if progress:
                    progress("Loading (or indexing once) source frame byte offsets for export...")
                offsets = cached_frame_offsets(cache_dir, result)
                reader_options = dict(units=options["units"], input_datafile=result["files"]["data"], type_symbols=mapping)

                def tasks():
                    first_structure = 0
                    indices = list(selected)
                    for offset in range(0, len(indices), batch_size):
                        batch = indices[offset:offset+batch_size]
                        if batch[-1]+1 >= len(offsets):
                            raise ValueError("Source dump ended before every selected frame was exported")
                        ranges = [(i, int(offsets[i]), int(offsets[i+1])) for i in batch]
                        subset = {i: selected[i] for i in batch}
                        yield (result["files"]["dump"], reader_options, ranges, subset, shell_radius, scale,
                               first_structure)
                        first_structure += sum(len(selected[i][1]) for i in batch)

                for pair_text, cluster_text, index_text, counts, count in ordered_tasks(
                        _export_frame_chunk, tasks(), workers):
                    pair_stream.write(pair_text)
                    index_stream.write(index_text)
                    if cluster_stream is not None:
                        cluster_stream.write(cluster_text)
                    frame_counts.extend(counts)
                    if progress:
                        progress(f"Exported {count} structures; completed {len(frame_counts)} selected frames")
    if _fingerprints(result["files"]) != source["sources"]:
        raise ValueError("Source files changed during export; output is incomplete")
    if len(frame_counts) != len(selected):
        raise ValueError("Source dump ended before every selected frame was exported")
    _csv(output / "frames.csv", frame_counts,
         fields=["frame", "timestep", "time_ps", "pair_count", "first_structure_index"])
    _json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "configuration": config,
                                     "structures": count, "pair_file": "pairs.extxyz",
                                     "cluster_file": "clusters.extxyz" if shell_radius is not None else None,
                                     "index": "index.jsonl", "frames": "frames.csv",
                                     "frames_processed": len(frame_counts),
                                     "force_array": "lammps_forces", "force_components": ["fx", "fy", "fz"],
                                     "force_units": "eV/Angstrom"})
    return output


def _export_frame_chunk(task):
    """Parse, validate, build shells and serialize one independent source chunk.

    Structure offsets are assigned from cached pair counts before dispatch, so
    all atom/structure IDs and file ordering are independent of worker scheduling.
    Forces are copied by source atom ID; no model is evaluated.
    """
    from . import graph
    from .geom_utils import (get_pair_positions_and_forces, get_pair_shell_clusters,
                             get_cluster_positions_and_forces, pair_data_to_extxyz)
    from .lammpsio import iter_lammps_frames
    from .pair_selection import PairCriteria, assess_prereactive_pairs, site_cache
    path, reader_options, ranges, selected, shell_radius, scale, count = task
    pair_stream, index_stream = StringIO(), StringIO()
    cluster_stream = StringIO() if shell_radius is not None else None
    frame_counts = []
    frames = (frame for index, lower, upper in ranges
              for frame in iter_lammps_frames(path, byte_range=(lower, upper), frame_offset=index, **reader_options))
    # Reactive sites are reused while molecules are unchanged; unwraps are
    # memoized within each frame only.
    with site_cache():
        for frame in frames:
            with graph.frame_cache():
                row, pairs, shifts = selected[frame.index]
                frame_counts.append(dict(frame=frame.index, timestep=frame.timestep,
                    time_ps=row["time_ps"], pair_count=len(pairs), first_structure_index=count))
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
                _export_identity(records, frame, count, row["time_ps"])
                buffer = StringIO()
                pair_data_to_extxyz(records, buffer)
                pair_stream.write(buffer.getvalue())
                assessments = assess_prereactive_pairs(frame.atoms, pairs, shifts, criteria=PairCriteria(), scale=scale, graph=bonded)
                if cluster_stream is not None:
                    clusters = get_pair_shell_clusters(frame.atoms, pairs, shifts, shell_radius=shell_radius,
                                                       scale=scale, graph=bonded, components=components)
                    buffer = StringIO()
                    cluster_records = get_cluster_positions_and_forces(frame.atoms, frame, clusters)
                    _export_identity(cluster_records, frame, count, row["time_ps"])
                    pair_data_to_extxyz(cluster_records, buffer)
                    cluster_stream.write(buffer.getvalue())
                for candidate, (pair, shift, assessment) in enumerate(zip(pairs, shifts, assessments)):
                    record = dict(structure_index=count, frame=frame.index, timestep=frame.timestep,
                                  time_ps=row["time_ps"], candidate=candidate, pair_atom_ids=pair,
                                  cpd_shift=shift, geometry=asdict(assessment.geometry) if assessment.geometry else None,
                                  geometry_errors=assessment.reasons)
                    index_stream.write(json.dumps(_plain(record), allow_nan=False) + "\n")
                    count += 1
    return (pair_stream.getvalue(), cluster_stream.getvalue() if cluster_stream is not None else None,
            index_stream.getvalue(), frame_counts, count)
