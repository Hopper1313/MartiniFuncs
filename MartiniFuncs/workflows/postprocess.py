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
import inspect
from io import StringIO
import json
from pathlib import Path
import platform

import numpy as np

from MartiniFuncs._checks import integer, plain as _plain
from MartiniFuncs.pairs import lifetimes
from MartiniFuncs.statistics import md
from MartiniFuncs.structure import periodic
from MartiniFuncs.chemistry import UNRECORDED_CACHE_PAIR_MODEL
from MartiniFuncs.pairs.model import get_collection, get_pair_model
from MartiniFuncs.trajectory.analysis import (analyze_run, molecular_window_statistics, resolve_run_files,
                                              select_frame_window)


ARRAY_KEYS = ("species", "com_angstrom", "wrapped_com_angstrom", "box_lengths_angstrom", "cell_angstrom",
              "com_velocity_angstrom_ps", "net_force_eV_angstrom", "cell_origin_angstrom",
              "com_nonaffine_angstrom", "orientation_axes")
SCHEMA_VERSION = 2
SUPPORTED_SCHEMAS = {1, 2}


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
    """Record installed version and a digest of every package source file, including editable installs.

    The digest covers all MartiniFuncs/**/*.py (paths relative to the package
    root), so any code change, in any subpackage, invalidates cache reuse.
    """
    try:
        package_version = version("MartiniFuncs")
    except PackageNotFoundError:
        package_version = "uninstalled"
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
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


def _export_sources(result, source, dump_file=None, data_file=None):
    """(dump, data) paths an export reads, checked against the cache's source fingerprints.

    By default the cached paths, which must be unchanged (path, size and
    modification time). dump_file reads a copy of the dump elsewhere, e.g. a
    cache made on a cluster and used on a laptop; the copy must have the cached
    dump's byte size (copies do not keep modification times). data_file
    overrides the data file the same way; without it, a cached data file that
    does not exist here is skipped, and the dump's element/mass columns give
    the species.
    """
    files = result["files"]
    if dump_file is None and data_file is None:
        if _fingerprints(files) != source["sources"]:
            raise ValueError("Source files changed since caching; refuse to export mismatched structures/forces "
                             "(a copy of the dump elsewhere can be given as dump_file / --dump-file)")
        return files["dump"], files["data"]
    dump = Path(dump_file if dump_file is not None else files["dump"]).expanduser().resolve()
    if not dump.is_file():
        raise ValueError(f"Dump not found: {dump}")
    expected = source["sources"]["dump"]["size"]
    if dump.stat().st_size != expected:
        raise ValueError(f"{dump} has {dump.stat().st_size} bytes, but the cached dump had {expected}; "
                         "it is not the same trajectory")
    if data_file is None:
        data_file = files["data"] if files["data"] is not None and Path(files["data"]).is_file() else None
    return str(dump), None if data_file is None else str(Path(data_file).expanduser().resolve())


def _manifest(cache_dir):
    """Read a completed cache manifest without loading any analysis data."""
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Incomplete or missing analysis cache: {cache_dir}; use a fresh output directory")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_SCHEMAS or not set(manifest["arrays"]) <= set(ARRAY_KEYS):
        raise ValueError("Unsupported analysis cache schema")
    return manifest


# analyze_run arguments that never enter the cache configuration: resolved source
# files are fingerprinted separately; workers/progress/offsets do not change results.
_UNRECORDED_OPTIONS = {"directory", "input_file", "dump_file", "data_file", "progress",
                       "pair_workers", "pair_batch_size", "frame_offsets"}


def _analysis_configuration(options):
    """JSON configuration of an analysis: every analyze_run setting, defaults filled in.

    Filling defaults makes an omitted option and the same value passed
    explicitly produce the same configuration. A PairModel instance is
    recorded by its registered name and the pair collection (stage 1, the only
    pair setting a cache depends on) by its fields, with the model's default
    when omitted. Raises ValueError for values that cannot be stored as JSON,
    before any analysis runs.
    """
    from MartiniFuncs.trajectory.analysis import analyze_run
    parameters = inspect.signature(analyze_run).parameters
    unknown = set(options) - set(parameters)
    if unknown:
        raise TypeError(f"Unknown analyze_run options: {sorted(unknown)}")
    config = {name: options.get(name, parameter.default) for name, parameter in parameters.items()
              if name not in _UNRECORDED_OPTIONS}
    # None, a registered name and a PairModel instance all resolve to the model's name, and an
    # omitted collection to that model's default, so equivalent requests share one configuration.
    if config["compute_pairs"]:
        config["pair_collection"] = asdict(get_collection(config["pair_model"], config["pair_collection"]))
        config["pair_model"] = get_pair_model(config["pair_model"]).name
    else:
        config["pair_collection"] = config["pair_model"] = None
    config = _plain(config)
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Analysis options cannot be recorded in the cache manifest: {exc}") from exc
    return config


def cache_analysis(directory, cache_dir, *, units, timestep_ps, pair_collection=None,
                   compute_pairs=True, pair_workers=1, pair_batch_size=32,
                   progress=print, **options):
    """Analyze once or reuse an exactly matching completed cache.

    Units/timestep are explicit. Every analyze_run setting is recorded with its
    default filled in (except progress, worker/batch counts and frame offsets).
    pair_collection (stage 1; None: the pair model's default) decides which
    pairs are collected and stored. Pair selection (PairCriteria) is not part
    of a cache: write_report/export_pairs apply it. Changing worker/batch count
    permits reuse; changing the collection or other analysis settings,
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
    config = _analysis_configuration(dict(options, units=units, timestep_ps=timestep_ps,
                                          compute_pairs=compute_pairs, pair_collection=pair_collection))
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
                         pair_collection=pair_collection, pair_workers=pair_workers,
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
    candidates = result.get("pair_candidates")
    if candidates is not None:
        for key in ("offsets", "molecules", "shifts", "metrics"):
            np.save(cache_dir / f"candidate_{key}.npy", candidates[key], allow_pickle=False)
    metadata = {k: v for k, v in result.items()
                if k not in ARRAY_KEYS and k not in {"frames", "pairs_by_frame", "pair_shifts_by_frame",
                                                     "pair_candidates"}}
    metadata["pair_storage"] = pair_storage
    metadata["pair_candidates_radius"] = None if candidates is None else candidates["reactive_radius"]
    if candidates is not None:
        metadata["pair_candidates_model"] = candidates["model"]
        metadata["pair_candidates_metrics"] = candidates["metric_names"]
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
    """Store pairs as molecule indices: frame offsets, (first, second member) index, shift.

    Every cached pair molecule is one of molecule_atom_ids (fixed membership),
    so a pair's atom IDs reduce to two indices. If a pair molecule is not an
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

    Item f is that frame's list of [first-member atom IDs, second-member atom IDs] pairs
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
        radius = result.pop("pair_candidates_radius", None)
        model = result.pop("pair_candidates_model", None) or UNRECORDED_CACHE_PAIR_MODEL
        names = result.pop("pair_candidates_metrics", None)
        result["pair_candidates"] = None
        if radius is not None:
            result["pair_candidates"] = {key: np.load(cache_dir / f"candidate_{key}.npy", allow_pickle=False)
                                         for key in ("offsets", "molecules", "shifts", "metrics")}
            model = get_pair_model(model)
            result["pair_candidates"].update(reactive_radius=float(radius), model=model.name,
                                             metric_names=list(names or model.metric_names))
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


def cached_frame_offsets(cache_dir, result, dump=None):
    """Dump frame byte offsets, computed once and saved in the cache.

    frame_offsets.npy is valid for the dump fingerprint recorded in the
    manifest; callers already refuse to export when the dump has changed.
    The file is written atomically, so a partial write is never used.
    """
    from MartiniFuncs.trajectory.parallel import frame_offsets
    path = Path(cache_dir) / "frame_offsets.npy"
    if path.is_file():
        return np.load(path, allow_pickle=False)
    offsets = frame_offsets(dump if dump is not None else result["files"]["dump"])
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
                 pair_lifetimes=True, frame_summary=True, progress=None, preloaded=None,
                 criteria=None):
    """Compute a source-frame-window report from cached molecular data.

    criteria (the cache's pair model's PairCriteria; stage 2) selects pairs
    from the collected candidates for the lifetimes, the per-frame
    selected_pairs column and pair_metric_histograms.csv; see
    pairs.candidates.select_pairs. None uses every collected pair.

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
    integer(max_gap_frames, "max_gap_frames", minimum=0)
    if rdf_bins is not None:
        integer(rdf_bins, "rdf_bins")
    config = dict(cache=str(Path(cache_dir).resolve()), source=source, startframe=startframe, endframe=endframe,
                  max_lag_ps=max_lag_ps, fit_interval_ps=fit_interval_ps,
                  max_gap_frames=max_gap_frames, stability=stability, rdf_bins=rdf_bins, software=_software())
    config.update(lag_statistics=requested, pair_lifetimes=pair_lifetimes, frame_summary=frame_summary,
                  criteria=asdict(criteria) if criteria is not None else None)
    from MartiniFuncs.pairs.candidates import candidate_mask, metric_histograms, select_pairs
    # Check completed reports before selecting pairs or computing statistics.
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        completed = _begin_product(output, config)[0]
        if progress:
            progress(f'Reusing completed report: {completed}')
        return completed
    selection = criteria is not None
    selected_pairs = select_pairs(result, criteria)[0] if result["pairs_by_frame"] is not None else None
    if progress:
        progress(f"Report selection: lag statistics={requested or 'none'}, pair lifetimes={pair_lifetimes}, frame summary={frame_summary}")
    window = molecular_window_statistics(result, startframe=startframe, endframe=endframe,
                                         max_lag_ps=max_lag_ps, lag_statistics=requested, progress=progress)
    rows = [r for r, keep in zip(result["frames"], window["frame_mask"]) if keep]
    if selection and selected_pairs is not None:
        rows = [dict(r, selected_pairs=len(p)) for r, p, keep in
                zip(result["frames"], selected_pairs, window["frame_mask"]) if keep]
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
    topology_events = result.get("topology_events") or []
    if topology_events:
        dt = result["settings"]["timestep"]
        _csv(output / "topology_events.csv",
             [dict(frame=e["frame"], timestep=e["timestep"], time_ps=e["timestep"] * dt,
                   persistence_frames=e["persistence_frames"], formulas=" + ".join(e["formulas"]),
                   changed_components=e["changed_components"]) for e in topology_events],
             fields=["frame", "timestep", "time_ps", "persistence_frames", "formulas", "changed_components"])
        if progress:
            progress(f"Wrote {len(topology_events)} topology (reaction) events")
    candidates = result.get("pair_candidates")
    if candidates is not None:
        # Distributions of the selected candidates inside this frame window.
        in_window = np.repeat(window["frame_mask"], np.diff(candidates["offsets"]))
        mask = in_window & candidate_mask(candidates, criteria)
        _csv(output / "pair_metric_histograms.csv", metric_histograms(candidates, mask),
             fields=["metric", "bin_low", "bin_high", "count"])
    if pair_lifetimes and result["pairs_by_frame"] is not None:
        if progress:
            progress(f'Computing pair lifetimes over {len(rows)} frames')
        pairs = [p for p, keep in zip(selected_pairs, window["frame_mask"]) if keep]
        episodes = lifetimes.get_pair_lifetime(pairs, times, max_gap_frames=max_gap_frames, progress=progress)
        # first_frame/last_frame count retained rows inside this window; the
        # source columns use the dump-frame numbering of every other output.
        source_frames = [r["frame"] for r in rows]
        _csv(output / "pair_lifetimes.csv",
             [dict(asdict(e), first_source_frame=source_frames[e.first_frame],
                   last_source_frame=source_frames[e.last_frame]) for e in episodes],
             fields=list(lifetimes.PairLifetime.__dataclass_fields__) + ["first_source_frame", "last_source_frame"])
        if progress:
            progress(f'Wrote {len(episodes)} pair lifetime records')
    elif pair_lifetimes and progress:
        progress('Pair lifetimes unavailable: cache has no pair analysis')
    if stability:
        from MartiniFuncs.statistics.stability import get_series_statistics
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
        from MartiniFuncs.statistics.stability import get_molecular_com_rdf
        cells = result.get("cell_angstrom")
        box = np.asarray(cells if cells is not None else result["box_lengths_angstrom"])[window["frame_mask"]]
        positions = result["wrapped_com_angstrom"][window["frame_mask"]]
        spacing = (min(periodic.cell_plane_spacings(c).min() for c in box) if box.ndim == 3 else box.min())
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
                                     "species": species_files, "topology_events": len(topology_events),
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
                 workers=1, batch_size=32, preloaded=None, criteria=None, dump_file=None, data_file=None):
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
    Pairs whose molecules are no longer separate components in a frame (e.g.
    after a reaction) are exported with membership_changed=true in index.jsonl
    and counted in the manifest; they never abort the export.
    criteria (the cache's pair model's PairCriteria) exports only the pairs it
    selects from the collected candidates (pairs.candidates.select_pairs);
    None exports every collected pair.
    Only selected frames are parsed (each from its own byte range), using
    frame offsets saved in the cache after the first export.
    preloaded=(result, manifest) skips reading the cache again. dump_file /
    data_file read copies of the source files elsewhere (see _export_sources).
    """
    integer(workers, "workers")
    integer(batch_size, "batch_size")
    result, source = preloaded if preloaded is not None else load_analysis_cache(cache_dir)
    if result["pairs_by_frame"] is None:
        raise ValueError("Cache contains no pair analysis; rerun analysis with pairs enabled")
    dump, data = _export_sources(result, source, dump_file, data_file)
    if shell_radius is not None and (not np.isfinite(shell_radius) or shell_radius <= 0):
        raise ValueError("shell_radius must be positive and finite")
    keep = select_frame_window(result, startframe=startframe, endframe=endframe)
    config = dict(cache=str(Path(cache_dir).resolve()), source=source, shell_radius=shell_radius,
                  startframe=startframe, endframe=endframe, software=_software())
    if dump_file is not None or data_file is not None:
        config.update(dump_file=dump, data_file=data)
    if criteria is not None:
        config.update(criteria=asdict(criteria))
    from MartiniFuncs.pairs.candidates import select_pairs
    chosen_pairs, chosen_shifts = select_pairs(result, criteria)
    output, create = _begin_product(output_dir, config)
    if not create:
        return output
    selected = {r["frame"]: (r, p, s)
                for r, p, s, retained in zip(result["frames"], chosen_pairs, chosen_shifts, keep) if retained}
    options = source["configuration"]
    mapping = options.get("type_symbols")
    if mapping is not None:
        mapping = {int(k): v for k, v in mapping.items()}
    scale = options.get("bond_scale", 1.3)
    count = flagged = 0
    frame_counts = []
    with (output / "pairs.extxyz").open("w", encoding="utf-8") as pair_stream, \
         (output / "index.jsonl").open("w", encoding="utf-8") as index_stream:
        from contextlib import ExitStack
        with ExitStack() as stack:
            cluster_stream = stack.enter_context((output / "clusters.extxyz").open("w", encoding="utf-8")) if shell_radius is not None else None
            from MartiniFuncs.trajectory.parallel import ordered_tasks
            if selected:
                if progress:
                    progress("Loading (or indexing once) source frame byte offsets for export...")
                offsets = cached_frame_offsets(cache_dir, result, dump)
                reader_options = dict(units=options["units"], input_datafile=data, type_symbols=mapping)

                def tasks():
                    first_structure = 0
                    indices = list(selected)
                    for offset in range(0, len(indices), batch_size):
                        batch = indices[offset:offset+batch_size]
                        if batch[-1]+1 >= len(offsets):
                            raise ValueError("Source dump ended before every selected frame was exported")
                        ranges = [(i, int(offsets[i]), int(offsets[i+1])) for i in batch]
                        subset = {i: selected[i] for i in batch}
                        yield (dump, reader_options, ranges, subset, shell_radius, scale,
                               first_structure, _pair_model(result).name)
                        first_structure += sum(len(selected[i][1]) for i in batch)

                for pair_text, cluster_text, index_text, counts, count, chunk_flagged in ordered_tasks(
                        _export_frame_chunk, tasks(), workers):
                    flagged += chunk_flagged
                    pair_stream.write(pair_text)
                    index_stream.write(index_text)
                    if cluster_stream is not None:
                        cluster_stream.write(cluster_text)
                    frame_counts.extend(counts)
                    if progress:
                        progress(f"Exported {count} structures; completed {len(frame_counts)} selected frames")
    if _export_sources(result, source, dump_file, data_file) != (dump, data):
        raise ValueError("Source files changed during export; output is incomplete")
    if len(frame_counts) != len(selected):
        raise ValueError("Source dump ended before every selected frame was exported")
    _csv(output / "frames.csv", frame_counts,
         fields=["frame", "timestep", "time_ps", "pair_count", "first_structure_index"])
    _json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "configuration": config,
                                     "structures": count, "structures_with_changed_membership": flagged,
                                     "pair_file": "pairs.extxyz",
                                     "cluster_file": "clusters.extxyz" if shell_radius is not None else None,
                                     "index": "index.jsonl", "frames": "frames.csv",
                                     "frames_processed": len(frame_counts),
                                     "force_array": "lammps_forces", "force_components": ["fx", "fy", "fz"],
                                     "force_units": "eV/Angstrom"})
    return output


def export_pair_trajectories(cache_dir, output_dir, *, criteria=None, startframe=0, endframe=None, every=1,
                             progress=print, workers=1, batch_size=32, preloaded=None, dump_file=None,
                             data_file=None):
    """Write one periodic Extended XYZ trajectory per collected pair, every selected frame flagged.

    Pairs are every distinct (first, second) molecule pair collected (inside the
    cache's reactive_radius) in at least one selected frame. Each pair's file
    holds every selected frame in [startframe, endframe), thinned to every
    every-th cached frame, including frames where the molecules are far apart.
    Per-frame info: pair_id, frame, timestep, time_ps, collected (inside the
    reactive radius), accepted (collected and passing criteria; None: every
    collected frame), membership_changed and the candidate metrics (NaN when
    not collected). Atoms carry the frame's cell with pbc, global_index,
    lammps_id, pair_role, lammps_forces, (when dumped) velocities, and
    accepted_binary (the frame's accepted flag as 0/1 on every atom); the
    candidate metrics are also per-atom arrays, so viewers can color by them.

    Positions are whole molecules. The first member follows its cached
    unwrapped COM (com_angstrom plus the first cell's center), so it moves
    continuously; the second member
    is the periodic image whose COM is nearest the first's. Flags and metrics
    come from the cache; only coordinates/forces are read from the dump.
    pairs.csv summarizes each pair (molecules, atom IDs, frame counts, file).
    all_pairs.extxyz holds, per frame, every pair's structure concatenated in
    pair_id order, with a per-atom pair_id array and frame info (frame,
    timestep, time_ps, n_pairs, n_collected, n_accepted). A molecule in two
    pairs appears twice, once placed with each partner.
    dump_file / data_file read copies of the source files elsewhere (see
    _export_sources); preloaded=(result, manifest) skips reading the cache.
    """
    from MartiniFuncs.pairs.candidates import candidate_mask, metric_columns
    integer(workers, "workers")
    integer(batch_size, "batch_size")
    integer(every, "every")
    if every < 1:
        raise ValueError("every must be a positive number of cached frames")
    result, source = preloaded if preloaded is not None else load_analysis_cache(cache_dir)
    candidates = result.get("pair_candidates")
    if candidates is None:
        raise ValueError("Cache has no pair candidates; rerun the analysis with pairs enabled")
    if result["com_angstrom"] is None:
        raise ValueError("Cache has no COM trajectory (non-periodic cell); pair trajectories need it")
    dump, data = _export_sources(result, source, dump_file, data_file)
    rows = np.flatnonzero(select_frame_window(result, startframe=startframe, endframe=endframe))[::every]
    model = _pair_model(result)
    config = dict(cache=str(Path(cache_dir).resolve()), source=source, startframe=startframe, endframe=endframe,
                  every=every, criteria=asdict(criteria) if criteria is not None else None, software=_software())
    if dump_file is not None or data_file is not None:
        config.update(dump_file=dump, data_file=data)

    # Candidate rows of the selected frames, and the distinct pairs among them.
    offsets, molecules = candidates["offsets"], candidates["molecules"]
    position = np.full(len(offsets) - 1, -1, dtype=np.int64)
    position[rows] = np.arange(len(rows))
    row_position = np.repeat(position, np.diff(offsets))
    selected = np.flatnonzero((row_position >= 0) & np.all(molecules >= 0, axis=1))
    skipped = int(np.count_nonzero((row_position >= 0) & np.any(molecules < 0, axis=1)))
    count = len(result["molecule_atom_ids"])
    keys = molecules[selected, 0].astype(np.int64) * count + molecules[selected, 1]
    unique = np.unique(keys)
    if not len(unique):
        raise ValueError("No collected pairs in the selected frames")
    pairs = np.column_stack((unique // count, unique % count))
    # status[t, p]: candidate row of pair p in selected frame t, -1 when not collected.
    status = np.full((len(rows), len(pairs)), -1, dtype=np.int64)
    status[row_position[selected], np.searchsorted(unique, keys)] = selected
    accepted = candidate_mask(candidates, criteria)

    output, create = _begin_product(output_dir, config)
    if not create:
        return output
    ids = [list(map(int, m)) for m in result["molecule_atom_ids"]]
    labels = [label.lower() for label in model.role_labels]
    names = [f"pair_{p:04d}_{labels[0]}{a}_{labels[1]}{b}.extxyz" for p, (a, b) in enumerate(pairs.tolist())]
    collected = status >= 0
    passed = collected & accepted[np.where(collected, status, 0)]
    frames = np.asarray([r["frame"] for r in result["frames"]])[rows]
    summary = []
    for p, (a, b) in enumerate(pairs.tolist()):
        hits, good = frames[collected[:, p]], frames[passed[:, p]]
        summary.append({"pair_id": p, f"{model.roles[0]}_molecule": a, f"{model.roles[1]}_molecule": b,
                        f"{model.roles[0]}_atom_ids": ids[a], f"{model.roles[1]}_atom_ids": ids[b],
                        "frames_written": len(rows), "frames_collected": len(hits), "frames_accepted": len(good),
                        "first_collected_frame": hits[0], "last_collected_frame": hits[-1],
                        "first_accepted_frame": good[0] if len(good) else None,
                        "last_accepted_frame": good[-1] if len(good) else None, "file": names[p]})
    if progress:
        progress(f"{len(pairs)} pairs x {len(rows)} frames = {len(pairs) * len(rows)} structures to write")
        progress("Loading (or indexing once) source frame byte offsets for export...")
    dump_offsets = cached_frame_offsets(cache_dir, result, dump)
    options = source["configuration"]
    mapping = options.get("type_symbols")
    if mapping is not None:
        mapping = {int(k): v for k, v in mapping.items()}
    reader_options = dict(units=options["units"], input_datafile=data, type_symbols=mapping)
    members = [(ids[a], ids[b]) for a, b in pairs.tolist()]
    metrics, columns = np.asarray(candidates["metrics"]), metric_columns(candidates)
    times = np.asarray([r["time_ps"] for r in result["frames"]])
    # com_angstrom starts from the COMs wrapped about the first cell's center, relative to that center.
    # Caches without cell_origin_angstrom predate it and assume the origin at 0.
    origin = result["cell_origin_angstrom"]
    center = (np.zeros(3) if origin is None else np.asarray(origin[0])) + np.asarray(result["cell_angstrom"][0]).sum(axis=0) / 2

    def tasks():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            if frames[start + len(chunk) - 1] + 1 >= len(dump_offsets):
                raise ValueError("Source dump ended before every selected frame was exported")
            per_frame = []
            for t, row in enumerate(chunk, start):
                # Metrics (NaN when not collected) and the cached COM of each pair's first member.
                state = status[t]
                values = np.where((state >= 0)[:, None], metrics[np.maximum(state, 0)], np.nan)
                per_frame.append((int(frames[t]), float(times[row]), state >= 0, passed[t], values,
                                  np.asarray(result["com_angstrom"][row])[pairs[:, 0]] + center))
            ranges = [(int(f), int(dump_offsets[f]), int(dump_offsets[f + 1])) for f in frames[start:start + len(chunk)]]
            yield (dump, reader_options, ranges, per_frame, members, columns,
                   model.role_labels, options.get("bond_scale", 1.3))

    from MartiniFuncs.trajectory.parallel import ordered_tasks
    done = 0
    for texts, together in ordered_tasks(_trajectory_chunk, tasks(), workers):
        for p, text in enumerate(texts):
            with (output / names[p]).open("a", encoding="utf-8") as stream:
                stream.write(text)
        with (output / "all_pairs.extxyz").open("a", encoding="utf-8") as stream:
            stream.write(together)
        done += batch_size
        if progress:
            progress(f"Wrote {min(done, len(rows))} of {len(rows)} frames for {len(pairs)} pairs")
    if _export_sources(result, source, dump_file, data_file) != (dump, data):
        raise ValueError("Source files changed during export; output is incomplete")
    _csv(output / "pairs.csv", summary)
    _json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "configuration": config,
                                     "pairs": len(pairs), "frames": len(rows), "summary": "pairs.csv",
                                     "all_pairs_file": "all_pairs.extxyz",
                                     "candidates_with_uninitial_molecules_skipped": skipped,
                                     "frame_info": ["pair_id", "frame", "timestep", "time_ps", "collected",
                                                    "accepted", "membership_changed", *columns],
                                     "force_array": "lammps_forces", "force_units": "eV/Angstrom"})
    return output


def _trajectory_chunk(task):
    """Serialize one chunk of selected source frames: (Extended XYZ texts by pair, all-pairs text)."""
    from ase import Atoms
    from MartiniFuncs.io.extxyz import write_extxyz
    from MartiniFuncs.io.lammps import iter_lammps_frames
    from MartiniFuncs.pairs.structures import _source_vectors
    from MartiniFuncs.structure import graph
    path, reader_options, ranges, per_frame, members, columns, role_labels, scale = task
    streams, combined = [StringIO() for _ in members], StringIO()
    frames = (frame for index, lower, upper in ranges
              for frame in iter_lammps_frames(path, byte_range=(lower, upper), frame_offset=index, **reader_options))
    for frame, (index, time_ps, collected, accepted, values, targets) in zip(frames, per_frame):
        if frame.index != index:
            raise ValueError(f"Read source frame {frame.index}, expected {index}")
        if "lammps_forces" not in frame.atoms.arrays:
            raise ValueError("Pair trajectory export requires source force columns")
        with graph.frame_cache():
            atoms, cell = frame.atoms, frame.atoms.cell.array
            bonded = graph.build_bond_graph(atoms, scale=scale)
            memberships = {frozenset(c) for c in graph.connected_components(bonded, atom_ids=graph.atom_ids(atoms))}
            masses = atoms.get_masses()
            together = None
            for p, (first, second) in enumerate(members):
                placed = []
                anchor = targets[p]
                for component in (first, second):
                    component_ids, xyz, _ = graph.place_component(atoms, bonded, component)
                    weights = masses[graph.local_indices(atoms, component_ids)]
                    center = weights @ xyz / weights.sum()
                    # The lattice translation that brings this COM nearest the anchor.
                    xyz = xyz + (anchor - center) - periodic.minimum_image(anchor - center, cell)
                    anchor = weights @ xyz / weights.sum()
                    placed.append((component_ids, xyz))
                atom_ids = np.concatenate([placed[0][0], placed[1][0]])
                local = graph.local_indices(atoms, atom_ids)
                vectors = _source_vectors(atoms, frame, atom_ids)
                structure = Atoms(np.asarray(atoms.get_chemical_symbols())[local].tolist(),
                                  positions=np.vstack([placed[0][1], placed[1][1]]), cell=cell, pbc=True)
                structure.new_array("global_index", atom_ids)
                structure.new_array("lammps_id", atoms.arrays["lammps_id"][local])
                structure.new_array("pair_role", np.array([role_labels[0]] * len(first) + [role_labels[1]] * len(second)))
                structure.new_array("lammps_forces", np.asarray(vectors["forces"], dtype=float))
                if "velocities" in vectors:
                    structure.new_array("velocities", np.asarray(vectors["velocities"], dtype=float))
                # The frame's accepted flag on every atom, as 0/1 (viewers such as OVITO color by numeric arrays).
                structure.new_array("accepted_binary", np.full(len(structure), int(accepted[p])))
                # The frame's candidate metrics, likewise on every atom (NaN when not collected).
                for name, value in zip(columns, values[p]):
                    structure.new_array(name, np.full(len(structure), float(value)))
                structure.info.update(pair_id=p, frame=index, timestep=frame.timestep, time_ps=time_ps,
                                      collected=bool(collected[p]), accepted=bool(accepted[p]),
                                      membership_changed=any(frozenset(c) not in memberships for c in (first, second)),
                                      **{name: float(v) for name, v in zip(columns, values[p])})
                write_extxyz([structure], streams[p])
                # Every pair of the frame in one structure, each atom labelled with its pair_id.
                part = structure.copy()
                part.info = {}
                part.new_array("pair_id", np.full(len(part), p))
                if together is None:
                    together = part
                else:
                    together.extend(part)
            together.info.update(frame=index, timestep=frame.timestep, time_ps=time_ps, n_pairs=len(members),
                                 n_collected=int(np.count_nonzero(collected)),
                                 n_accepted=int(np.count_nonzero(accepted)))
            write_extxyz([together], combined)
    return [stream.getvalue() for stream in streams], combined.getvalue()


def cache_pair_model(cache_dir):
    """The PairModel that produced a completed analysis cache (reads analysis.json only)."""
    _manifest(Path(cache_dir))
    metadata = json.loads((Path(cache_dir) / "analysis.json").read_text(encoding="utf-8"))
    return _pair_model(metadata)


def _pair_model(result):
    """The pair model of an analysis (chemistry.UNRECORDED_CACHE_PAIR_MODEL for caches that predate recording it)."""
    return get_pair_model((result.get("pair_analysis") or {}).get("model") or UNRECORDED_CACHE_PAIR_MODEL)


def _export_frame_chunk(task):
    """Parse, validate, build shells and serialize one independent source chunk.

    Structure offsets are assigned from cached pair counts before dispatch, so
    all atom/structure IDs and file ordering are independent of worker scheduling.
    Forces are copied by source atom ID; no model is evaluated.
    """
    from MartiniFuncs.structure import graph
    from MartiniFuncs.pairs.structures import (get_pair_positions_and_forces, get_pair_shell_clusters,
                                               get_cluster_positions_and_forces, pair_data_to_extxyz)
    from MartiniFuncs.io.lammps import iter_lammps_frames
    from MartiniFuncs.pairs.assessment import assess_pairs
    from MartiniFuncs.pairs.model import site_cache
    path, reader_options, ranges, selected, shell_radius, scale, count, model = task
    model = get_pair_model(model)
    pair_stream, index_stream = StringIO(), StringIO()
    cluster_stream = StringIO() if shell_radius is not None else None
    frame_counts, flagged = [], 0
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
                # A pair whose molecules are no longer separate components (e.g. they
                # have reacted) is still exported, flagged, so rare events never abort.
                memberships = {frozenset(c) for c in components}
                changed = [any(frozenset(c) not in memberships for c in pair) for pair in pairs]
                flagged += sum(changed)
                records = get_pair_positions_and_forces(frame.atoms, frame, pairs, shifts, bond_scale=scale, graph=bonded)
                _export_identity(records, frame, count, row["time_ps"])
                buffer = StringIO()
                pair_data_to_extxyz(records, buffer, role_labels=model.role_labels)
                pair_stream.write(buffer.getvalue())
                assessments = assess_pairs(frame.atoms, pairs, shifts, model, scale=scale, graph=bonded)
                if cluster_stream is not None:
                    clusters = get_pair_shell_clusters(frame.atoms, pairs, shifts, shell_radius=shell_radius,
                                                       scale=scale, graph=bonded, components=components)
                    buffer = StringIO()
                    cluster_records = get_cluster_positions_and_forces(frame.atoms, frame, clusters)
                    _export_identity(cluster_records, frame, count, row["time_ps"])
                    pair_data_to_extxyz(cluster_records, buffer, role_labels=model.role_labels)
                    cluster_stream.write(buffer.getvalue())
                for candidate, (pair, shift, assessment) in enumerate(zip(pairs, shifts, assessments)):
                    record = dict(structure_index=count, frame=frame.index, timestep=frame.timestep,
                                  time_ps=row["time_ps"], candidate=candidate, pair_atom_ids=pair,
                                  **{f"{model.roles[0]}_shift": shift}, geometry=asdict(assessment.geometry) if assessment.geometry else None,
                                  geometry_errors=(assessment.error,) if assessment.error else (),
                                  membership_changed=changed[candidate])
                    index_stream.write(json.dumps(_plain(record), allow_nan=False) + "\n")
                    count += 1
    return (pair_stream.getvalue(), cluster_stream.getvalue() if cluster_stream is not None else None,
            index_stream.getvalue(), frame_counts, count, flagged)
