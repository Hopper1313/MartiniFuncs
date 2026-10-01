"""Read LAMMPS one-line thermo tables from stdout .out files or logs.

No LAMMPS execution, interpolation, unit guessing, or merging of repeated steps.
Multiline/YAML thermo formats are not supported. Original column names and table
indices are retained alongside metal-unit aliases. Energy normalization remains
whatever thermo_modify norm requested; parsing cannot establish its physical meaning.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd


ALIASES = {"Step": "timestep", "Time": "thermo_time_ps", "Temp": "temperature_K",
           "Press": "pressure_bar", "PotEng": "potential_energy_eV",
           "KinEng": "kinetic_energy_eV", "TotEng": "total_energy_eV",
           "Volume": "thermo_volume_angstrom3", "Density": "thermo_density_g_cm3",
           **{key: key.lower()+"_bar" for key in ("Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz")}}


def _header(line):
    fields = line.split()
    return fields if len(fields) > 1 and fields[0] == "Step" and not any(
        re.fullmatch(r"[+-]?\d+(\.\d*)?", field) for field in fields[1:]) else None


def discover_thermo_file(directory, *, thermo_file=None):
    """Find a unique .out/.log/log* containing a Step table; None if absent.

    Search the directory first, then recursively only if no matching table was
    found. Multiple candidates require an explicit relative/absolute filename.
    File content is read; extensions alone never establish that it is thermo data.
    """
    directory = Path(directory).expanduser().resolve()
    if thermo_file is not None:
        path = Path(thermo_file).expanduser()
        path = path if path.is_absolute() else directory/path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for recursive in (False, True):
        glob = directory.rglob if recursive else directory.glob
        candidates = sorted({p for pattern in ("*.out", "*.log", "log*") for p in glob(pattern) if p.is_file()})
        found = []
        for path in candidates:
            with path.open(encoding="utf-8", errors="replace") as stream:
                if any(_header(line) for line in stream):
                    found.append(path)
        if len(found) > 1:
            raise ValueError(f"Multiple thermo outputs found; set thermo_file explicitly: {found}")
        if found:
            return found[0]
    return None


def read_lammps_thermo(path, *, units):
    """Return all one-line Step tables as a DataFrame, preserving each header block.

    units='metal' must be supplied: Temp is K, Press/Pij bar, Time ps, energies
    eV (possibly normalized per atom), Volume Angstrom^3 and Density g/cm^3.
    Unknown/custom columns remain under their LAMMPS names with no inferred units.
    thermo_block is a zero-based header occurrence, source_line a one-based line.
    Numeric rows must match the header width and be finite; malformed rows raise.
    Text warnings between numeric rows are ignored. Loop/ERROR terminates a table.
    Fortran D exponents are supported. Step is parsed as an integer, never float.
    """
    if units != "metal":
        raise ValueError("Thermo aliases currently require explicit units='metal'")
    rows, header, block = [], None, -1
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            current = _header(line)
            if current:
                if len(set(current)) != len(current) or set(current) & {"thermo_block", "source_line"}:
                    raise ValueError(f"Duplicate/reserved thermo headings at line {line_number}")
                header, block = current, block+1
                continue
            if not fields or header is None:
                continue
            if fields[0] in {"Loop", "ERROR:"}:
                header = None
                continue
            if not re.fullmatch(r"[+-]?\d+", fields[0]):
                continue
            if len(fields) != len(header):
                raise ValueError(f"Malformed thermo row at line {line_number}: expected {len(header)} columns")
            try:
                values = [int(fields[0])] + [float(v.replace("D", "E").replace("d", "e")) for v in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid thermo number at line {line_number}") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite thermo row at line {line_number}")
            rows.append(dict(zip(header, values), thermo_block=block, source_line=line_number))
    if not rows:
        raise ValueError("No one-line LAMMPS Step thermo records found")
    table = pd.DataFrame(rows)
    for source, alias in ALIASES.items():
        if source in table:
            if alias in table:
                raise ValueError(f"Thermo column conflicts with alias {alias}")
            table[alias] = table[source]
    table.attrs.update(source=str(Path(path).resolve()), units=units,
                       energy_normalization="as printed; inspect thermo_modify norm")
    return table


def _integer_ids(values, name):
    """Accept integer values, decimal strings and exactly integral safe floats.

    Floating-point IDs at magnitude >=2**53 are rejected: integer identity may
    already have been lost. No rounding of fractional or missing IDs is allowed.
    """
    converted = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} contains a boolean, not an integer ID")
        if isinstance(value, (int, np.integer)):
            integer = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            integer = int(value)
        elif isinstance(value, (float, np.floating)) and np.isfinite(value) and abs(value) < 2**53 and value == np.floor(value):
            integer = int(value)
        else:
            raise ValueError(f"{name} requires finite exact integer IDs; found {value!r}")
        if not -(2**63) <= integer < 2**63:
            raise ValueError(f"{name} is outside the supported signed 64-bit integer range")
        converted.append(integer)
    return np.asarray(converted, dtype=np.int64)


def _source_steps(dump_file, frame_ids):
    """Read exact ITEM: TIMESTEP integers for requested source-frame indices.

    Stream text through the last requested frame without parsing atom arrays.
    Preserve caller order, including strided/reordered frame selections.
    """
    ids = _integer_ids(frame_ids, "frame")
    if not len(ids) or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("Recovery requires nonempty, unique nonnegative source frame indices")
    needed, steps, index = set(ids.tolist()), {}, -1
    with Path(dump_file).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() != "ITEM: TIMESTEP":
                continue
            index += 1
            raw = next(stream, "").strip()
            if not re.fullmatch(r"[+-]?\d+", raw):
                raise ValueError(f"Invalid timestep header in source frame {index}")
            if index in needed:
                steps[index] = int(raw)
            if len(steps) == len(needed):
                break
    if len(steps) != len(needed):
        raise ValueError(f"Source dump lacks requested frame indices: {sorted(needed-set(steps))[:10]}")
    return _integer_ids([steps[int(i)] for i in ids], "dump timestep")


def attach_thermo(frames, thermo, *, block=None, dump_file=None, on_conflict="prefix"):
    """Left-join thermo aliases to frame records by exact integer timestep.

    Repeated thermo steps are ambiguous (e.g. restarts/run boundaries): select a
    header block explicitly. Unmatched frames receive NaN; no interpolation or
    forward fill. Keep only aliases to avoid treating provenance/custom columns
    as physical observables. If Time exists, verify matching frame time_ps with
    rtol=1e-6, atol=1e-8 ps (printed precision). Return (joined, coverage).
    If an older result lacks timestep, supply dump_file to recover exact integers
    from its source-frame indices and dump headers. No time/dt inference occurs.
    Integral float/string IDs are accepted after lossless validation.

    A thermo alias that already names a frame column (e.g. temperature_K, which
    analyze_run computes when degrees_of_freedom is set) is joined as
    thermo_<alias> with on_conflict="prefix" (default), so neither value is
    overwritten; on_conflict="error" raises instead. coverage["renamed"] maps
    each renamed alias to its joined column.
    """
    frames = pd.DataFrame(frames).copy()
    selected = (thermo if block is None else thermo.loc[thermo["thermo_block"] == block]).copy()
    if selected.empty:
        raise ValueError("No thermo rows in the requested block")
    recovered = "timestep" not in frames
    if recovered:
        if dump_file is None or "frame" not in frames:
            raise ValueError("Frame records lack timestep. Supply dump_file and source frame indices to recover exact steps from the dump.")
        frames["timestep"] = _source_steps(dump_file, frames["frame"])
    for name, table in (("frame", frames), ("thermo", selected)):
        if "timestep" not in table:
            raise ValueError(f"{name} records lack timestep")
        table["timestep"] = _integer_ids(table["timestep"], f"{name} timestep")
        if table["timestep"].duplicated().any():
            raise ValueError("Repeated timesteps are ambiguous; select one thermo block / trajectory segment")
    if on_conflict not in {"prefix", "error"}:
        raise ValueError("on_conflict must be 'prefix' or 'error'")
    columns = [c for c in ALIASES.values() if c in selected]
    overlap = sorted((set(columns)-{"timestep"}) & set(frames.columns))
    if overlap and on_conflict == "error":
        raise ValueError(f"Thermo columns would overwrite frame data: {overlap}")
    renamed = {c: "thermo_" + c for c in overlap}
    if set(renamed.values()) & set(frames.columns):
        raise ValueError(f"Thermo columns would overwrite frame data: {sorted(set(renamed.values()) & set(frames.columns))}")
    joined = frames.merge(selected[columns].rename(columns=renamed), on="timestep", how="left", sort=False,
                          validate="one_to_one", indicator=True)
    matches = joined.pop("_merge") == "both"
    if not matches.any():
        raise ValueError("No dump timesteps match thermo records; check files and selected block")
    if "thermo_time_ps" in joined and "time_ps" in joined:
        timed = matches & joined["thermo_time_ps"].notna()
        if not np.allclose(joined.loc[timed, "time_ps"], joined.loc[timed, "thermo_time_ps"], rtol=1e-6, atol=1e-8):
            raise ValueError("Thermo Time disagrees with dump time_ps; check timestep/reset conventions")
    return joined, {"matched_frames": int(matches.sum()), "unmatched_frames": int((~matches).sum()),
                    "thermo_rows": len(selected), "block": block, "timesteps_recovered_from_dump": recovered,
                    "renamed": renamed}
