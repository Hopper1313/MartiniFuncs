"""Bounded-memory text LAMMPS readers with explicit units and stable atom IDs.

New code should use iter_lammps_frames or read_lammps_frame. Raw columns remain
in source units; source forces attached to Atoms use eV/Angstrom. No simulation
calculator or placeholder energy is attached.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import shlex
import warnings

import numpy as np
import periodictable as pt
from ase import Atoms
from ase.calculators.lammps import convert
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsrun import construct_cell


@dataclass
class LammpsFrame:
    """Synchronized frame with ID-sorted raw columns in source units.

    atoms.arrays['lammps_forces'] contains source forces in eV/Angstrom, not
    recalculated forces after geometry changes. Image flags are retained, not
    automatically applied to wrapped coordinates.
    """
    atoms: Atoms
    columns: dict[str, np.ndarray]
    box_bounds: dict
    index: int
    timestep: int
    units: str
    id_offset: int = 1

    def as_legacy_tuple(self):
        """Return historical raw values without unit conversion."""
        return self.timestep, len(self.atoms), self.box_bounds, _records(self.columns, self.id_offset)


def normalize_type_symbols(type_symbols):
    """Return {LAMMPS type: element symbol}, the one type_symbols convention.

    Accepts a mapping such as {1: "C", 2: "H"} (keys may be strings, e.g. from
    JSON) or a sequence in type order, ["C", "H"], where element k is type k+1
    (the order written by write_initial_data and render_lammps_input).
    """
    if type_symbols is None:
        return None
    if isinstance(type_symbols, str):
        raise TypeError("type_symbols must be a mapping or a sequence of symbols, not a string")
    if hasattr(type_symbols, "items"):
        mapping = {int(k): str(v) for k, v in type_symbols.items()}
    else:
        mapping = {i: str(v) for i, v in enumerate(type_symbols, 1)}
    if not mapping or any(k <= 0 for k in mapping) or any(not v for v in mapping.values()):
        raise ValueError("type_symbols needs positive LAMMPS types and nonempty symbols")
    return mapping


def type_symbol_list(type_symbols):
    """Symbols in LAMMPS type order 1..n, from a sequence or a {type: symbol} mapping.

    A mapping must cover exactly types 1..n (data files number types contiguously).
    """
    mapping = normalize_type_symbols(type_symbols)
    if mapping is None:
        return None
    if sorted(mapping) != list(range(1, len(mapping) + 1)):
        raise ValueError("type_symbols must define types 1..n exactly once")
    return [mapping[i] for i in range(1, len(mapping) + 1)]


def mass_to_symbol(target_mass):
    """Nearest elemental mass; prefer explicit mappings for nonstandard masses.

    Choose Z minimizing |element_mass[Z]-target_mass|, with masses in amu.
    This nearest-mass heuristic is not reliable for isotope or coarse-grain mapping.
    """
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("Atomic mass must be positive and finite")
    return min((e for e in pt.elements if e.mass is not None),
               key=lambda e: abs(e.mass - target_mass)).symbol


_FRAME_MARKER = b"ITEM: TIMESTEP"
_READ_BLOCK = 1 << 24


def _iter_blocks(path, byte_range=None):
    """Yield one text frame at a time without losing boundary markers.

    Frames are split at lines equal to "ITEM: TIMESTEP" using bytes.find on
    16 MiB reads (no per-line Python work); CRLF is normalized to LF.
    byte_range=(start, stop) reads only that half-open byte range.
    """
    start, stop = (0, None) if byte_range is None else byte_range
    pattern = b"\n" + _FRAME_MARKER
    with open(path, "rb") as stream:
        stream.seek(start)
        remaining = None if stop is None else stop - start
        buffer, checked = b"", False
        while True:
            size = _READ_BLOCK if remaining is None else min(_READ_BLOCK, remaining)
            chunk = stream.read(size) if size > 0 else b""
            if remaining is not None:
                remaining -= len(chunk)
            eof = not chunk or remaining == 0
            buffer += chunk
            if not checked:
                stripped = buffer.lstrip()
                if not stripped:
                    if eof:
                        return
                    continue
                if len(stripped) <= len(_FRAME_MARKER) and not eof:
                    continue
                if not stripped.startswith(_FRAME_MARKER):
                    line_number = buffer[:len(buffer) - len(stripped)].count(b"\n") + 1
                    raise ValueError(f"{path}:{line_number}: expected ITEM: TIMESTEP")
                buffer, checked = stripped, True
            cut, search = 0, 1
            while True:
                hit = buffer.find(pattern, search)
                if hit < 0:
                    break
                end = hit + len(pattern)
                if end >= len(buffer) and not eof:
                    break   # the byte after the marker decides; read more first
                if buffer[end:end + 1] in (b"\n", b"\r", b""):
                    yield buffer[cut:hit + 1].decode("utf-8").replace("\r\n", "\n")
                    cut = hit + 1
                search = hit + 1
            buffer = buffer[cut:]
            if eof:
                if buffer.strip():
                    yield buffer.decode("utf-8").replace("\r\n", "\n")
                return


def _parse_block(text):
    lines = iter(text.splitlines())

    def take():
        try:
            return next(lines)
        except StopIteration as exc:
            raise ValueError("Truncated frame") from exc

    if take().strip() != "ITEM: TIMESTEP":
        raise ValueError("Expected ITEM: TIMESTEP")
    timestep = int(take())
    count = bounds = declared_units = None
    seen = set()
    for line in lines:
        if not line.strip():
            continue
        section = line.split()[1:3]
        key = section[0] if section else ""
        if key in seen:
            raise ValueError(f"Duplicate header: {line}")
        seen.add(key)
        if line.strip() == "ITEM: UNITS":
            declared_units = take().strip()
        elif line.strip() == "ITEM: TIME":
            if not np.isfinite(float(take())):
                raise ValueError("Non-finite simulation time")
        elif line.strip() == "ITEM: NUMBER OF ATOMS":
            count = int(take())
            if count <= 0:
                raise ValueError("Atom count must be positive")
        elif line.startswith("ITEM: BOX BOUNDS"):
            labels = line.split()[3:]
            rows = [list(map(float, take().split())) for _ in range(3)]
            width = 4 if labels[:2] == ["abc", "origin"] else 3 if "xy" in labels else 2
            if any(len(row) != width for row in rows) or not np.isfinite(rows).all():
                raise ValueError("Invalid BOX BOUNDS rows")
            bounds = {"header": labels, "rows": rows}
            cell, origin = _raw_cell(labels, rows)
            bounds.update(origin=origin.tolist(), cell=cell.tolist())
            if width != 4:
                bounds.update(dict(zip(("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"),
                                       np.asarray(rows)[:, :2].ravel().tolist())))
                bounds.update(dict(zip(("xy", "xz", "yz"),
                                       [row[2] for row in rows] if width == 3 else [0., 0., 0.])))
        elif line.startswith("ITEM: ATOMS "):
            if count is None or bounds is None:
                raise ValueError("NUMBER OF ATOMS and BOX BOUNDS must precede ATOMS")
            names = line.split()[2:]
            if "id" not in names or len(set(names)) != len(names):
                raise ValueError("ATOMS requires unique column names including 'id'")
            lines_text = [take() for _ in range(count)]
            columns = _atom_columns_fast(lines_text, names)
            if columns is None:
                columns = _atom_columns_strict(lines_text, names)
            ids = columns["id"]
            if np.any(ids <= 0) or len(np.unique(ids)) != count:
                raise ValueError("Atom IDs must be positive and unique")
            if "type" in columns and np.any(columns["type"] <= 0):
                raise ValueError("Atom types must be positive")
            for group in [("x", "y", "z"), ("xs", "ys", "zs"), ("xu", "yu", "zu"),
                          ("xsu", "ysu", "zsu"), ("fx", "fy", "fz"),
                          ("vx", "vy", "vz"), ("ix", "iy", "iz")]:
                present = [name in columns for name in group]
                if any(present) and not all(present):
                    raise ValueError(f"Incomplete vector columns: {group}")
            if any(line.strip() for line in lines):
                raise ValueError("Unexpected content after atom rows")
            order = np.argsort(ids)
            return timestep, {k: v[order] for k, v in columns.items()}, bounds, declared_units
        else:
            raise ValueError(f"Unsupported header: {line}")
    raise ValueError("Frame contains no ATOMS section")


def _column_type(name):
    """Python type of a dump column: element is str, IDs/types/images/i_ are int."""
    if name == "element":
        return str
    if name in {"id", "type", "mol", "ix", "iy", "iz"} or name.startswith(("i_", "i2_")):
        return int
    return float


def _atom_columns_strict(lines, names):
    """Row-by-row parse with precise errors; the reference for _atom_columns_fast."""
    rows = []
    for row_index, line in enumerate(lines):
        values = line.split()
        if len(values) != len(names):
            raise ValueError(f"Atom row {row_index + 1}: expected {len(names)} columns, got {len(values)}")
        rows.append(values)
    columns = {}
    for i, name in enumerate(names):
        dtype = _column_type(name)
        values = np.asarray([row[i] for row in rows], dtype=dtype)
        if dtype is not str and not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {name}")
        columns[name] = values
    return columns


def _atom_columns_fast(lines, names):
    """Single C-level parse of numeric columns, or None to use the strict parser.

    Only taken when every row has exactly len(names)-1 single spaces and no
    tabs, so loadtxt (which ignores surplus columns) sees exactly one value per
    name. Integer columns parse as strictly as int(); floats are bit-identical
    to float(). Any anomaly (non-finite, bad token) defers to the strict parser,
    which raises the detailed error.
    """
    text = "\n".join(lines)
    if "\t" in text or text.count(" ") != len(lines) * (len(names) - 1):
        return None
    numeric = [k for k, name in enumerate(names) if _column_type(name) is not str]
    dtype = [(names[k], np.int64 if _column_type(names[k]) is int else np.float64) for k in numeric]
    try:
        data = np.loadtxt(StringIO(text), usecols=numeric, dtype=dtype, ndmin=1, comments=None)
    except ValueError:
        return None
    if data.shape != (len(lines),):
        return None
    columns = {}
    for k, name in enumerate(names):
        if _column_type(name) is str:
            columns[name] = np.asarray([line.split(" ", k + 1)[k] for line in lines])
        else:
            values = np.ascontiguousarray(data[name])
            if values.dtype.kind == "f" and not np.isfinite(values).all():
                return None
            columns[name] = values
    return columns


def _raw_cell(labels, rows):
    """Cell rows and origin (LAMMPS lo corner) in source units from BOX BOUNDS.

    Restricted triclinic bounds are bounding-box values; ase.io.lammpsrun.
    construct_cell removes the tilts. "abc origin" boxes give rows directly.
    """
    rows = np.asarray(rows, dtype=float)
    if labels[:2] == ["abc", "origin"]:
        return rows[:, :3].copy(), rows[:, 3].copy()
    offdiag = rows[:, 2] if rows.shape[1] > 2 else np.zeros(3)
    if rows.shape[1] > 2 and len(labels) >= 3:
        offdiag = offdiag[[labels.index(i) for i in ("xy", "xz", "yz")]]
    return construct_cell(rows[:, :2].flatten(), offdiag)


def _cell_from_bounds(bounds, units):
    """ASE cell, origin (celldisp) and PBC from parsed BOX BOUNDS, as ASE's dump reader does.

    Orthogonal/restricted-triclinic bounds go through ase.io.lammpsrun.construct_cell
    (bounding box minus tilts); "abc origin" boxes give rows directly.
    PBC comes from the last three boundary labels ("p" means periodic).
    """
    labels = bounds["header"]
    pbc = ["p" in d.lower() for d in labels[-3:]] if len(labels) >= 3 else [False] * 3
    cell, celldisp = _raw_cell(labels, bounds["rows"])
    return convert(cell, "distance", units, "ASE"), convert(celldisp, "distance", units, "ASE"), pbc


def _atoms_from_columns(columns, symbols, bounds, units):
    """Build Atoms from parsed, ID-sorted columns with ASE's dump conventions.

    Mirrors ase.io.lammpsrun: positions from x, else xs (scaled), else xu,
    else xsu (scaled); velocities/forces/charges unit-converted; f_/v_/c_/d_
    columns become float arrays, i_ columns int arrays and type an int array.
    Forces are returned rather than attached as a calculator. Avoids ASE
    re-parsing the frame text.
    """
    cell, celldisp, pbc = _cell_from_bounds(bounds, units)

    def vector(names, quantity=None):
        if not all(name in columns for name in names):
            return None
        values = np.column_stack([columns[name] for name in names]).astype(float)
        return convert(values, quantity, units, "ASE") if quantity else values

    quaternions = vector(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])
    if "x" in columns:
        geometry = dict(positions=vector(["x", "y", "z"], "distance"))
    elif "xs" in columns:
        geometry = dict(scaled_positions=vector(["xs", "ys", "zs"]))
    elif "xu" in columns:
        geometry = dict(positions=vector(["xu", "yu", "zu"], "distance"))
    elif "xsu" in columns:
        geometry = dict(scaled_positions=vector(["xsu", "ysu", "zsu"]))
    else:
        raise ValueError("No atomic positions found in LAMMPS output")
    if quaternions is not None and "positions" not in geometry:
        raise ValueError("Quaternion dumps require unscaled positions")
    atoms = Atoms(symbols=list(symbols), pbc=pbc, celldisp=celldisp, cell=cell, **geometry)
    if quaternions is not None:
        atoms.new_array("quaternions", quaternions, dtype=float)
    if "q" in columns:
        atoms.set_initial_charges(convert(columns["q"].astype(float), "charge", units, "ASE"))
    for name, values in columns.items():
        if name.startswith(("f_", "v_", "d_", "d2_")) or (name.startswith("c_") and not name.startswith("c_q[")):
            atoms.new_array(name, values, dtype="float")
        elif name.startswith(("i_", "i2_")) or name == "type":
            atoms.new_array(name, values, dtype="int")
    return atoms, vector(["vx", "vy", "vz"], "velocity"), vector(["fx", "fy", "fz"], "force")


def _records(columns, id_offset):
    """Map each ID-sorted raw row to key columns['id'][row]-id_offset.
    Column values remain in source units; the key is not a local ASE row.
    """
    records = {}
    for row, atom_id in enumerate(columns["id"]):
        record = {name: values[row].item() for name, values in columns.items()}
        record["global_index"] = int(atom_id) - id_offset
        records[record["global_index"]] = record
    return records


def _select(iterator, index):
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer; use iter_lammps_frames for iteration")
    try:
        if index < 0:
            tail = deque(iterator, maxlen=int(-index))
            if len(tail) == -index:
                return tail[0]
        else:
            for i, value in enumerate(iterator):
                if i == index:
                    return value
    finally:
        iterator.close()
    raise IndexError(f"Frame index {index} is out of range")


def iter_lammps_frames(path, *, units, input_datafile=None, type_symbols=None,
                       id_offset=1, wrap=False, byte_range=None, frame_offset=0):
    """Stream synchronized frames with explicit LAMMPS units.

    Each bounded frame is parsed for raw metadata and by ASE for geometry and
    unit conversion. Species priority: dump elements, explicit type mapping,
    data-file mapping, then dump masses. Conflicting element/type mappings fail.
    Reduced LJ units are unsupported. Coordinates include the box origin.
    type_symbols is {type: symbol} or a type-order list (see normalize_type_symbols).

    Raw columns stay in source units; ASE conversion applies
    q_ASE=q_source*unit_factor(source, ASE) to distances, masses, velocities
    and forces. Scaled coordinates obey r=s @ cell+origin; optional wrapping
    acts on r-origin. global_index=lammps_id-id_offset; image flags are stored,
    not added to positions. arrays['image_flags'] always refers to the stored
    positions: r + n @ cell is the LAMMPS-unwrapped position. It equals the
    dumped ix/iy/iz for x/xs columns, is zero for already-unwrapped xu/xsu
    columns, and includes the lattice shift applied by wrap=True. The raw
    ix/iy/iz values stay in frame.columns. box_bounds keeps the dump's own
    xlo..zhi (LAMMPS bounding-box values for triclinic boxes) and adds the
    unambiguous origin and cell (rows a, b, c), in source units.
    byte_range is an internal half-open range from a frame-offset index;
    frame_offset preserves source frame numbering when reading that range.
    """
    if units not in {"metal", "real", "si", "cgs", "electron", "micro", "nano"}:
        raise ValueError(f"Unsupported physical LAMMPS units: {units!r}")
    if not isinstance(id_offset, (int, np.integer)):
        raise TypeError("id_offset must be an integer")
    type_symbols = normalize_type_symbols(type_symbols)
    data = read_input_datafile(input_datafile, units=units)[-1] if input_datafile is not None else None
    for index, text in enumerate(_iter_blocks(path, byte_range), start=frame_offset):
        try:
            timestep, columns, bounds, declared = _parse_block(text)
            if declared is not None and declared != units:
                raise ValueError(f"Dump declares units {declared!r}, requested {units!r}")
            ids = columns["id"]
            if data is not None and set(ids) != set(data):
                raise ValueError("Dump/data-file atom ID sets differ")
            symbols = columns.get("element")
            if type_symbols is not None:
                if "type" not in columns:
                    raise ValueError("type_symbols requires a type column")
                mapped = np.asarray([type_symbols[int(t)] for t in columns["type"]])
                if symbols is not None and not np.array_equal(symbols, mapped):
                    raise ValueError("Dump elements conflict with type_symbols")
                symbols = mapped
            if symbols is None and data is not None:
                symbols = [data[int(i)]["symbol"] for i in ids]
            if symbols is None and "mass" in columns:
                symbols = [mass_to_symbol(m) for m in convert(columns["mass"], "mass", units, "ASE")]
            if symbols is None:
                raise ValueError("Provide dump elements, type_symbols, data file, or masses")
            # Symbols are passed explicitly, so arbitrary type labels are never
            # interpreted as atomic numbers.
            atoms, velocities, forces = _atoms_from_columns(columns, symbols, bounds, units)
            # ASE stores origin separately and omits it from scaled positions.
            scaled = "x" not in columns and ("xs" in columns or
                     ("xu" not in columns and "xsu" in columns))
            origin = np.asarray(atoms.get_celldisp()).ravel()
            if scaled:
                atoms.positions += origin
            if "mass" in columns:
                if np.any(columns["mass"] <= 0):
                    raise ValueError("Masses must be positive")
                atoms.set_masses(convert(columns["mass"], "mass", units, "ASE"))
            elif data is not None:
                # Data-file masses are already in ASE units. Do not silently
                # substitute standard elemental masses for the MD masses.
                atoms.set_masses([data[int(i)]["mass"] for i in ids])
            if velocities is not None:
                atoms.set_velocities(velocities)
            if "ix" in columns:
                # LAMMPS images of the dumped coordinates. Unwrapped columns
                # (xu/xsu) already contain them, so relative to the stored
                # positions the images are zero.
                images = np.column_stack([columns[k] for k in ("ix", "iy", "iz")])
                if "x" not in columns and "xs" not in columns:
                    images = np.zeros_like(images)
            if wrap:
                before = atoms.get_scaled_positions(wrap=False)
                atoms.positions -= origin
                atoms.wrap()
                atoms.positions += origin
                if "ix" in columns:
                    # Keep r + n @ cell invariant: add the lattice shift wrap removed.
                    images = images + np.rint(before - atoms.get_scaled_positions(wrap=False)).astype(int)
            atoms.new_array("lammps_id", ids.copy())
            atoms.new_array("global_index", ids - id_offset)
            if forces is not None:
                atoms.new_array("lammps_forces", forces.copy())
            if "mol" in columns:
                atoms.new_array("molecule_id", columns["mol"].copy())
            if "ix" in columns:
                atoms.new_array("image_flags", images)
            atoms.info.update(source=str(Path(path).resolve()), frame_index=index,
                              timestep=timestep, lammps_units=units,
                              force_units="eV/Angstrom", velocity_units="ASE", id_offset=int(id_offset))
            yield LammpsFrame(atoms, columns, bounds, index, timestep, units, int(id_offset))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{path}: frame {index}: {exc}") from exc


def read_lammps_frame(path, index=0, **kwargs):
    """Read one frame; negative indices retain only required trailing frames."""
    return _select(iter_lammps_frames(path, **kwargs), index)


def read_lammps_input_settings(path):
    """Extract literal input settings without executing LAMMPS.

    Reads units, atom_style, boundary, timestep and dump_modify element maps.
    Comments do not establish model/task/charge/spin. Includes, branches, loops
    and substitutions in settings require a resolved input file.
    """
    result = {"source": str(Path(path).resolve()), "dump_elements": {}}
    pending = ""
    simulation_started = False
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            tokens = shlex.split(line, comments=True)
            continued = bool(tokens and tokens[-1] == "&")
            if continued:
                tokens.pop()
            pending += " " + " ".join(tokens)
            if continued:
                continue
            tokens, pending = pending.split(), ""
            if not tokens:
                continue
            command, *args = tokens
            if command == "clear":
                if simulation_started or len(result) > 2 or result["dump_elements"]:
                    raise ValueError(f"{path}:{line_number}: clear after simulation settings requires resolved input settings")
                # A normal initialization clear does not discard any settings
                # collected by this reader. No LAMMPS commands are executed.
                continue
            if command in {"read_data", "read_restart", "create_box", "run", "minimize", "dump"}:
                simulation_started = True
            if command in {"include", "jump", "if", "next"}:
                raise ValueError(f"{path}:{line_number}: {command} requires resolved input settings")
            if command in {"units", "atom_style", "boundary", "timestep"}:
                if any("$" in token for token in args):
                    raise ValueError(f"{path}:{line_number}: unresolved {command}")
                expected = 3 if command == "boundary" else 1
                if len(args) != expected:
                    raise ValueError(f"{path}:{line_number}: invalid {command}")
                value = tuple(args) if command == "boundary" else float(args[0]) if command == "timestep" else args[0]
                if command in result and result[command] != value:
                    raise ValueError(f"{path}:{line_number}: changing {command}; provide frame-specific settings")
                result[command] = value
            elif command == "dump_modify" and "element" in args:
                from ase.data import atomic_numbers
                start = args.index("element") + 1
                symbols = []
                for token in args[start:]:
                    if token not in atomic_numbers:
                        break
                    symbols.append(token)
                if not symbols:
                    raise ValueError(f"{path}:{line_number}: unresolved dump element mapping")
                mapping = dict(enumerate(symbols, 1))
                previous = result["dump_elements"].get(args[0])
                if previous is not None and previous != mapping:
                    raise ValueError(f"{path}:{line_number}: changing dump element mapping")
                result["dump_elements"][args[0]] = mapping
    if pending.strip():
        raise ValueError(f"{path}: unfinished line continuation")
    return result


def my_readlammps(lammps_dump, id_offset=1, *, index=-1):
    """Legacy raw tuple reader; default remains the final frame, in raw units.

    Record keys are raw atom_id-id_offset. All numeric values retain the
    dump's original units; no equation converts them to ASE units.
    """
    def frames():
        for frame_index, text in enumerate(_iter_blocks(lammps_dump)):
            try:
                timestep, columns, bounds, _ = _parse_block(text)
                yield timestep, len(columns["id"]), bounds, _records(columns, id_offset)
            except ValueError as exc:
                raise ValueError(f"{lammps_dump}: frame {frame_index}: {exc}") from exc
    return _select(frames(), index)


def read_input_datafile(filename, *, units="metal", atom_style=None):
    """Legacy six-item result using ASE's data-file parser and mass inference.

    Coordinates/masses use ASE units. Type metadata covers populated types.
    Prefer explicit species mappings for nonstandard masses.

    ASE converts positions/masses to Angstrom/amu. For restricted triclinic
    cell H, returned bounds have hi=origin+diag(H) and tilts xy=H[1,0],
    xz=H[2,0], yz=H[2,1]; these are cell limits, not enclosing dump-box bounds.
    origin and cell (rows a, b, c) are included under the same keys as in dump
    box_bounds, so the two sources can be compared directly (Angstrom here).
    """
    atoms = read_lammps_data(filename, units=units, atom_style=atom_style,
                             sort_by_id=False, read_image_flags=False)
    if "masses" not in atoms.arrays:
        raise ValueError("Data-file element inference requires a Masses section")
    ids = atoms.arrays["id"]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate atom IDs in data file")
    atoms = atoms[np.argsort(ids)]
    ids, types = atoms.arrays["id"], atoms.arrays["type"]
    records = [(int(i), int(t), s, *xyz) for i, t, s, xyz in
               zip(ids, types, atoms.get_chemical_symbols(), atoms.positions)]
    by_id = {row[0]: {"type": row[1], "symbol": row[2], "coords": tuple(row[3:])} for row in records}
    for atom_id, mass in zip(ids, atoms.get_masses()):
        by_id[int(atom_id)]["mass"] = float(mass)
    type_data = {int(t): (s, float(m)) for t, s, m in
                 zip(types, atoms.get_chemical_symbols(), atoms.get_masses())}
    origin = np.asarray(atoms.get_celldisp()).ravel()
    cell = atoms.cell.array
    bounds = dict(zip(("xlo", "ylo", "zlo"), origin))
    bounds.update(dict(zip(("xhi", "yhi", "zhi"), origin + cell.diagonal())))
    bounds.update(xy=cell[1, 0], xz=cell[2, 0], yz=cell[2, 1], origin=origin.tolist(), cell=cell.tolist())
    return len(atoms), len(type_data), bounds, type_data, records, by_id


def split_lammpstrj(lammpstrj_path, output_dir):
    """Split every frame, validating before writing (one frame in memory)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(_iter_blocks(lammpstrj_path)):
        _parse_block(text)
        target = output_dir / f"frame_{index}.lammpstrj"
        if target.resolve() == Path(lammpstrj_path).resolve():
            raise ValueError("Split output must not overwrite the source dump")
        target.write_text(text, encoding="utf-8")


def make_ase_obj_and_fix_atomtypes(lammps_dump, input_datafile=None, index=0, *,
                                  id_offset=1, wrap=True, units=None):
    """Compatibility wrapper; source forces live in arrays['lammps_forces']."""
    if units is None:
        warnings.warn("Implicit units='metal' is deprecated; pass the MD units explicitly",
                      FutureWarning, stacklevel=2)
        units = "metal"
    return read_lammps_frame(lammps_dump, index=index, units=units,
                             input_datafile=input_datafile, id_offset=id_offset, wrap=wrap).atoms
