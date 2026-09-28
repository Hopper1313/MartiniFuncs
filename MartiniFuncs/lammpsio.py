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
from ase.io.lammpsrun import read_lammps_dump_text


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


def mass_to_symbol(target_mass):
    """Nearest elemental mass; prefer explicit mappings for nonstandard masses.

    Choose Z minimizing |element_mass[Z]-target_mass|, with masses in amu.
    This nearest-mass heuristic is not reliable for isotope or coarse-grain mapping.
    """
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("Atomic mass must be positive and finite")
    return min((e for e in pt.elements if e.mass is not None),
               key=lambda e: abs(e.mass - target_mass)).symbol


def _iter_blocks(path):
    """Yield one text frame at a time without losing boundary markers."""
    with open(path, encoding="utf-8") as stream:
        block = []
        for line_number, line in enumerate(stream, 1):
            if line.strip() == "ITEM: TIMESTEP":
                if block:
                    yield "".join(block)
                block = [line]
            elif block:
                block.append(line)
            elif line.strip():
                raise ValueError(f"{path}:{line_number}: expected ITEM: TIMESTEP")
        if block:
            yield "".join(block)


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
            rows = []
            for row_index in range(count):
                values = take().split()
                if len(values) != len(names):
                    raise ValueError(f"Atom row {row_index + 1}: expected {len(names)} columns, got {len(values)}")
                rows.append(values)
            columns = {}
            for i, name in enumerate(names):
                dtype = str if name == "element" else int if (
                    name in {"id", "type", "mol", "ix", "iy", "iz"}
                    or name.startswith(("i_", "i2_"))
                ) else float
                values = np.asarray([row[i] for row in rows], dtype=dtype)
                if dtype is not str and not np.isfinite(values).all():
                    raise ValueError(f"Non-finite values in {name}")
                columns[name] = values
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
                       id_offset=1, wrap=False):
    """Stream synchronized frames with explicit LAMMPS units.

    Each bounded frame is parsed for raw metadata and by ASE for geometry and
    unit conversion. Species priority: dump elements, explicit type mapping,
    data-file mapping, then dump masses. Conflicting element/type mappings fail.
    Reduced LJ units are unsupported. Coordinates include the box origin.

    Raw columns stay in source units; ASE conversion applies
    q_ASE=q_source*unit_factor(source, ASE) to distances, masses, velocities
    and forces. Scaled coordinates obey r=s @ cell+origin; optional wrapping
    acts on r-origin. global_index=lammps_id-id_offset; image flags are stored,
    not added to positions. For wrapped r and image n, whole r is r+n @ cell.
    """
    if units not in {"metal", "real", "si", "cgs", "electron", "micro", "nano"}:
        raise ValueError(f"Unsupported physical LAMMPS units: {units!r}")
    if not isinstance(id_offset, (int, np.integer)):
        raise TypeError("id_offset must be an integer")
    data = read_input_datafile(input_datafile, units=units)[-1] if input_datafile is not None else None
    for index, text in enumerate(_iter_blocks(path)):
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
            # Insert explicit elements so arbitrary type labels cannot be
            # interpreted as atomic numbers by ASE.
            if "element" not in columns:
                lookup = dict(zip(ids, symbols))
                rows = text.splitlines()
                header = next(i for i, row in enumerate(rows) if row.startswith("ITEM: ATOMS "))
                id_col = rows[header].split()[2:].index("id")
                rows[header] += " element"
                for i in range(header + 1, header + 1 + len(ids)):
                    rows[i] += " " + str(lookup[int(rows[i].split()[id_col])])
                text = "\n".join(rows) + "\n"
            atoms = read_lammps_dump_text(StringIO(text), index=0, order=True, units=units)
            velocities = atoms.get_velocities() if "vx" in columns else None
            forces = atoms.get_forces() if "fx" in columns else None
            atoms.calc = None
            # ASE stores origin separately and omits it from scaled positions.
            scaled = "x" not in columns and ("xs" in columns or
                     ("xu" not in columns and "xsu" in columns))
            origin = np.asarray(atoms.get_celldisp()).ravel()
            if scaled:
                atoms.positions += origin
            atoms.set_chemical_symbols(symbols)
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
            if wrap:
                atoms.positions -= origin
                atoms.wrap()
                atoms.positions += origin
            atoms.new_array("lammps_id", ids.copy())
            atoms.new_array("global_index", ids - id_offset)
            if forces is not None:
                atoms.new_array("lammps_forces", forces.copy())
            if "mol" in columns:
                atoms.new_array("molecule_id", columns["mol"].copy())
            if "ix" in columns:
                atoms.new_array("image_flags", np.column_stack([columns[k] for k in ("ix", "iy", "iz")]))
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
    bounds.update(xy=cell[1, 0], xz=cell[2, 0], yz=cell[2, 1])
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
