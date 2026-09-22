"""LAMMPS I/O helpers for MartiniFuncs.

Persistent ``global_index`` values are derived from LAMMPS atom IDs rather than
from dump-file row order. By default LAMMPS IDs 1..N map to persistent IDs
0..N-1, preserving compatibility with the historical package behavior while
remaining stable if dump rows are reordered.
"""

from __future__ import annotations

import numpy as np
import periodictable as pt
import os


def mass_to_symbol(target_mass):
    closest_element = None
    min_diff = float("inf")
    for element in pt.elements:
        if element.mass is not None:
            diff = abs(element.mass - target_mass)
            if diff < min_diff:
                min_diff = diff
                closest_element = element
    return closest_element.symbol if closest_element else None


def my_readlammps(lammps_dump, id_offset: int = 1):
    """Read the final frame of a text LAMMPS dump.

    The returned ``atoms_data`` dictionary is keyed by persistent atom ID:
    ``LAMMPS id - id_offset``. Each record still contains the original ``id``.
    This keeps force/velocity lookup consistent with ``Atoms.global_index``.
    """
    timestep = None
    natoms = None
    boxbounds = None
    atoms_data = None

    with open(lammps_dump, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                continue

            if "ITEM: NUMBER OF ATOMS" in line:
                natoms = int(f.readline().strip())
                continue

            if "ITEM: BOX BOUNDS" in line:
                rows = [list(map(float, f.readline().split())) for _ in range(3)]
                # Standard triclinic dump: xlo_bound xhi_bound xy, etc.
                xlo, xhi = rows[0][:2]
                ylo, yhi = rows[1][:2]
                zlo, zhi = rows[2][:2]
                xy = rows[0][2] if len(rows[0]) > 2 else 0.0
                xz = rows[1][2] if len(rows[1]) > 2 else 0.0
                yz = rows[2][2] if len(rows[2]) > 2 else 0.0
                boxbounds = {
                    "xlo": xlo,
                    "xhi": xhi,
                    "ylo": ylo,
                    "yhi": yhi,
                    "zlo": zlo,
                    "zhi": zhi,
                    "xy": xy,
                    "xz": xz,
                    "yz": yz,
                }
                continue

            if "ITEM: ATOMS" in line:
                if natoms is None:
                    raise ValueError("NUMBER OF ATOMS must appear before ATOMS")
                colnames = line.strip().split()[2:]
                if "id" not in colnames:
                    raise ValueError(
                        "LAMMPS dump must contain an 'id' column to maintain "
                        "persistent atom identity."
                    )

                atoms_data = {}
                for _ in range(natoms):
                    values = f.readline().split()
                    record = {
                        col: (
                            str(val) if col == "element"
                            else int(val) if col in {"id", "type"}
                            else float(val)
                        )
                        for col, val in zip(colnames, values)
                    }
                    persistent_id = int(record["id"]) - int(id_offset)
                    if persistent_id in atoms_data:
                        raise ValueError(
                            f"Duplicate LAMMPS atom ID {record['id']} in dump."
                        )
                    record["global_index"] = persistent_id
                    atoms_data[persistent_id] = record

    if atoms_data is None:
        raise ValueError("No atom data found in the LAMMPS dump file.")

    return timestep, natoms, boxbounds, atoms_data


def read_input_datafile(filename):
    natoms = None
    natom_types = None
    box_bounds = {}
    atom_types = {}
    atoms_data = []

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "Velocit" in line:
                break

            if "atoms" in line:
                fields = line.split()
                if len(fields) >= 2 and fields[0].isdigit() and fields[1] == "atoms":
                    natoms = int(fields[0])

            if "atom types" in line:
                natom_types = int(line.split()[0])

            if "Masses" in line:
                f.readline()
                for _ in range(natom_types):
                    fields = f.readline().split()
                    atom_types[int(fields[0])] = (
                        mass_to_symbol(float(fields[1])),
                        float(fields[1]),
                    )

            if "xlo" in line and "xhi" in line:
                xlo, xhi = map(float, line.split()[:2])
                yline = f.readline().split()
                zline = f.readline().split()
                ylo, yhi = map(float, yline[:2])
                zlo, zhi = map(float, zline[:2])
                try:
                    tilt = f.readline().split()
                    xy, xz, yz = map(float, tilt[:3])
                except ValueError:
                    xy = xz = yz = 0.0
                box_bounds = {
                    "xlo": xlo,
                    "xhi": xhi,
                    "ylo": ylo,
                    "yhi": yhi,
                    "zlo": zlo,
                    "zhi": zhi,
                    "xy": xy,
                    "xz": xz,
                    "yz": yz,
                }

            if "Atoms # atomic" in line:
                f.readline()
                for _ in range(natoms):
                    fields = f.readline().split()
                    atom_id = int(fields[0])
                    atom_type = int(fields[1])
                    atoms_data.append(
                        (
                            atom_id,
                            atom_type,
                            atom_types[atom_type][0],
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                        )
                    )

    atoms_data_dict = {
        row[0]: {
            "type": row[1],
            "symbol": row[2],
            "coords": (row[3], row[4], row[5]),
        }
        for row in atoms_data
    }
    return natoms, natom_types, box_bounds, atom_types, atoms_data, atoms_data_dict

def split_lammpstrj(lammpstrj_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(lammpstrj_path, "r") as f:
        frame_lines = []
        frame_index = 0
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" in line:
                frame_lines.append(line)
                while True:
                    line = f.readline()
                    if not line or "ITEM: TIMESTEP" in line:
                        with open(os.path.join(output_dir, f"frame_{frame_index}.lammpstrj"), "w") as out_f:
                            out_f.writelines(frame_lines)
                        frame_lines = []
                        frame_index += 1
                        break
                    frame_lines.append(line)


def make_ase_obj_and_fix_atomtypes(
    lammps_dump,
    input_datafile,
    index=0,
    *,
    id_offset: int = 1,
    wrap: bool = True,
):
    """Read a LAMMPS dump and attach stable persistent atom IDs.

    ASE's LAMMPS dump reader sorts atoms by LAMMPS ``id`` by default. We read
    the IDs explicitly as well, assign chemical symbols by those IDs, and set
    ``global_index = LAMMPS id - id_offset``. This is stable across frames and
    does not depend on dump row order.
    """
    import ase.io.lammpsrun as lmps

    if index == ":":
        raise ValueError("Please specify one frame index rather than reading all frames.")

    (
        natoms,
        _,
        _,
        _,
        _,
        atoms_data_dict,
    ) = read_input_datafile(input_datafile)

    # Explicit order=True documents the identity assumption used below.
    atoms = lmps.read_lammps_dump(lammps_dump, index=index, order=True)

    # The helper is intended for the split single-frame files used by the
    # existing workflow. It provides the LAMMPS IDs in a robust way.
    _, dump_natoms, _, raw = my_readlammps(lammps_dump, id_offset=id_offset)
    if dump_natoms != len(atoms) or natoms != len(atoms):
        raise ValueError(
            f"Atom-count mismatch: data={natoms}, dump={dump_natoms}, ASE={len(atoms)}"
        )

    lammps_ids = np.asarray(
        sorted(int(record["id"]) for record in raw.values()),
        dtype=int,
    )
    if len(lammps_ids) != len(atoms):
        raise ValueError("Could not recover one LAMMPS ID per ASE atom.")

    try:
        symbols = [atoms_data_dict[int(atom_id)]["symbol"] for atom_id in lammps_ids]
    except KeyError as exc:
        raise KeyError(
            f"LAMMPS atom ID {exc.args[0]} is present in the dump but not the data file."
        ) from exc

    atoms.set_chemical_symbols(symbols)
    if wrap:
        atoms.wrap()

    persistent_ids = lammps_ids - int(id_offset)
    if "global_index" in atoms.arrays:
        atoms.arrays["global_index"] = persistent_ids.astype(int)
    else:
        atoms.new_array("global_index", persistent_ids.astype(int))

    if "lammps_id" in atoms.arrays:
        atoms.arrays["lammps_id"] = lammps_ids
    else:
        atoms.new_array("lammps_id", lammps_ids)

    return atoms

######################################################################
#######-------------------------------------------------------########
####### Below is before refactoring.  Saving for reference.   ########
#######-------------------------------------------------------########
######################################################################

# import periodictable as pt
# import numpy as np

# def mass_to_symbol(target_mass):
#     closest_element = None
#     min_diff = float('inf')
    
#     # Loop through all valid elements in the periodic table
#     for element in pt.elements:
#         if element.mass is not None:
#             diff = abs(element.mass - target_mass)
#             if diff < min_diff:
#                 min_diff = diff
#                 closest_element = element
                
#     return closest_element.symbol if closest_element else None

# def my_readlammps(lammps_dump):
#     timestep = None
#     natoms = None
#     boxbounds = None
#     atoms_data = None
#     with open(lammps_dump, 'r') as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 break
#             if 'ITEM: TIMESTEP' in line:
#                 timestep = int(f.readline().strip())
#             if 'ITEM: NUMBER OF ATOMS' in line:
#                 natoms = int(f.readline().strip())
#             if 'BOX BOUNDS' in line:
#                 xlo, xhi, xy = map(float, f.readline().split())
#                 ylo, yhi, xz = map(float, f.readline().split())
#                 zlo, zhi, yz = map(float, f.readline().split())
#                 boxbounds = {'xlo': xlo, 'xhi': xhi, 'ylo': ylo, 'yhi': yhi, 'zlo': zlo, 'zhi': zhi, 'xy': xy, 'xz': xz, 'yz': yz}
#             if 'ITEM: ATOMS' in line:
#                 colnames = line.strip().split()[2:]
#                 atoms_data = {}
#                 for _, atomindex in enumerate(range(natoms)):
#                     line = f.readline().split()
#                     atoms_data[atomindex] = {col: float(val) if col not in ['id', 'type'] else int(val) for col, val in zip(colnames, line)}
#                 # Ensure atoms_data is not empty
#                 if not atoms_data:
#                     raise ValueError("No atom data found in the LAMMPS dump file.")
            
#     return timestep, natoms, boxbounds, atoms_data  


# def read_input_datafile(filename):
#     natoms = None
#     natom_types = None
#     box_bounds = {}
#     atom_types = {}
#     atoms_data = []
#     with open(filename, 'r') as f:
#         while True:
#             line = f.readline()
#             if not line:
#                 break
#             if 'Velocit' in line:
#                 break
#             if 'atoms' in line: 
#                 if line.split()[0].isdigit() and line.split()[1] == 'atoms':
#                     natoms = int(line.split()[0])
#             if 'atom types' in line:
#                 natom_types = int(line.split()[0])
#             if 'Masses' in line:
#                 f.readline() # skip the next line
#                 i = 0
#                 while i < natom_types:
#                     line = f.readline().split()
#                     i += 1
#                     atom_types[int(line[0])] = (mass_to_symbol(float(line[1])), float(line[1]))
#             if 'xlo' in line and 'xhi' in line:
#                 xlo = float(line.split()[0])
#                 xhi = float(line.split()[1])
#                 line = f.readline()
#                 ylo = float(line.split()[0])
#                 yhi = float(line.split()[1])
#                 line = f.readline()
#                 zlo = float(line.split()[0])
#                 zhi = float(line.split()[1])
#                 line = f.readline()
#                 xy = float(line.split()[0])
#                 xz = float(line.split()[1])
#                 yz = float(line.split()[2])
#                 box_bounds = {'xlo': xlo,'xhi': xhi, 'ylo': ylo, 'yhi': yhi, 'zlo': zlo, 'zhi': zhi, 'xy': xy, 'xz': xz, 'yz': yz}

#             if 'Atoms # atomic' in line:
#                 f.readline() # skip the next line
#                 i = 0
#                 while i < natoms:
#                     line = f.readline().split()
#                     i += 1
#                     atoms_data.append((int(line[0]), int(line[1]), atom_types[int(line[1])][0], float(line[2]), float(line[3]),float(line[4])))
#     atoms_data_dict = {atoms_data[i][0]: {'type': atoms_data[i][1], 'symbol': atoms_data[i][2], 'coords': (atoms_data[i][3], atoms_data[i][4], atoms_data[i][5])} for i in range(len(atoms_data))}
#     return natoms, natom_types, box_bounds, atom_types, atoms_data, atoms_data_dict

# def make_ase_obj_and_fix_atomtypes(lammps_dump, input_datafile, index = 0):
#     import ase.io.lammpsrun as lmps
#     if index == ":":
#         print("I will not read all the frames, this will take a long time. Please specify a frame index to read.")
#         raise ValueError("Please specify a frame index to read.")
#     natoms, natom_types, box_bounds, atom_types, atoms_data, atoms_data_dict = read_input_datafile(input_datafile)
#     test = lmps.read_lammps_dump(lammps_dump, index=index)
#     test.set_chemical_symbols([atoms_data_dict[i]['symbol'] for i in range(1, len(atoms_data_dict)+1)])
#     test.wrap()
#     test.arrays['global_index'] = np.arange(len(test), dtype=int)
#     return test