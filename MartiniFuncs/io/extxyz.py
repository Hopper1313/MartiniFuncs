"""Extended XYZ output with full floating-point precision.

ASE's own writer formats floats as %16.8f, which truncates small MD force
components. This writer keeps ASE's Extended XYZ header and column schema
(so ase.io.read reads the files unchanged) but writes every float with 17
significant digits, which round-trips float64 exactly.
"""
from contextlib import nullcontext

import numpy as np
from ase.io.extxyz import output_column_format


def write_extxyz(images, filename):
    """Write Atoms (an iterable, consumed once) as Extended XYZ frames.

    Every per-atom array is written as a column and Atoms.info as header
    key=value pairs. filename is a path (overwritten) or an open text stream.
    Constraints are not written; record anything that must survive (e.g. a
    frozen-atom mask) as a per-atom array.
    """
    context = nullcontext(filename) if hasattr(filename, "write") else open(filename, "w", encoding="utf-8")
    with context as stream:
        for atoms in images:
            arrays = {"symbols": np.asarray(atoms.get_chemical_symbols()),
                      "positions": atoms.positions,
                      **{k: v for k, v in atoms.arrays.items() if k not in {"numbers", "positions"}}}
            columns = list(arrays)
            comment, _, _, _ = output_column_format(atoms, columns, arrays)
            stream.write(f"{len(atoms)}\n{comment}\n")
            # One %-format per row ("%.17g" % x == format(x, ".17g"); "%s" == str).
            blocks, tokens = [], []
            for name in columns:
                array = np.asarray(arrays[name])
                block = array.reshape(len(atoms), -1)
                if array.dtype.kind == "b":
                    block = np.where(block, "T", "F")     # Extended XYZ logical columns
                tokens += ["%.17g" if array.dtype.kind == "f" else "%s"] * block.shape[1]
                blocks.append(block.tolist())
            line = " ".join(tokens) + "\n"
            stream.write("".join(line % tuple(v for part in row for v in part) for row in zip(*blocks)))
