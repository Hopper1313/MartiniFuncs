"""Run a recompute task over every structure of an Extended XYZ file."""
from pathlib import Path

from ase.io import iread

from MartiniFuncs.io.extxyz import write_extxyz


def recompute_file(input_path, output_path, task, *, extras_path=None, progress=None):
    """Read input_path frame by frame, apply task, write output_path (and extras_path).

    task is ForceDecomposition, FragmentRelaxation or FrozenShellOptimization
    (anything with apply(structure) -> RecomputeResult). Each output frame is
    the input frame with the task's arrays and info added, in input order. Extra
    structures (FragmentRelaxation's relaxed fragments) go to extras_path,
    default <output stem>.relaxed<suffix>. Existing outputs are never
    overwritten. Returns {"structures", "output", "extras", "extra_structures"}.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    extras_path = Path(extras_path) if extras_path is not None else output_path.with_name(
        f"{output_path.stem}.relaxed{output_path.suffix}")
    for path in (output_path, extras_path):
        if path.resolve() == input_path.resolve():
            raise ValueError("Output must differ from the input file")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new output name")
    count, extras = 0, []

    def results():
        nonlocal count
        for structure in iread(input_path, index=":", format="extxyz"):
            result = task.apply(structure)
            extras.extend(result.extras)
            count += 1
            if progress:
                progress(f"Recomputed structure {count}")
            yield result.structure

    _write_atomically(results(), output_path)
    if extras:
        _write_atomically(extras, extras_path)
    return {"structures": count, "output": str(output_path),
            "extras": str(extras_path) if extras else None, "extra_structures": len(extras)}


def _write_atomically(images, path):
    """Write to <path>.tmp and rename on success, so a failed task never leaves a partial file."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        write_extxyz(images, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
