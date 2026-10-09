"""MartiniFuncs: LAMMPS trajectory postprocessing for molecular pair chemistry.

Subpackages
-----------
io          file formats (LAMMPS dumps/inputs/thermo, Psience bridges)
structure   bond graphs, periodic cells, alignment, packing
trajectory  streaming analysis, parallel frames, reaction events
statistics  MD observables, lag statistics, dynamics and diffusion diagnostics
pairs       chemistry-independent pair machinery: pair models, screening,
            candidates, selection, lifetimes, structures
chemistry   pair models for specific chemistry (cpd_n4t)
recompute   isolated-fragment forces/energies (F_env) and optimizations of exported structures
workflows   CLI, cache/report/export, force validation
legacy      superseded helpers kept for reference

The names below are re-exported lazily (importing MartiniFuncs stays cheap).
"""
from importlib import import_module

_API = {
    "read_lammps_frame": "io.lammps", "iter_lammps_frames": "io.lammps",
    "analyze_run": "trajectory.analysis", "analyze_segments": "trajectory.analysis",
    "select_frame_window": "trajectory.analysis",
    "cache_analysis": "workflows.postprocess", "load_analysis_cache": "workflows.postprocess",
    "write_report": "workflows.postprocess", "export_pairs": "workflows.postprocess",
    "select_pairs": "pairs.candidates", "candidate_table": "pairs.candidates",
    "get_pair_lifetime": "pairs.lifetimes",
    "PairModel": "pairs.model", "PairCollection": "pairs.model", "PairCriteria": "pairs.model",
    "get_pair_model": "pairs.model",
    "register_pair_model": "pairs.model",
    "CpdN4tCollection": "chemistry.cpd_n4t.model", "CpdN4tCriteria": "chemistry.cpd_n4t.model",
    "CpdN4tModel": "chemistry.cpd_n4t.model",
}

__all__ = sorted(_API)


def __getattr__(name):
    if name in _API:
        value = getattr(import_module(f"{__name__}.{_API[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_API))
