"""Post-simulation analysis with explicit source-frame selection.

Usage (also available as the installed martini-postprocess command)::

    python -m MartiniFuncs.cli run RUN_DIR --output ANALYSIS_DIR --units metal --timestep-ps 0.0005
    python -m MartiniFuncs.cli report ANALYSIS_DIR/cache --output REPORT_DIR --startframe 1000 --endframe 2000
    python -m MartiniFuncs.cli extract ANALYSIS_DIR/cache --output PAIRS_DIR --startframe 1000 --endframe 2000 --shell-radius 3

Frame convention
----------------
startframe is inclusive and endframe is exclusive, like Python slicing. Both
refer to zero-based frame positions in the original dump, NOT LAMMPS step numbers
or row numbers after striding. [1000, 2000) includes frames 1000 through 1999.
With --frame-stride 2, [3, 9) contains retained source frames 4, 6, 8. Omitting
both bounds selects the entire available trajectory. A --max-frames preview
restricts what is available; it cannot be used to select a late portion directly.

To clarify, if your .lammpstrj output looks like
    ITEM: TIMESTEP
    0
    ... < rest of frame data here >
    ... < end of frame data >
    ITEM: TIMESTEP
    10000
    ...
    ...

This would be frame 0 and frame 1 [not frame 0 and frame 10000]

Data flow
---------
run discovers source files and creates/reuses OUTPUT/cache through cache_analysis.
It caches all frames allowed by stride/preview settings so later frame selections
can reuse it. Bounds apply to BOTH the report and structure exports, not to cache
creation or temporal unwrapping. Default reports/exports cover the entire cache.
report calls write_report using only cached molecular arrays and pair identities;
source simulation files are unnecessary. extract calls export_pairs and rereads
the source dump for accepted pair coordinates, atomic forces and optional shells.
Completed matching products are reused; conflicting products are never overwritten.

Units and resources
-------------------
Trajectory selection uses frames; lag and diffusion-fit intervals remain in ps.
--timestep-ps is ps per MD integration step, not per dump frame. Temperature
requires explicit --temperature-dof. Pair screening is CPD/dienophile-specific;
--no-pairs enables generic molecular statistics. Pair workers are local processes
inside one Slurm task, not MPI ranks; no GPU/model inference occurs here.
--workers N and --batch-size B pass through cache_analysis to analyze_run's
existing pair process pool (B retained frames per task). The cached pair backend
supports multiple workers; the reference backend requires one. Reading, temporal
unwrapping, report calculations, and exports remain sequential. Reusing a finished
cache skips pair computation regardless of the requested worker count.

This module only parses/validates options and dispatches package functions.
Scientific equations live in trajectory_analysis, md_statistics, and pair_selection;
persistence/export behavior lives in postprocessing. See docs/postprocessing_cli.md
for output schemas, cache reuse, and Slurm submission examples.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

from .pair_selection import PairCriteria
from .postprocessing import cache_analysis, export_pairs, write_report


def _frame_options(parser):
    """Add the shared half-open source-frame interval to a command parser.

    Keep one definition for run/report/extract so striding and boundary semantics
    cannot differ between reports and exports. None means through the cached end.
    """
    parser.add_argument("--startframe", type=int, default=0,
                        help="inclusive zero-based source dump frame (default: 0)")
    parser.add_argument("--endframe", type=int, default=None,
                        help="exclusive source dump frame; omitted means through the end")


def _report_options(parser):
    """Add frame selection and independent statistical settings.

    max_lag_ps limits comparison separation, while fit_ps selects lags used in
    D=slope(MSD)/6. Neither option selects which source frames enter the report.
    """
    _frame_options(parser)
    parser.add_argument("--max-lag-ps", type=float, default=100., help="longest correlation/MSD time separation in ps")
    parser.add_argument("--fit-ps", type=float, nargs=2, metavar=("START", "END"), help="explicit diffusion fit lag interval in ps")
    parser.add_argument("--max-gap-frames", type=int, default=0, help="allowed missing retained samples within pair lifetimes")
    parser.add_argument("--stability", action="store_true", help="include scalar ACF/batch errors; quadratic time for long series")
    parser.add_argument("--rdf-bins", type=int, help="include molecular COM RDFs with this many radial bins")


def _report_kwargs(args):
    """Pass frame bounds and lag settings unchanged to postprocessing.write_report.

    This conversion performs no frame-to-time conversion: cached timestamps
    determine lag spacing after the report selects the requested source frames.
    """
    return dict(startframe=args.startframe, endframe=args.endframe,
                max_lag_ps=args.max_lag_ps, fit_interval_ps=args.fit_ps, max_gap_frames=args.max_gap_frames,
                stability=args.stability, rdf_bins=args.rdf_bins)


def build_parser():
    """Return the CLI parser; parsing does not read or write simulation data.

    run owns expensive cache settings (physics, topology, sampling, workers).
    report owns reusable statistics settings. extract owns geometry export and
    shell radius. All three share the same source-frame selection convention.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="analyze a finished simulation, cache results, report, and export pairs")
    run.add_argument("directory", type=Path, help="simulation directory; unique input/data/dump discovered automatically")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--units", required=True, choices=["metal"], help="source LAMMPS units; current statistics support metal")
    run.add_argument("--timestep-ps", required=True, type=float, help="ps per MD step, not per saved frame")
    for name in ("input", "dump", "data"):
        run.add_argument(f"--{name}-file", help="optional relative or absolute filename override")
    run.add_argument("--type-symbols", type=Path, help='JSON type map, e.g. {"1":"C","2":"H"}')
    run.add_argument("--frame-stride", type=int, default=1, help="retain every Nth source frame, anchored at frame 0; reading/unwrapping still uses every frame")
    run.add_argument("--max-frames", type=int, help="preview only: cap source frames read")
    run.add_argument("--bond-scale", type=float, default=1.3)
    run.add_argument("--temperature-dof", help="positive number or 3N-3; omitted means no temperature")
    run.add_argument("--no-pairs", action="store_true", help="generic molecular statistics without chemistry-specific pair screening")
    run.add_argument("--no-export-pairs", action="store_true", help="cache pair identities/lifetimes without a second structure-export pass")
    run.add_argument("--pair-backend", choices=["cached", "reference"], default="cached")
    run.add_argument("--workers", type=int, default=1, help="local pair worker processes, not MPI ranks (default: 1)")
    run.add_argument("--batch-size", type=int, default=32, help="retained frames per pair-worker task (default: 32)")
    for field in fields(PairCriteria):
        run.add_argument("--" + field.name.replace("_", "-"), type=float,
                         help="optional pair threshold (distances Angstrom; acute angles degrees)")
    run.add_argument("--shell-radius", type=float, help="also export whole-molecule solvent shells, radius in Angstrom")
    run.add_argument("--report-name", default="default", help="single folder name below output/reports")
    run.add_argument("--quiet", action="store_true")
    _report_options(run)

    report = commands.add_parser("report", help="regenerate statistics from cache without reading simulation files")
    report.add_argument("cache", type=Path, help="the output/cache directory from run")
    report.add_argument("--output", type=Path, required=True, help="new report directory")
    _report_options(report)

    extract = commands.add_parser("extract", help="export cached accepted pairs and optional shells from source frames")
    extract.add_argument("cache", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    _frame_options(extract)
    extract.add_argument("--shell-radius", type=float)
    extract.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    """Dispatch argv (or sys.argv) and return a shell-compatible exit status.

    Return 0 on success, 2 for caught validation/I/O errors. argparse handles
    invalid command syntax separately with SystemExit. run writes cache first,
    then the selected report and optional exports; a later report selection can
    reuse that cache. Windows process workers are protected by the module's
    __main__ guard. Errors print to stderr for Slurm logs.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.startframe < 0 or (args.endframe is not None and args.endframe <= args.startframe):
            raise ValueError("Require 0 <= startframe < endframe; omit endframe to use the cached end")
        if args.command == "run":
            if args.report_name in {"", ".", ".."} or any(c in args.report_name for c in "/\\:"):
                raise ValueError("--report-name must be one folder name")
            if args.shell_radius is not None and (args.no_pairs or args.no_export_pairs):
                raise ValueError("--shell-radius requires pair analysis and pair exports")
            allocated = os.environ.get("SLURM_CPUS_PER_TASK")
            if allocated is not None and args.workers > int(allocated):
                raise ValueError("--workers exceeds SLURM_CPUS_PER_TASK; request more CPUs or reduce workers")
            mapping = None
            if args.type_symbols:
                mapping = {int(k): v for k, v in json.loads(args.type_symbols.read_text(encoding="utf-8")).items()}
            dof = args.temperature_dof
            if dof is not None and dof != "3N-3":
                dof = float(dof)
            criteria = PairCriteria(**{f.name: getattr(args, f.name) for f in fields(PairCriteria)})
            progress = None if args.quiet else lambda message: print(message, flush=True)
            args.output.mkdir(parents=True, exist_ok=True)
            cache = args.output / "cache"
            cache_analysis(args.directory, cache, units=args.units, timestep_ps=args.timestep_ps,
                           input_file=args.input_file, dump_file=args.dump_file, data_file=args.data_file,
                           frame_stride=args.frame_stride, max_frames=args.max_frames, bond_scale=args.bond_scale,
                           type_symbols=mapping, degrees_of_freedom=dof, compute_pairs=not args.no_pairs,
                           pair_criteria=criteria, pair_backend=args.pair_backend, pair_workers=args.workers,
                           pair_batch_size=args.batch_size, progress=progress)
            report = write_report(cache, args.output / "reports" / args.report_name, **_report_kwargs(args))
            if not args.no_pairs and not args.no_export_pairs:
                # Separate explicitly selected exports from the full-cache product.
                pair_folder = "pairs" if args.startframe == 0 and args.endframe is None else f"pairs_frames_{args.startframe}_{args.endframe if args.endframe is not None else 'end'}"
                export_pairs(cache, args.output / pair_folder, shell_radius=args.shell_radius,
                             startframe=args.startframe, endframe=args.endframe, progress=progress)
            print(f"Cache: {cache.resolve()}\nReport: {report}")
        elif args.command == "report":
            print(write_report(args.cache, args.output, **_report_kwargs(args)))
        else:
            print(export_pairs(args.cache, args.output, shell_radius=args.shell_radius,
                               startframe=args.startframe, endframe=args.endframe,
                               progress=None if args.quiet else lambda message: print(message, flush=True)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
