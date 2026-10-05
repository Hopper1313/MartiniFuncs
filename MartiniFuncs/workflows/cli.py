"""Post-simulation analysis with explicit source-frame selection.

Usage (also available as the installed martini-postprocess command)::

    python -m MartiniFuncs run RUN_DIR --output ANALYSIS_DIR --units metal --timestep-ps 0.0005
    python -m MartiniFuncs run RUN_DIR --output PAIR_ANALYSIS --units metal --timestep-ps 0.0005 --no-lag-statistics --shell-radius 3
    python -m MartiniFuncs report ANALYSIS_DIR/cache --output REPORT_DIR --startframe 1000 --endframe 2000
    python -m MartiniFuncs extract ANALYSIS_DIR/cache --output PAIRS_DIR --startframe 1000 --endframe 2000 --shell-radius 3
    python -m MartiniFuncs collect RUN_DIR --output PAIRS_DIR --units metal --timestep-ps 0.0005 --shell-radius 3

collect uses the same indexed parallel frame processing as run, and writes both
pairs.extxyz and clusters.extxyz without generating statistical reports; for
collect, --shell-radius is required. Default stride is 1 (all source frames).
Reaction (topology) events are recorded by default and do not stop run or
collect; --reaction-events stop ends the analysis cleanly before the first
event and off disables the check (see also --reaction-persistence). All pairs
means all accepted pair-model candidates (cpd_n4t: CPD + N4T dienophile) inside
the --reactive-radius screen (default 3.25 Angstrom) and optional criteria, not
every possible Cartesian combination of molecules.

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
requires explicit --temperature-dof. Pair screening uses a pair model
(--pair-model, default cpd_n4t: CPD + N4T dienophile); the pair-threshold flags
are that model's criteria. --no-pairs enables generic molecular statistics. Frame workers are local processes
inside one Slurm task, not MPI ranks; no GPU/model inference occurs here.
--workers N and --batch-size B pass through cache_analysis to analyze_run's
indexed frame process pool (B source frames per analysis task). Both pair backends
support multiple workers. Parsing, molecular reconstruction, reaction screening,
pair screening and shell export run in workers. Temporal COM accumulation,
report calculations and final ordered file writing remain sequential. Reusing a
finished cache skips analysis regardless of the requested worker count.

This module only parses/validates options and dispatches package functions.
run/report support independent lag-statistic, lifetime and frame-summary choices.
--no-lag-statistics skips MSD/correlation work entirely while retaining lifetimes.
CLI stages log timestamps, actual progress counts and 30-second elapsed heartbeats;
--quiet suppresses these messages. Heartbeats are not completion estimates.
Scientific equations live in trajectory.analysis, statistics.md, pairs and
chemistry.cpd_n4t; persistence/export behavior lives in workflows.postprocess. See docs/postprocessing_cli.md
for output schemas, cache reuse, and Slurm submission examples.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

from MartiniFuncs.pairs.model import DEFAULT_PAIR_MODEL, DEFAULT_REACTIVE_RADIUS, get_pair_model, pair_model_names
from MartiniFuncs.workflows.postprocess import cache_analysis, cache_pair_model, export_pairs, write_report
from MartiniFuncs.workflows.progress import log, run_stage


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
    lag_options = parser.add_mutually_exclusive_group()
    lag_options.add_argument('--lag-statistics', nargs='+', choices=['msd', 'vacf', 'force-acf'],
                             help='calculate only these lag statistics (default: all three)')
    lag_options.add_argument('--no-lag-statistics', dest='lag_statistics', action='store_const', const=[],
                             help='skip MSD and velocity/force correlations; retain other requested reports')
    parser.add_argument('--no-pair-lifetimes', action='store_true', help='omit pair lifetime report')
    parser.add_argument('--no-frame-summary', action='store_true', help='omit report frames.csv; cache is unchanged')
    parser.add_argument("--max-lag-ps", type=float, default=100., help="longest correlation/MSD time separation in ps")
    parser.add_argument("--fit-ps", type=float, nargs=2, metavar=("START", "END"), help="explicit diffusion fit lag interval in ps")
    parser.add_argument("--max-gap-frames", type=int, default=0, help="allowed missing retained samples within pair lifetimes")
    parser.add_argument("--stability", action="store_true", help="include scalar ACF (FFT) and batch-error summaries")
    parser.add_argument("--rdf-bins", type=int, help="include molecular COM RDFs with this many radial bins")


def _criteria_fields():
    """Threshold fields of every pair model's PairCriteria, deduplicated: (field, models)."""
    found = {}
    for name in pair_model_names():
        for field in fields(get_pair_model(name).criteria_class):
            found.setdefault(field.name, (field, []))[1].append(name)
    return list(found.values())


def _criteria_options(parser, verb):
    for field, models in _criteria_fields():
        meta = field.metadata
        detail = ", ".join(x for x in (meta.get("help"), meta.get("unit")) if x)
        parser.add_argument("--" + field.name.replace("_", "-"), type=float,
                            help=f"{verb} at or below this threshold ({detail}; pair model "
                                 f"{'/'.join(models)})")


def _criteria(args, model):
    """The model's PairCriteria from the threshold flags; flags of other models are an error."""
    own = {f.name for f in fields(model.criteria_class)}
    given = {f.name: getattr(args, f.name) for f, _ in _criteria_fields() if getattr(args, f.name) is not None}
    foreign = sorted(set(given) - own)
    if foreign:
        raise ValueError(f"pair model {model.name!r} has no threshold "
                         + ", ".join("--" + f.replace("_", "-") for f in foreign))
    return model.criteria_class(**given)


def _selection_options(parser):
    """Pair selection from cached candidates (report/extract): radius and the model's criteria."""
    parser.add_argument("--reactive-radius", type=float,
                        help="select pairs with screen distance below this radius (Angstrom, at most the "
                             "cached radius; cpd_n4t: CPD COM to C=C COM); default: the analysis's own pairs")
    _criteria_options(parser, "select pairs")
    parser.add_argument("--geometry-required", action="store_true",
                        help="select only pairs whose geometry could be measured (implied by any threshold)")


def _selection_kwargs(args):
    model = cache_pair_model(args.cache)
    criteria = _criteria(args, model)
    use_criteria = args.geometry_required or bool(criteria.active())
    return dict(reactive_radius=args.reactive_radius, pair_criteria=criteria if use_criteria else None)


def _report_kwargs(args):
    """Pass frame bounds and lag settings unchanged to workflows.postprocess.write_report.

    This conversion performs no frame-to-time conversion: cached timestamps
    determine lag spacing after the report selects the requested source frames.
    """
    return dict(startframe=args.startframe, endframe=args.endframe,
                max_lag_ps=args.max_lag_ps, fit_interval_ps=args.fit_ps, max_gap_frames=args.max_gap_frames,
                stability=args.stability, rdf_bins=args.rdf_bins,
                lag_statistics=args.lag_statistics, pair_lifetimes=not args.no_pair_lifetimes,
                frame_summary=not args.no_frame_summary)


def _analysis_options(parser):
    """Shared source discovery, pair selection and process-pool options.

    run and collect must use identical screening and frame-stride conventions.
    Worker options apply to analysis and export; final writes remain ordered.
    """
    parser.add_argument("directory", type=Path, help="simulation directory; unique input/data/dump discovered automatically")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units", required=True, choices=["metal"], help="source LAMMPS units; current statistics support metal")
    parser.add_argument("--timestep-ps", required=True, type=float, help="ps per MD step, not per saved frame")
    for name in ("input", "dump", "data"):
        parser.add_argument(f"--{name}-file", help="optional relative or absolute filename override")
    parser.add_argument("--type-symbols", type=Path, help='JSON type map, e.g. {"1":"C","2":"H"}')
    parser.add_argument("--frame-stride", type=int, default=1, help="retain every Nth source frame, anchored at frame 0; reading/unwrapping still uses every frame")
    parser.add_argument("--max-frames", type=int, help="preview only: cap source frames read")
    parser.add_argument("--bond-scale", type=float, default=1.3)
    parser.add_argument("--reaction-events", choices=["record", "stop", "off"], default="record",
                        help="topology changes (bonds forming between molecules, or molecules splitting), checked "
                             "on every source frame: record = log them and keep going (default); stop = end the "
                             "analysis cleanly before the first one; off = do not check")
    parser.add_argument("--reaction-persistence", type=int, default=5,
                        help="source frames a topology change must last to count as an event (default: 5)")
    parser.add_argument("--reaction-atom-counts", type=int, nargs=2, metavar=("A", "B"),
                        help="legacy screen, off by default: abort with an error at the first A+B-atom "
                             "component (prefer --reaction-events)")
    parser.add_argument("--pair-model", choices=pair_model_names(), default=DEFAULT_PAIR_MODEL,
                        help=f"pair chemistry (default {DEFAULT_PAIR_MODEL}: CPD + N4T dienophile)")
    parser.add_argument("--pair-backend", choices=["cached", "reference"], default="cached")
    parser.add_argument("--reactive-radius", type=float, default=DEFAULT_REACTIVE_RADIUS,
                        help="pair screen radius cached by the analysis in Angstrom (cpd_n4t: CPD COM to "
                             f"C=C COM; default {DEFAULT_REACTIVE_RADIUS:g}); report/extract can later select "
                             "any radius up to it")
    parser.add_argument("--orientations", action="store_true",
                        help="also cache each molecule's long axis and plane normal per frame (rotational dynamics)")
    parser.add_argument("--workers", type=int, default=1, help="local frame-processing and export processes, not MPI ranks (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="source frames per analysis chunk; retained frames per export chunk (default: 32)")
    _criteria_options(parser, "keep pairs")
    parser.add_argument("--quiet", action="store_true")


def build_parser():
    """Return the CLI parser; parsing does not read or write simulation data.

    run and collect share cache settings (physics, topology, sampling, workers).
    report owns reusable statistics settings. extract owns geometry export and
    shell radius. collect skips reports and requires shells. All commands share
    the same source-frame selection convention.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="analyze a finished simulation, cache results, report, and export pairs")
    _analysis_options(run)
    run.add_argument("--temperature-dof", help="positive number or 3N-3; omitted means no temperature")
    run.add_argument("--no-pairs", action="store_true", help="generic molecular statistics without chemistry-specific pair screening")
    run.add_argument("--no-export-pairs", action="store_true", help="cache pair identities/lifetimes without a second structure-export pass")
    run.add_argument("--shell-radius", type=float, help="also export whole-molecule solvent shells, radius in Angstrom")
    run.add_argument("--report-name", default="default", help="single folder name below output/reports")
    _report_options(run)

    collect = commands.add_parser("collect", help="collect every selected frame's pairs and pairs+shells as Extended XYZ")
    _analysis_options(collect)
    _frame_options(collect)
    collect.add_argument("--shell-radius", type=float, required=True,
                         help="atom-to-pair shell cutoff in Angstrom; retain whole neighboring molecules")
    collect.set_defaults(no_pairs=False, no_export_pairs=False, temperature_dof=None,
                         report_name=None)

    report = commands.add_parser("report", help="regenerate statistics from cache without reading simulation files")
    report.add_argument("cache", type=Path, help="the output/cache directory from run")
    report.add_argument("--output", type=Path, required=True, help="new report directory")
    report.add_argument('--quiet', action='store_true', help='suppress progress and heartbeat logging')
    _report_options(report)
    _selection_options(report)

    extract = commands.add_parser("extract", help="export cached accepted pairs and optional shells from source frames")
    extract.add_argument("cache", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    _frame_options(extract)
    extract.add_argument("--shell-radius", type=float)
    extract.add_argument("--quiet", action="store_true")
    extract.add_argument("--workers", type=int, default=1, help="local frame/export worker processes")
    extract.add_argument("--batch-size", type=int, default=32, help="selected frames per export chunk")
    _selection_options(extract)
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
    progress = None if args.quiet else log

    def stage(label, function, *values, **options):
        return run_stage(label, lambda: function(*values, progress=progress, **options), progress=progress)
    try:
        if (getattr(args, 'fit_ps', None) is not None and args.lag_statistics is not None
                and 'msd' not in args.lag_statistics):
            raise ValueError('--fit-ps requires msd; remove --no-lag-statistics or include msd in --lag-statistics')
        if args.startframe < 0 or (args.endframe is not None and args.endframe <= args.startframe):
            raise ValueError("Require 0 <= startframe < endframe; omit endframe to use the cached end")
        if hasattr(args, "workers"):
            if args.workers < 1 or args.batch_size < 1:
                raise ValueError("workers and batch-size must be positive")
            if getattr(args, "reaction_persistence", 1) < 1:
                raise ValueError("--reaction-persistence must be a positive number of frames")
            allocated = os.environ.get("SLURM_CPUS_PER_TASK")
            if allocated is not None and args.workers > int(allocated):
                raise ValueError("--workers exceeds SLURM_CPUS_PER_TASK; request more CPUs or reduce workers")
        if args.command in {"run", "collect"}:
            if args.command == "run" and (args.report_name in {"", ".", ".."} or any(c in args.report_name for c in "/\\:")):
                raise ValueError("--report-name must be one folder name")
            if args.shell_radius is not None and (args.no_pairs or args.no_export_pairs):
                raise ValueError("--shell-radius requires pair analysis and pair exports")
            mapping = None
            if args.type_symbols:
                mapping = {int(k): v for k, v in json.loads(args.type_symbols.read_text(encoding="utf-8")).items()}
            dof = args.temperature_dof
            if dof is not None and dof != "3N-3":
                dof = float(dof)
            criteria = _criteria(args, get_pair_model(args.pair_model))
            args.output.mkdir(parents=True, exist_ok=True)
            cache = args.output / "cache"
            # The in-memory result is reused by the report and export stages.
            cached = stage('Cache analysis/write', cache_analysis, args.directory, cache, units=args.units, timestep_ps=args.timestep_ps,
                           input_file=args.input_file, dump_file=args.dump_file, data_file=args.data_file,
                           frame_stride=args.frame_stride, max_frames=args.max_frames, bond_scale=args.bond_scale,
                           type_symbols=mapping, degrees_of_freedom=dof, compute_pairs=not args.no_pairs,
                           reaction_atom_counts=args.reaction_atom_counts,
                           **({} if args.reaction_events == "off" else
                              {"on_topology_change": {"record": "record", "stop": "truncate"}[args.reaction_events],
                               "topology_persistence": args.reaction_persistence}),
                           pair_criteria=criteria, pair_backend=args.pair_backend, pair_workers=args.workers,
                           pair_model=args.pair_model, reactive_radius=args.reactive_radius,
                           orientations=args.orientations, pair_batch_size=args.batch_size)
            report = None
            if args.command == "run":
                report = stage('Statistical report', write_report, cache, args.output / "reports" / args.report_name,
                               preloaded=cached, **_report_kwargs(args))
            if not args.no_pairs and not args.no_export_pairs:
                # Separate explicitly selected exports from the full-cache product.
                pair_folder = "pairs" if args.startframe == 0 and args.endframe is None else f"pairs_frames_{args.startframe}_{args.endframe if args.endframe is not None else 'end'}"
                stage('Pair/shell export', export_pairs, cache, args.output / pair_folder, shell_radius=args.shell_radius,
                             startframe=args.startframe, endframe=args.endframe,
                             workers=args.workers, batch_size=args.batch_size, preloaded=cached)
            print(f"Cache: {cache.resolve()}")
            if report is not None:
                print(f"Report: {report}")
            if not args.no_pairs and not args.no_export_pairs:
                print(f"Pair and cluster exports: {(args.output / pair_folder).resolve()}")
        elif args.command == "report":
            print(stage('Statistical report', write_report, args.cache, args.output, **_report_kwargs(args),
                        **_selection_kwargs(args)))
        else:
            print(stage('Pair/shell export', export_pairs, args.cache, args.output, shell_radius=args.shell_radius,
                               startframe=args.startframe, endframe=args.endframe,
                               workers=args.workers, batch_size=args.batch_size, **_selection_kwargs(args)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
