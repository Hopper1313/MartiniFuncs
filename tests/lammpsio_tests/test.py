"""Run from the repository root: python -m tests.lammpsio_tests.test.

No UMA model is needed for extraction and geometry diagnostics.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json

from MartiniFuncs.io.lammps import read_lammps_frame, read_lammps_input_settings
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_all_prereactive_pairs
from MartiniFuncs.pairs.structures import (get_pair_shell_clusters, get_cluster_positions_and_forces,
                                           pair_data_to_extxyz)
from MartiniFuncs.chemistry.cpd_n4t.selection import PairCriteria, assess_prereactive_pairs


def main():
    fixtures = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=fixtures / "multiple_frames.lammpstrj")
    parser.add_argument("--input", type=Path, default=fixtures / "infile.in")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.3)
    parser.add_argument("--max-axis-angle", type=float)
    parser.add_argument("--max-forming-distance", type=float)
    parser.add_argument("--output", type=Path, help="optional finite-cluster extxyz output")
    args = parser.parse_args()
    settings = read_lammps_input_settings(args.input)
    frame = read_lammps_frame(args.dump, index=args.index, units=settings["units"])
    pairs, shifts = get_all_prereactive_pairs(frame.atoms, scale=args.scale)
    criteria = PairCriteria(max_axis_angle=args.max_axis_angle,
                            max_forming_distance=args.max_forming_distance)
    assessments = assess_prereactive_pairs(frame.atoms, pairs, shifts,
                                           criteria=criteria, scale=args.scale)
    keep = [r.pair_index for r in assessments if r.accepted]
    clusters = get_pair_shell_clusters(frame.atoms, [pairs[i] for i in keep],
                                       [shifts[i] for i in keep], scale=args.scale)
    records = get_cluster_positions_and_forces(frame.atoms, frame, clusters)
    print(json.dumps({"timestep": frame.timestep, "criteria": asdict(criteria),
                      "assessments": [asdict(r) | {"accepted": r.accepted} for r in assessments],
                      "cluster_sizes": [len(r["positions"]) for r in records.values()]}, indent=2))
    if args.output:
        if not records:
            parser.error("No accepted candidates; no output was written")
        pair_data_to_extxyz(records, args.output)


if __name__ == "__main__":
    main()
