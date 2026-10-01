# Collect pairs and their environments from every frame

From the environment containing MartiniFuncs, run this in PowerShell:

```powershell
$run = "C:/Users/nicho/Documents/Martini collab/uma_data/smaller_system/continued_run"
python -m MartiniFuncs collect "$run" --output "./pair_exports" --units metal --timestep-ps 0.0005 --shell-radius 3.0 --workers 4 --batch-size 32
```

`martini-postprocess collect` is the installed equivalent. Choose your physical
shell cutoff; 3 Å is an example. Source input/dump/data discovery is shared with
notebook 04. Use `--dump-file`, `--input-file`, `--data-file` or `--type-symbols`
when necessary. The default stride is **1**, covering every source frame.

## What is collected

Every **accepted CPD/MAL candidate in every retained frame** becomes one pair entry
and one cluster entry. The same molecular pair in 20 frames produces 20 entries.
This uses notebook 04's screening: the whole CPD COM must be within 3.25 Å of the
reactive MAL C=C center under periodic boundaries, followed by geometry validity
and any optional `--max-*-angle`/distance criteria. It is not every possible CPD×MAL
combination. Default numerical orientation limits are unset.

For shell radius R, neighboring atoms within R of any pair atom select their
whole molecules, using periodic images. Exported coordinates form finite, whole
structures in Å. Source forces are copied by persistent atom ID in eV/Å; periodic
translations do not change force vectors. No UMA evaluation or force modification
occurs. Pair atoms precede shell atoms, with CPD then MAL.

The command defaults to checking every source frame for a possible 11+23 atom
product and stops on a candidate. This exact-size distance-graph screen cannot
establish a reaction or rule out other reactions. Stopping prevents continuing
fixed-species collection across detected membership changes.

## Files and identity

```text
pair_exports/
  cache/                 reusable analysis and accepted pair identities
  pairs/
    pairs.extxyz         one isolated pair per encounter/frame
    clusters.extxyz      corresponding pair plus whole-molecule shell
    index.jsonl          frame, time, candidate and proposed-bond geometry
    frames.csv           every selected frame, including zero-pair frames
    manifest.json        completion marker, settings and source fingerprints
```

Both XYZ files have the same entry order. Each entry has `structure_index`,
`frame_index`, `timestep`, `time_ps`, `candidate_index`, source-cell metadata and
force units. Structure index is zero-based across that export; candidate index
is the accepted-pair ordinal within its frame. It is not a lifetime ID.

Each atom carries:

| Array | Meaning |
|---|---|
| `local_index` | Zero-based row in this exported pair/cluster |
| `global_index` | Persistent package atom ID; use `id_offset` for its relation to LAMMPS IDs |
| `lammps_id` | Original LAMMPS atom ID, explicitly stored |
| `frame_local_index` | Zero-based row in the ID-sorted source ASE frame, not the raw dump line number |
| `pair_role` | `CPD`, `MAL`, or `shell` |
| `lammps_forces` | Three real columns: fx, fy, fz, in eV/Å |

The `lammps_forces:R:3` Extended XYZ property contains all three force components;
it is not just a force magnitude. Shell forces have the same source-ID mapping
as pair forces. Floating values use 17 significant digits, avoiding the default
ASE writer's eight-decimal truncation of small force components. Roles identify shell atoms collectively; use atom IDs to track
individual solvent molecules if needed.

```python
from ase.io import iread

for cluster in iread("pair_exports/pairs/clusters.extxyz", format="extxyz"):
    xyz = cluster.positions
    fx, fy, fz = cluster.arrays["lammps_forces"].T
    cpd = cluster.arrays["pair_role"] == "CPD"
    cpd_local = cluster.arrays["local_index"][cpd]
    cpd_global = cluster.arrays["global_index"][cpd]
    source_frame = cluster.info["frame_index"]
```

`frames.csv` gives `pair_count` and `first_structure_index`; an empty frame adds
no XYZ entry. With zero accepted pairs in the entire selection, the XYZ files
are empty and the frame table still documents coverage. The corresponding
`index.jsonl` geometry includes global, pair-local and frame-local forming-bond
indices. Keep the source dump and cache for alternative exports.

## Parallelism and reuse

The frame process pool uses `--workers` and `--batch-size` (source frames per
analysis chunk; selected frames per export chunk). Workers parse indexed byte
ranges, reconstruct molecules, check reactions and screen pairs. Export workers
parse frames and construct pairs/shells. Temporal COM accumulation and final
ordered file writing remain sequential. Use one Slurm
task and allocate at least the requested workers; no GPU is used. This command
does not generate diffusion or stability reports. It still uses the existing
analysis cache, including molecular arrays, then makes one source pass for export.

Identical completed requests are reused; conflicting or partial output is not
overwritten. To change only shell radius, reuse the cache with a new directory:

```console
python -m MartiniFuncs extract pair_exports/cache --output shells_5A --shell-radius 5
```

`--startframe`/`--endframe` use zero-based inclusive/exclusive source indices;
omit them for all frames. Explicit selections create `pairs_frames_START_END/`.
`--frame-stride` and `--max-frames` deliberately reduce available sampling; omit
them for the requested every-frame collection. Large output scales with the
number of encounters and shell atoms, not just the number of source frames.

See [parallel processing](parallel_postprocessing.md) for scheduling, timings,
memory limits and serial-equivalence checks.
