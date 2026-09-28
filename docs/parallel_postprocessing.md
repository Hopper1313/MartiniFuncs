# Parallel full-trajectory processing

`run` and `collect` process the entire trajectory by default. Existing commands
now use `--workers` for complete frame parsing/preparation and for structure
export, rather than only for pair screening. No new selection flags are needed.

```bash
martini-postprocess run ./ --output ./postprocess --units metal --timestep-ps 0.0005 --shell-radius 3 --reaction-atom-counts 11 23 --workers 8 --batch-size 32
```

Use `collect` instead of `run` to omit statistical reports. `extract` also accepts
`--workers` and `--batch-size`. In Python, `analyze_run`/`cache_analysis` retain
their existing argument names `pair_workers` and `pair_batch_size` for compatibility;
`export_pairs` uses `workers` and `batch_size`. Spawned processes require the usual
`if __name__ == '__main__':` guard in standalone Python scripts.

## Execution

1. Parse frame zero once to establish common atom IDs, molecular membership and
   reactive sites. Every worker uses that same topology.
2. Scan the uncompressed dump for frame byte offsets. This scan does not parse
   atom columns; the temporary index uses eight bytes per frame boundary.
3. Workers open independent file handles and seek to contiguous chunks. They
   parse atoms/forces, validate IDs and topology, reconstruct whole molecules,
   screen possible reaction products when enabled, and identify accepted pairs.
4. Workers return molecular summaries (COM, velocity, net force and scalars), not
   full atomic frames. The parent receives chunks in order and accumulates temporal
   COM displacements. Statistical reports remain in the parent.
5. During export, workers independently parse chunks, validate connectivity,
   construct pairs and whole-molecule shells, and serialize Extended XYZ records.
   The parent writes those records in trajectory order to the existing files.

Source frames are not split into individual files. Both cached and reference
pair-selection backends support parallel frame processing. Workers do not launch
nested pair-worker pools. The export stage currently builds its own byte index;
these indices are not persistent caches.

At most `2 * workers` tasks are queued; `batch-size` is source frames per analysis
chunk and retained frames per export chunk. Frame zero is initialized separately.
Larger chunks reduce scheduling overhead but increase buffered memory. More
workers cannot exceed the storage system's read bandwidth or accelerate serial
report calculations. The parent also needs CPU time. Benchmark modest worker
counts before allocating a full node; there is no promised linear speedup.

## Boundary correctness

For wrapped molecular COMs w and instantaneous orthorhombic lengths L, the
existing TOR convention remains

\[
\Delta R_t = w_t-w_{t-1}-L_t\left\lfloor\frac{w_t-w_{t-1}}{L_t}+\frac12\right\rfloor,
\qquad R_t=R_{t-1}+\Delta R_t.
\]

The ordered parent applies this update across every source frame, including
chunk boundaries and frames omitted by existing stride settings. Parallelization
does not extend TOR to sheared/triclinic cells. Pair lifetimes and correlations
are computed over the assembled trajectory, not averaged from separate chunks.
Possible-product exceptions retain their atom IDs and are delivered in frame
order. Work may already be in flight beyond a detected product, but no completed
cache/export manifest is published for a failed stage.

Structure ordinals are assigned from cached pair counts before export tasks are
submitted, keeping pair/shell files and index.jsonl aligned. Source forces retain
their existing units and precision. Export byte content should match serial
processing regardless of worker count or chunk boundaries.

## Timing and remaining scale limits

The CLI prints elapsed seconds separately for cache analysis/writing, statistical
reporting, and pair/shell export. `Processed ... frames` means ordered preparation
has completed, not merely that text was read. Its `time ... ps` is simulation time.

Only in-flight atomic chunks are bounded in memory. **The existing cache builder
still retains full molecular arrays and pair histories**, and the final outputs
are single ordered files. This update does not implement disk-backed accumulation,
resumable chunk products or multi-node scheduling. For millions of frames these
remain important limits: size memory for the molecular history plus worker buffers,
and benchmark a representative trajectory before a large allocation.

Completed matching caches/products retain the existing reuse policy. Interrupted
stages are not resumable within a trajectory. Stage timings of reused outputs are
not useful performance comparisons; use distinct output directories for benchmarks.

## Validation

Regression checks compare serial and parallel molecular arrays and byte-identical
pair/shell exports, including source forces and indices. Synthetic cases check TOR
across periodic crossings with changing box lengths, CRLF byte offsets, and a
reaction detected in a worker before frame striding.

A local Windows trial on the full 2001-frame `continued_run` trajectory took
72.1 s with one worker and 15.7 s with four (batch size 32). This measured analysis
with the cached pair backend, without reaction screening, report generation or
structure export. Pair identities/shifts, scalar histories, COM trajectories and
lifetimes agreed. This is a single local comparison, not a cluster scaling claim;
filesystem cache state was not controlled. Reproduce using:

```bash
python -m examples.benchmark_pair_analysis RUN_DIR --units metal --timestep-ps 0.0005 --max-frames 2001 --workers 1 4 --batch-size 32 --skip-reference
```

The frame cap above belongs to that benchmark example and equals this dataset's
full length. Normal `run`/`collect` commands process the entire trajectory by default.
