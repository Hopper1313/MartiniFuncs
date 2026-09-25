#!/usr/bin/env bash
#SBATCH --job-name=martini-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=martini-analysis-%A_%a.out

set -euo pipefail
if (( $# < 1 )); then
    echo "Usage: sbatch --array=1-N postprocess_array.slurm RUNS.tsv [CLI options...]" >&2
    exit 2
fi
manifest=$1
shift
# No header; each line has four tab-separated fields. Paths may contain spaces.
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$manifest")
IFS=$'\t' read -r run_dir output_dir units timestep_ps <<< "$line"
if [[ -z "$run_dir" || -z "$output_dir" || -z "$units" || -z "$timestep_ps" ]]; then
    echo "Missing manifest fields at line ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
fi

# Add site-specific environment activation here if needed.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -m MartiniFuncs.cli run "$run_dir" --output "$output_dir" \
    --units "$units" --timestep-ps "$timestep_ps" \
    --workers "${MARTINI_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}" "$@"
