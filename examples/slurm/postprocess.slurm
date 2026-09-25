#!/usr/bin/env bash
#SBATCH --job-name=martini-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=martini-analysis-%j.out

set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: sbatch postprocess.slurm RUN_DIR OUTPUT_DIR UNITS TIMESTEP_PS [CLI options...]" >&2
    exit 2
fi
run_dir=$1
output_dir=$2
units=$3
timestep_ps=$4
shift 4

# Activate your site's Python environment before submission, or add its module/
# activation commands here. MartiniFuncs must be installed in that environment.
# Keep numerical libraries from starting extra threads in every worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m MartiniFuncs.cli run "$run_dir" --output "$output_dir" \
    --units "$units" --timestep-ps "$timestep_ps" \
    --workers "${MARTINI_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}" "$@"
