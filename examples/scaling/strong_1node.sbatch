#!/bin/bash
#SBATCH --job-name=ieti-strong-1n
#SBATCH --time=TIME_S1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa
#SBATCH --output=ieti-strong-1n-%j.out
##SBATCH --account=CHANGE_ME
#
# IETI-DP elasticity, strong scaling inside one node.
# Submit from this directory: sbatch strong_1node.sbatch
set -euo pipefail

# ---- site variables: edit for the cluster -------------------------------
# Module names differ per site; these are placeholders.
MODULES_TOOLCHAIN="foss/2026.1"
MODULES_EXTRA="CMake/3.31.11-GCCcore-15.2.0"
GISMO_BUILD_DIR="${GISMO_BUILD_DIR:-$HOME/gismo-build}"
# -------------------------------------------------------------------------

module purge
module load $MODULES_TOOLCHAIN $MODULES_EXTRA

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
EXE="$GISMO_BUILD_DIR/bin/ieti_elasticity_mpi_example"
OUTDIR="$SLURM_SUBMIT_DIR/results-ieti-strong-1n-$SLURM_JOB_ID"
mkdir -p "$OUTDIR"
test -x "$EXE"
export EXE OUTDIR

source "$SLURM_SUBMIT_DIR/common.sh"

# Strong scaling on one node: 512 patches (8x8x8), degree 4, 3 refinements,
# about 2.1 million unknowns, ranks 1, 2, 4, ..., 128, 192.
for R in 1 2 4 8 16 32 64 128 192; do
  run_case "strong_n1_r${R}" "$R" 8,8,8
done
