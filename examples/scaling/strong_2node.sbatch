#!/bin/bash
#SBATCH --job-name=ieti-strong-2n
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa
#SBATCH --output=ieti-strong-2n-%j.out
##SBATCH --account=CHANGE_ME
#
# IETI-DP elasticity, strong scaling on two nodes.
# Submit from this directory: sbatch strong_2node.sbatch
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
OUTDIR="$SLURM_SUBMIT_DIR/results-ieti-strong-2n-$SLURM_JOB_ID"
mkdir -p "$OUTDIR"
test -x "$EXE"
export EXE OUTDIR

source "$SLURM_SUBMIT_DIR/common.sh"

# Strong scaling, two nodes: the 512-patch problem on 384 ranks.
run_case "strong_n2_r384" 384 8,8,8
