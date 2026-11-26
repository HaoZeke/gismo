#!/bin/bash
#SBATCH --job-name=ieti-weak-2n
#SBATCH --time=TIME_W2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa
#SBATCH --output=ieti-weak-2n-%j.out
##SBATCH --account=CHANGE_ME
#
# IETI-DP elasticity, weak scaling on two nodes.
# Submit from this directory: sbatch weak_2node.sbatch
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
OUTDIR="$SLURM_SUBMIT_DIR/results-ieti-weak-2n-$SLURM_JOB_ID"
mkdir -p "$OUTDIR"
test -x "$EXE"
export EXE OUTDIR

source "$SLURM_SUBMIT_DIR/common.sh"

# Weak scaling, two nodes: 384 ranks, 3072 patches.
run_case "weak_n2_r384" 384 "$(weak_grid 384)"
