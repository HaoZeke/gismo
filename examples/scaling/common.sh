# Shared helpers for the IETI-DP elasticity scaling jobs. Sourced by the
# job scripts; defines the problem sizes and the run function.
#
# Problem sizes (degree 3, 3 refinements per patch, 10 basis functions per
# patch and direction):
#   unknowns = 3 * (nx*9+1) * (ny*9+1) * (nz*9+1) minus the clamped face;
#   the strong problem of 8x8x8 patches has 1.57 million unknowns, the weak
#   series has eight patches (about 24 thousand unknowns) per rank.

# Executable and output directory are set by the job script.
: "${EXE:?EXE is not set}"
: "${OUTDIR:?OUTDIR is not set}"

DEGREE=${DEGREE:-3}
REFINE=${REFINE:-3}
PRIMAL=${PRIMAL:-ve}
TOL=${TOL:-1e-8}
REPEATS=${REPEATS:-3}

# Patch grids for the weak-scaling series: eight patches per rank
# (about 24 thousand unknowns per rank at degree 3, 3 refinements).
weak_grid() {
  case "$1" in
    1)    echo 2,2,2 ;;
    2)    echo 4,2,2 ;;
    4)    echo 4,4,2 ;;
    8)    echo 4,4,4 ;;
    16)   echo 8,4,4 ;;
    32)   echo 8,8,4 ;;
    64)   echo 8,8,8 ;;
    128)  echo 16,8,8 ;;
    192)  echo 16,12,8 ;;
    384)  echo 16,16,12 ;;
    768)  echo 24,16,16 ;;
    *)    echo "no weak grid for $1 ranks" >&2; return 1 ;;
  esac
}

# One run: run_case TAG RANKS GRID [REPEATS]. OMP_NUM_THREADS=1 and one rank
# per core.
run_case() {
  local tag=$1 ranks=$2 grid=$3 reps=${4:-$REPEATS} rep
  for rep in $(seq 1 "$reps"); do
    srun --ntasks="$ranks" --cpus-per-task=1 --cpu-bind=cores --distribution=block:block \
      "$EXE" --quiet --patches "$grid" --degree "$DEGREE" --refine "$REFINE" \
      --primal "$PRIMAL" --tol "$TOL" --partition morton --tag "${tag}_rep${rep}" \
      | tee -a "$OUTDIR/results.log"
  done
}
