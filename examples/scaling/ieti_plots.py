#!/usr/bin/env python3
"""Figures from the RESULT lines of ieti_elasticity_mpi_example.

    ieti_plots.py strong results.log -o strong.pdf
    ieti_plots.py weak   results.log -o weak.pdf

Needs matplotlib; without it the script exits with a message and the tables
of ieti_tables.py remain available.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ieti_tables import aggregate, parse  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is not available; use ieti_tables.py for the tables")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("strong", "weak"))
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    lines = []
    for fn in a.logs:
        with open(fn) as f:
            lines.extend(f)
    rows = aggregate(parse(lines))
    if not rows:
        sys.exit("no RESULT lines found")
    base = min(rows, key=lambda r: r["ranks"])
    ranks = [r["ranks"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: speedup (strong) or efficiency (weak)
    if a.kind == "strong":
        ax[0].loglog(ranks, [base["t_total"] / r["t_total"] for r in rows], "o-", label="total")
        ax[0].loglog(ranks, [base["t_solve"] / r["t_solve"] for r in rows], "s--", label="CG solve")
        ax[0].loglog(ranks, [base["setup"] / r["setup"] for r in rows], "^:", label="setup")
        ax[0].loglog(ranks, [n / base["ranks"] for n in ranks], "k-", lw=0.5, label="ideal")
        ax[0].set_ylabel("speedup")
        ax[0].set_title("strong scaling, %d unknowns" % rows[0]["unknowns"])
    else:
        ax[0].semilogx(ranks, [base["t_total"] / r["t_total"] for r in rows], "o-", label="total")
        ax[0].semilogx(ranks, [base["t_iter"] / r["t_iter"] for r in rows], "s--",
                       label="time per iteration")
        ax[0].semilogx(ranks, [base["setup"] / r["setup"] for r in rows], "^:", label="setup")
        ax[0].axhline(1.0, color="k", lw=0.5)
        ax[0].set_ylabel("efficiency")
        ax[0].set_title("weak scaling, %d unknowns per rank" %
                        (rows[0]["unknowns"] // rows[0]["ranks"]))
    ax[0].set_xlabel("MPI ranks")
    ax[0].legend()
    ax[0].grid(True, which="both", lw=0.3)

    # Panel 2: stacked time split
    labels = ["assembly", "factor", "coarse", "rhs", "solve"]
    keys = ["t_assembly", "t_factor", "t_coarse", "t_rhs", "t_solve"]
    xs = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for lab, key in zip(labels, keys):
        vals = [r[key] for r in rows]
        ax[1].bar(xs, vals, bottom=bottom, label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([str(int(r)) for r in ranks])
    ax[1].set_xlabel("MPI ranks")
    ax[1].set_ylabel("time (s)")
    ax[1].set_yscale("log")
    ax[1].set_title("time split")
    ax[1].legend()

    # Panel 3: iterations
    ax[2].semilogx(ranks, [r["iterations"] for r in rows], "o-")
    ax[2].set_xlabel("MPI ranks")
    ax[2].set_ylabel("CG iterations")
    ax[2].set_title("iterations")
    ax[2].set_ylim(bottom=0)
    ax[2].grid(True, which="both", lw=0.3)

    fig.tight_layout()
    out = a.output or ("ieti_%s.pdf" % a.kind)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
