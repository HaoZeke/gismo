#!/usr/bin/env python3
"""Tables from the RESULT lines of ieti_elasticity_mpi_example.

Reads the key=value lines, groups repeats of the same configuration (median),
and prints speedup, efficiency, time split and iteration tables.

    ieti_tables.py strong results.log
    ieti_tables.py weak   results.log --csv > weak.csv

Strong scaling: speedup = T(base)/T(N), efficiency = speedup/(N/N_base).
Weak scaling:   efficiency = T(base)/T(N) at fixed work per rank.
The base is the run with the fewest ranks.
"""
import argparse
import statistics
import sys

SETUP = ("t_assembly", "t_factor", "t_coarse", "t_rhs")
NUMERIC = ("ranks", "threads", "patches", "unknowns", "multipliers", "primal",
           "iterations", "t_mapper", "t_assembly", "t_factor", "t_coarse",
           "t_rhs", "t_solve", "t_iter", "t_total", "t_comm", "cond",
           "relres", "compliance")


def parse(lines):
    """Return a list of dicts, one per RESULT line."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("RESULT "):
            continue
        row = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                row[k] = v
        for k in NUMERIC:
            if k in row:
                try:
                    row[k] = float(row[k])
                except ValueError:
                    row[k] = float("nan")
        rows.append(row)
    return rows


def aggregate(rows):
    """Median over repeats of equal (ranks, patches, degree, refine, primal)."""
    groups = {}
    for r in rows:
        key = (int(r["ranks"]), int(r["patches"]), r.get("degree"),
               r.get("refine"), r.get("primal_type"))
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        m = dict(g[0])
        for k in NUMERIC:
            if k in g[0]:
                m[k] = statistics.median(x[k] for x in g)
        m["setup"] = sum(m[k] for k in SETUP)
        m["repeats"] = len(g)
        m["failed"] = sum(1 for x in g if x.get("status") != "PASS")
        out.append(m)
    return out


def fmt(x, w=10, p=3):
    if isinstance(x, float):
        if abs(x) < 1e-3 or abs(x) >= 1e5:
            return "%*.*g" % (w, p + 1, x)
        return "%*.*f" % (w, p, x)
    return "%*s" % (w, x)


def table(header, body, csv):
    if csv:
        print(",".join(header))
        for r in body:
            print(",".join(str(x) for x in r))
        return
    print(" ".join("%10s" % h for h in header))
    for r in body:
        print(" ".join(fmt(x) for x in r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("strong", "weak"))
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    lines = []
    for fn in a.logs:
        with open(fn) as f:
            lines.extend(f)
    rows = aggregate(parse(lines))
    if not rows:
        sys.exit("no RESULT lines found")
    base = min(rows, key=lambda r: r["ranks"])

    print("# %s scaling, base = %d ranks, repeats per row: %s" %
          (a.kind, base["ranks"], "/".join(str(r["repeats"]) for r in rows)))
    bad = [r for r in rows if r["failed"]]
    if bad:
        print("# WARNING: runs with status != PASS at ranks %s" %
              ", ".join(str(int(r["ranks"])) for r in bad))

    print("\n## time to solution")
    body = []
    for r in rows:
        n = r["ranks"] / base["ranks"]
        if a.kind == "strong":
            s_tot = base["t_total"] / r["t_total"]
            s_set = base["setup"] / r["setup"]
            s_sol = base["t_solve"] / r["t_solve"]
            body.append([int(r["ranks"]), int(r["unknowns"]), r["setup"], r["t_solve"],
                         r["t_total"], s_tot, s_tot / n, s_set, s_set / n, s_sol, s_sol / n])
            hdr = ["ranks", "unknowns", "t_setup", "t_solve", "t_total", "speedup",
                   "eff", "sp_setup", "eff_setup", "sp_solve", "eff_solve"]
        else:
            e_tot = base["t_total"] / r["t_total"]
            e_set = base["setup"] / r["setup"]
            e_it = base["t_iter"] / r["t_iter"]
            body.append([int(r["ranks"]), int(r["unknowns"]), int(r["unknowns"] // r["ranks"]),
                         r["setup"], r["t_solve"], r["t_total"], e_tot, e_set, e_it])
            hdr = ["ranks", "unknowns", "unk/rank", "t_setup", "t_solve", "t_total",
                   "eff_total", "eff_setup", "eff_t_iter"]
    table(hdr, body, a.csv)

    print("\n## time split (max over ranks, seconds and fraction of t_total)")
    hdr = ["ranks", "mapper", "assembly", "factor", "coarse", "rhs", "solve", "comm",
           "f_assem", "f_factor", "f_solve", "f_comm"]
    body = []
    for r in rows:
        t = r["t_total"]
        body.append([int(r["ranks"]), r["t_mapper"], r["t_assembly"], r["t_factor"],
                     r["t_coarse"], r["t_rhs"], r["t_solve"], r["t_comm"],
                     r["t_assembly"] / t, r["t_factor"] / t, r["t_solve"] / t,
                     r["t_comm"] / t])
    table(hdr, body, a.csv)

    print("\n## iterations and conditioning")
    hdr = ["ranks", "patches", "multipliers", "primal", "iters", "t_iter", "cond", "relres"]
    body = [[int(r["ranks"]), int(r["patches"]), int(r["multipliers"]), int(r["primal"]),
             int(r["iterations"]), r["t_iter"], r["cond"], r["relres"]] for r in rows]
    table(hdr, body, a.csv)


if __name__ == "__main__":
    main()
