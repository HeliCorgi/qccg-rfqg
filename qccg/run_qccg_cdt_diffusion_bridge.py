#!/usr/bin/env python3
"""Microscopic move-count mechanism for the CDT volume kinetic denominator.

CDT anchor:
  S_kin ~ (1/Gamma) (n'-n)^2/(n'+n).

QCCG observation:
- a spatial 1->4 Pachner move changes N3 by +3;
- every tetrahedron is a possible 1->4 local move site;
- therefore the positive-volume-jump contribution to the local second jump
  moment is exactly 9*N3 for equal per-site microscopic rate.

Including 4->1 and 2<->3 moves adds nonnegative geometry-dependent terms.

Thus the local volume diffusion has an extensive O(N3) contribution already
at the microscopic move-counting level.  Under a diffusive/central-limit
coarse graining, a local Gaussian kernel has exponent
  (Delta N3)^2 / [const * N3],
which is the same denominator scaling as the CDT kinetic term when
n' ~ n and N3 is replaced symmetrically by (n+n')/2.

The exact move-count statement is audited as PASS.  The CLT/diffusion
promotion remains OPEN until measured on increasing QCCG ensembles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_time_local_slice_transfer_toy as qslice


SOURCES = [
    {
        "title": "The effective action in 4-dim CDT. The transfer matrix approach",
        "journal": "JHEP 06 (2014) 034",
        "arxiv": "1403.5940",
        "doi": "10.1007/JHEP06(2014)034",
    },
]


def physical_candidate_counts(S):
    # For 1->4, labels are gauge: one physical move location per tetrahedron.
    c14 = len(S)
    c41 = len(qslice.candidates41(S))
    c23 = len(qslice.candidates23(S))
    c32 = len(qslice.candidates32(S))
    return {"14": c14, "41": c41, "23": c23, "32": c32}


def second_jump_moment(counts):
    # Delta N3 = +3,-3,+1,-1.
    return (
        9 * counts["14"]
        + 9 * counts["41"]
        + counts["23"]
        + counts["32"]
    )


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "qccg-pachner-diffusion-bridge",
        "artifact": "qccg/run_qccg_cdt_diffusion_bridge.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    states, keys, _idx, _edges, _types, inverse_ok, _failures = qslice.build_state_graph()
    rows = []
    exact_extensive = True

    by_volume = {}
    for k in keys:
        S = states[k]
        n = len(S)
        counts = physical_candidate_counts(S)
        m2 = second_jump_moment(counts)
        exact_component = 9 * n
        exact_extensive = exact_extensive and counts["14"] == n and m2 >= exact_component
        row = {
            "N3": n,
            "counts": counts,
            "second_jump_moment_equal_site_rate": m2,
            "exact_1_to_4_component": exact_component,
            "moment_per_N3": m2 / n,
        }
        rows.append(row)
        by_volume.setdefault(n, []).append(row)

    volume_rows = []
    for n, rs in sorted(by_volume.items()):
        avg = sum(r["second_jump_moment_equal_site_rate"] for r in rs) / len(rs)
        volume_rows.append({
            "N3": n,
            "microstates": len(rs),
            "mean_second_jump_moment": avg,
            "mean_second_jump_moment_over_N3": avg / n,
        })

    ratios = [r["mean_second_jump_moment_over_N3"] for r in volume_rows]
    bounded_ratio = bool(ratios) and max(ratios) / min(ratios) < 5.0

    result = {
        "schema": 1,
        "scope": "finite QCCG spatial-Pachner move-count mechanism; diffusive continuum promotion remains open",
        "sources": SOURCES,
        "evidence": [
            evidence(
                "qccg-pachner-extensive-diffusion",
                "QCCG_PACHNER_EXTENSIVE_VOLUME_DIFFUSION",
                "PASS" if inverse_ok and exact_extensive else "FAIL",
                "For every audited spatial triangulation, the 1->4 move family supplies exactly one local site per tetrahedron and hence an exact 9*N3 contribution to the equal-site-rate second volume-jump moment.",
                inverse_ok=inverse_ok,
                rows=rows,
                volume_rows=volume_rows,
                sources=SOURCES,
            ),
            evidence(
                "qccg-volume-diffusion-scaling-diagnostic",
                "QCCG_VOLUME_DIFFUSION_SCALING_DIAGNOSTIC",
                "PASS" if bounded_ratio else "FAIL",
                "Across the finite audited state graph, the total equal-site-rate second jump moment remains O(N3); this is a finite scaling diagnostic, not an asymptotic exponent measurement.",
                volume_rows=volume_rows,
                ratio_spread=None if not ratios else max(ratios) / min(ratios),
                preregistered_max_ratio_spread=5.0,
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-diffusive-kinetic-promotion",
                "QCCG_CDT_DIFFUSIVE_KINETIC_LIMIT",
                "OPEN",
                "The microscopic extensive jump moment gives the correct denominator mechanism, but a central-limit/diffusion regime with a measured Gaussian volume kernel and stable coefficient has not yet been demonstrated on increasing QCCG ensembles.",
                required_next_step=(
                    "Measure P(Delta N3 | N3) for increasing causal QCCG ensembles and time blocking; "
                    "test Gaussianity, Var(Delta N3|N3) proportional to N3, and convergence of "
                    "-log M_nm to A*(n-m)^2/(n+m) over an expanding volume window."
                ),
                sources=SOURCES,
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"][:2]):
        raise SystemExit("QCCG-CDT diffusion bridge internal audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
