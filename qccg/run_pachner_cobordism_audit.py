#!/usr/bin/env python3
"""Local 4-simplex cobordism audit for 3D Pachner transitions.

A d-dimensional Pachner move is the replacement of a subset of boundary
d-simplices of a (d+1)-simplex by the complementary subset.  For d=3:
- 1 <-> 4 is a 1/4 split of the five tetrahedral facets of a 4-simplex;
- 2 <-> 3 is a 2/3 split.

This audit verifies that both sides of each split have the same boundary
2-complex, so one 4-simplex gives a local four-dimensional cobordism between
the spatial triangulations.  It also checks inverse-complementarity.

This is a local combinatorial spacetime interpolation.  It does not yet prove
that arbitrary sequences of these blocks admit the strict global QCCG/CDT-like
foliation and causal incidence required of the continuum phase.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


VERTS = (0, 1, 2, 3, 4)
FACETS = tuple(
    tuple(v for v in VERTS if v != omit)
    for omit in VERTS
)


def boundary_triangles(tets):
    counts = collections.Counter()
    for tet in tets:
        for tri in itertools.combinations(tet, 3):
            counts[tuple(sorted(tri))] += 1
    return {tri for tri, n in counts.items() if n == 1}


def internal_triangles(tets):
    counts = collections.Counter()
    for tet in tets:
        for tri in itertools.combinations(tet, 3):
            counts[tuple(sorted(tri))] += 1
    return {tri for tri, n in counts.items() if n == 2}


def vertex_set(tets):
    return {v for tet in tets for v in tet}


def audit_split(lower_ids):
    lower = tuple(FACETS[i] for i in lower_ids)
    upper_ids = tuple(i for i in range(5) if i not in lower_ids)
    upper = tuple(FACETS[i] for i in upper_ids)

    bl = boundary_triangles(lower)
    bu = boundary_triangles(upper)
    same_boundary = bl == bu

    # Each side must be a 3-ball Pachner cluster: all triangle incidences <=2.
    def manifold_ball(tets):
        counts = collections.Counter()
        for tet in tets:
            for tri in itertools.combinations(tet, 3):
                counts[tuple(sorted(tri))] += 1
        return all(n in (1, 2) for n in counts.values())

    return {
        "lower_facet_ids": list(lower_ids),
        "upper_facet_ids": list(upper_ids),
        "lower_tetrahedra": [list(x) for x in lower],
        "upper_tetrahedra": [list(x) for x in upper],
        "lower_count": len(lower),
        "upper_count": len(upper),
        "same_boundary_2complex": same_boundary,
        "boundary_triangles": [list(x) for x in sorted(bl)],
        "lower_internal_triangles": [list(x) for x in sorted(internal_triangles(lower))],
        "upper_internal_triangles": [list(x) for x in sorted(internal_triangles(upper))],
        "lower_vertex_count": len(vertex_set(lower)),
        "upper_vertex_count": len(vertex_set(upper)),
        "lower_ball_incidence": manifold_ball(lower),
        "upper_ball_incidence": manifold_ball(upper),
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "pachner-4simplex-cobordism-audit",
        "artifact": "qccg/run_pachner_cobordism_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    one_four = audit_split((4,))      # facet 4 vs remaining four
    two_three = audit_split((3, 4))   # two adjacent facets vs remaining three

    pass14 = (
        one_four["lower_count"] == 1
        and one_four["upper_count"] == 4
        and one_four["same_boundary_2complex"]
        and one_four["lower_ball_incidence"]
        and one_four["upper_ball_incidence"]
    )
    pass23 = (
        two_three["lower_count"] == 2
        and two_three["upper_count"] == 3
        and two_three["same_boundary_2complex"]
        and two_three["lower_ball_incidence"]
        and two_three["upper_ball_incidence"]
    )

    # Inverse check is literally swapping lower/upper facet sets.
    inverse14 = audit_split(tuple(one_four["upper_facet_ids"]))
    inverse23 = audit_split(tuple(two_three["upper_facet_ids"]))
    inverse_pass = (
        inverse14["lower_count"] == 4
        and inverse14["upper_count"] == 1
        and inverse14["same_boundary_2complex"]
        and inverse23["lower_count"] == 3
        and inverse23["upper_count"] == 2
        and inverse23["same_boundary_2complex"]
    )

    result = {
        "schema": 1,
        "scope": "local 4-simplex combinatorial cobordism for spatial Pachner moves",
        "evidence": [
            evidence(
                "qccg-pachner-4simplex-cobordism",
                "PACHNER_4SIMPLEX_COBORDISM_TOY",
                "PASS" if pass14 and pass23 else "FAIL",
                "The 3D 1<->4 and 2<->3 Pachner transitions are realized as complementary boundary-facet decompositions of a single 4-simplex, with identical spatial 2-boundaries.",
                one_to_four=one_four,
                two_to_three=two_three,
            ),
            evidence(
                "qccg-pachner-cobordism-inverse",
                "PACHNER_COBORDISM_INVERSE_CONTROL",
                "PASS" if inverse_pass else "FAIL",
                "Swapping the complementary facet sets gives exact 4->1 and 3->2 inverse cobordisms with the same boundary 2-complex.",
                inverse_four_to_one=inverse14,
                inverse_three_to_two=inverse23,
            ),
            evidence(
                "qccg-global-causal-cobordism-open",
                "QCCG_GLOBAL_CAUSAL_COBORDISM_COMPATIBILITY",
                "OPEN",
                "Local 4-simplex Pachner cobordisms are established, but arbitrary compositions have not yet been shown to admit the strict global time labeling/causal foliation and manifold incidence required by QCCG.",
                next_step=(
                    "Compose several local Pachner cobordism blocks along the time-local slice-transfer paths, "
                    "assign global time labels, and audit all shared tetrahedral incidences plus absence of time-skipping simplices."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("Pachner cobordism audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
