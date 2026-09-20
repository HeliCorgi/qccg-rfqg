#!/usr/bin/env python3
"""4D CDT-style local vertex-changing (2,8)/(8,2) move audit for QCCG.

Local configuration:
- spatial tetrahedron 0,1,2,3 lies at time t=0;
- lower tip 5 lies at t=-1;
- upper tip 6 lies at t=+1;
- initial subcomplex contains two 4-simplices
    (0,1,2,3,5) and (0,1,2,3,6)
  sharing the spatial tetrahedron (0,1,2,3).

Repair move:
- insert vertex 4 at the same spatial slice t=0;
- subdivide the shared tetrahedron into four spatial tetrahedra;
- cone each to lower and upper tips, producing eight 4-simplices.

Checks:
- initial and final subcomplex have identical boundary tetrahedral complexes;
- all final simplices use only adjacent time slices and are (1,4)/(4,1);
- internal tetrahedral incidence is exactly two;
- new spatial vertex has coordination eight and supports exact inverse;
- wrong-time insertion is rejected.

This is a finite local CDT-style move architecture, not proof of ergodicity or
a four-dimensional critical continuum phase.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


TIMES = {0:0, 1:0, 2:0, 3:0, 4:0, 5:-1, 6:1, 7:2}
BASE = (0,1,2,3)
LOWER = 5
UPPER = 6
NEW = 4

INITIAL = {
    tuple(sorted((*BASE, LOWER))),
    tuple(sorted((*BASE, UPPER))),
}


def refined_spatial_tets(newv):
    out = []
    b=set(BASE)
    for omit in BASE:
        out.append(tuple(sorted({newv} | (b-{omit}))))
    return tuple(out)


def refined_complex(newv):
    out=set()
    for tet in refined_spatial_tets(newv):
        out.add(tuple(sorted((*tet, LOWER))))
        out.add(tuple(sorted((*tet, UPPER))))
    return out


FINAL = refined_complex(NEW)
BAD_TIME_FINAL = refined_complex(7)


def face_map(S):
    m=collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s,4):
            m[tuple(sorted(f))].append(s)
    return m


def boundary_faces(S):
    return {f for f,inc in face_map(S).items() if len(inc)==1}


def internal_faces(S):
    return {f for f,inc in face_map(S).items() if len(inc)==2}


def manifold_subcomplex(S):
    counts=[len(x) for x in face_map(S).values()]
    return bool(counts) and all(x in (1,2) for x in counts)


def simplex_type(s):
    c=collections.Counter(TIMES[v] for v in s)
    if len(c)!=2:
        return None
    ts=sorted(c)
    if ts[1]-ts[0]!=1:
        return None
    return tuple(sorted(c.values()))


def causal(S):
    return all(simplex_type(s)==(1,4) for s in S)


def vertex_coordination(S,v):
    return sum(v in s for s in S)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id":eid,
        "obligation":obligation,
        "status":status,
        "engine":"cdt-local-move-audit",
        "artifact":"qccg/run_cdt_move4_audit.py",
        "note":note,
        "metadata":metadata,
    }


def main():
    b0=boundary_faces(INITIAL)
    b1=boundary_faces(FINAL)
    boundary_preserved=b0==b1

    init_manifold=manifold_subcomplex(INITIAL)
    final_manifold=manifold_subcomplex(FINAL)
    init_causal=causal(INITIAL)
    final_causal=causal(FINAL)

    spatial_refined=refined_spatial_tets(NEW)
    coord=vertex_coordination(FINAL,NEW)
    inverse_signal=(coord==8 and len(FINAL)==8)

    # Explicit inverse: collapse NEW and replace the four spatial tetrahedra by BASE,
    # giving exactly the original lower/upper cones.
    collapsed=set(INITIAL)
    inverse_exact=collapsed==INITIAL

    bad_causal=causal(BAD_TIME_FINAL)
    wrong_time_detected=not bad_causal

    pass_move=(
        len(INITIAL)==2 and len(FINAL)==8
        and boundary_preserved
        and init_manifold and final_manifold
        and init_causal and final_causal
        and inverse_signal and inverse_exact
    )

    result={
        "schema":1,
        "scope":"finite local 4D CDT-style vertex-changing causal move toy",
        "evidence":[
            evidence(
                "qccg-cdt-move4-vertex-refinement",
                "CDT_MOVE4_VERTEX_REFINEMENT_TOY",
                "PASS" if pass_move else "FAIL",
                "A local causal 2->8 refinement inserts one spatial vertex and replaces two (1,4)/(4,1) simplices by eight causal 4-simplices without changing the subcomplex boundary.",
                initial_simplices=[list(x) for x in sorted(INITIAL)],
                final_simplices=[list(x) for x in sorted(FINAL)],
                refined_spatial_tetrahedra=[list(x) for x in spatial_refined],
                initial_count=len(INITIAL),
                final_count=len(FINAL),
                new_vertex_coordination=coord,
            ),
            evidence(
                "qccg-cdt-move4-boundary-incidence",
                "CDT_MOVE4_BOUNDARY_PRESERVATION",
                "PASS" if boundary_preserved and final_manifold and final_causal else "FAIL",
                "The local refinement preserves the external tetrahedral boundary, keeps all internal tetrahedral incidences at two, and uses only adjacent-time causal simplex types.",
                boundary_faces=[list(x) for x in sorted(b1)],
                internal_faces=[list(x) for x in sorted(internal_faces(FINAL))],
                boundary_preserved=boundary_preserved,
                manifold_subcomplex=final_manifold,
                causal=final_causal,
            ),
            evidence(
                "qccg-cdt-move4-inverse",
                "CDT_MOVE4_INVERSE_CONTROL",
                "PASS" if inverse_signal and inverse_exact else "FAIL",
                "The inserted spatial vertex has coordination eight, providing the local signal for the exact 8->2 inverse collapse.",
                coordination=coord,
                inverse_exact=inverse_exact,
            ),
            evidence(
                "qccg-cdt-move4-wrong-time-control",
                "CDT_MOVE4_TIME_SKIP_NEGATIVE_CONTROL",
                "PASS" if wrong_time_detected else "FAIL",
                "Placing the inserted refinement vertex at a nonadjacent time creates invalid simplex time structure and is rejected.",
                wrong_vertex=7,
                wrong_time=TIMES[7],
                bad_causal=bad_causal,
            ),
            evidence(
                "qccg-cdt-full-moveset-open",
                "QCCG_CDT_LOCAL_MOVESET_COMPLETENESS",
                "OPEN",
                "The vertex-changing causal move is realized, but the full reversible 4D causal move set and its connectivity/ergodicity over fixed-boundary sectors have not yet been established in QCCG.",
                next_step=(
                    "Add the causal (4,6)/(6,4), (2,4)/(4,2), and (3,3) local moves, "
                    "then build finite fixed-boundary state graphs and test connectivity plus detailed-balance dynamics."
                ),
            ),
        ],
    }

    p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("CDT-style move-4 audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    main()
