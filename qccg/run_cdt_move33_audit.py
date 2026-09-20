#!/usr/bin/env python3
"""4D CDT-style self-dual (3,3) causal move audit.

Implements one standard Lorentzian variant
  12[456] + 13[456] + 23[456]
    <->
  [123]45 + [123]46 + [123]56

Time assignment:
- vertices 1,2,4,5 at t=0
- vertices 3,6 at t=1
so both swapped triangles 123 and 456 are time-like with their space-like
edges in the same spatial slice.

Checks boundary preservation, causal simplex types, manifold incidence,
self-duality and a wrong-time negative control.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


TIMES={1:0,2:0,3:1,4:0,5:0,6:1,7:2}

INITIAL={
    tuple(sorted((1,2,4,5,6))),
    tuple(sorted((1,3,4,5,6))),
    tuple(sorted((2,3,4,5,6))),
}
FINAL={
    tuple(sorted((1,2,3,4,5))),
    tuple(sorted((1,2,3,4,6))),
    tuple(sorted((1,2,3,5,6))),
}
BAD_FINAL={tuple(sorted(7 if v==6 else v for v in s)) for s in FINAL}


def face_map(S):
    m=collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s,4):
            m[tuple(sorted(f))].append(s)
    return m


def boundary(S):
    return {f for f,inc in face_map(S).items() if len(inc)==1}


def internal(S):
    return {f for f,inc in face_map(S).items() if len(inc)==2}


def manifold(S):
    return all(len(inc) in (1,2) for inc in face_map(S).values())


def stype(s):
    c=collections.Counter(TIMES[v] for v in s)
    if len(c)!=2: return None
    ts=sorted(c)
    if ts[1]-ts[0]!=1: return None
    return tuple(sorted(c.values()))


def causal(S):
    return all(stype(s) in {(1,4),(2,3)} for s in S)


def triangle_coord(S,tri):
    T=set(tri)
    return sum(T.issubset(s) for s in S)


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"cdt-selfdual-move-audit",
        "artifact":"qccg/run_cdt_move33_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    bpres=boundary(INITIAL)==boundary(FINAL)
    mpass=manifold(INITIAL) and manifold(FINAL)
    cpass=causal(INITIAL) and causal(FINAL)
    tri_initial=(4,5,6)
    tri_final=(1,2,3)
    coord_i=triangle_coord(INITIAL,tri_initial)
    coord_f=triangle_coord(FINAL,tri_final)
    selfdual=(coord_i==3 and coord_f==3 and len(INITIAL)==len(FINAL)==3)
    bad_detected=not causal(BAD_FINAL)

    passed=bpres and mpass and cpass and selfdual

    result={
        "schema":1,
        "scope":"finite local 4D CDT-style self-dual causal move toy",
        "evidence":[
            evidence(
                "qccg-cdt-move33-selfdual",
                "CDT_MOVE33_SELFDUAL_TOY",
                "PASS" if passed else "FAIL",
                "A local causal 3->3 move replaces one timelike coordination-three triangle by its dual timelike triangle while preserving the external subcomplex boundary.",
                initial_simplices=[list(x) for x in sorted(INITIAL)],
                final_simplices=[list(x) for x in sorted(FINAL)],
                initial_triangle=list(tri_initial),
                final_triangle=list(tri_final),
                initial_coordination=coord_i,
                final_coordination=coord_f,
            ),
            evidence(
                "qccg-cdt-move33-boundary",
                "CDT_MOVE33_BOUNDARY_PRESERVATION",
                "PASS" if bpres and mpass and cpass else "FAIL",
                "The self-dual connectivity move preserves tetrahedral boundary, manifold incidence and adjacent-time causal simplex types.",
                boundary_preserved=bpres,
                internal_initial=[list(x) for x in sorted(internal(INITIAL))],
                internal_final=[list(x) for x in sorted(internal(FINAL))],
            ),
            evidence(
                "qccg-cdt-move33-wrong-time-control",
                "CDT_MOVE33_TIME_SKIP_NEGATIVE_CONTROL",
                "PASS" if bad_detected else "FAIL",
                "Moving one vertex outside the adjacent time slices invalidates the causal self-dual move.",
                bad_causal=causal(BAD_FINAL),
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("CDT-style 3->3 move audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
