#!/usr/bin/env python3
"""4D CDT-style local connectivity-changing (4,6)/(6,4) move audit.

Implements the standard causal move on two spatial tetrahedra 2345 and 3456
sharing triangle 345 at time t=0, coned to lower tip 1 (t=-1) and upper tip 7
(t=+1).

Initial: four 4-simplices.
Final: triangle 345 is replaced by the spatial dual edge 26, giving three
spatial tetrahedra 2346,2356,2456, each coned to both tips => six 4-simplices.

Checks boundary preservation, causal simplex types, manifold incidence, exact
inverse signal, and a wrong-time negative control.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


TIMES={1:-1,2:0,3:0,4:0,5:0,6:0,7:1,8:2}

INITIAL={
    tuple(sorted((1,2,3,4,5))),
    tuple(sorted((2,3,4,5,7))),
    tuple(sorted((1,3,4,5,6))),
    tuple(sorted((3,4,5,6,7))),
}
FINAL={
    tuple(sorted((1,2,3,4,6))),
    tuple(sorted((2,3,4,6,7))),
    tuple(sorted((1,2,3,5,6))),
    tuple(sorted((2,3,5,6,7))),
    tuple(sorted((1,2,4,5,6))),
    tuple(sorted((2,4,5,6,7))),
}

# Wrong-time variant: replace spatial vertex 6 by vertex 8 at t=2.
BAD_FINAL={tuple(sorted(8 if v==6 else v for v in s)) for s in FINAL}


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


def causal_type(s):
    c=collections.Counter(TIMES[v] for v in s)
    if len(c)!=2:
        return None
    ts=sorted(c)
    if ts[1]-ts[0]!=1:
        return None
    return tuple(sorted(c.values()))


def causal(S):
    return all(causal_type(s) in {(1,4),(2,3)} for s in S)


def edge_coord(S,e):
    E=set(e)
    return sum(E.issubset(s) for s in S)


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"cdt-local-connectivity-move-audit",
        "artifact":"qccg/run_cdt_move46_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    boundary_preserved=boundary(INITIAL)==boundary(FINAL)
    m0,m1=manifold(INITIAL),manifold(FINAL)
    c0,c1=causal(INITIAL),causal(FINAL)
    new_edge=(2,6)
    coord=edge_coord(FINAL,new_edge)
    inverse_signal=coord==6
    wrong_time_detected=not causal(BAD_FINAL)

    passed=(
        len(INITIAL)==4 and len(FINAL)==6
        and boundary_preserved and m0 and m1 and c0 and c1
        and inverse_signal
    )

    result={
        "schema":1,
        "scope":"finite local 4D CDT-style connectivity-changing causal move toy",
        "evidence":[
            evidence(
                "qccg-cdt-move46-connectivity",
                "CDT_MOVE46_CONNECTIVITY_TOY",
                "PASS" if passed else "FAIL",
                "A local causal 4->6 move replaces a shared spatial triangle by its dual spatial edge while preserving the external 4D subcomplex boundary.",
                initial_simplices=[list(x) for x in sorted(INITIAL)],
                final_simplices=[list(x) for x in sorted(FINAL)],
                boundary_preserved=boundary_preserved,
                initial_causal=c0,final_causal=c1,
            ),
            evidence(
                "qccg-cdt-move46-incidence",
                "CDT_MOVE46_BOUNDARY_PRESERVATION",
                "PASS" if boundary_preserved and m1 and c1 else "FAIL",
                "The connectivity-changing move preserves manifold incidence and adjacent-time causal simplex structure.",
                internal_tetrahedra=[list(x) for x in sorted(internal(FINAL))],
                manifold=m1,causal=c1,
            ),
            evidence(
                "qccg-cdt-move46-inverse",
                "CDT_MOVE46_INVERSE_CONTROL",
                "PASS" if inverse_signal else "FAIL",
                "The new spatial edge has coordination six, providing the local signal for the exact 6->4 inverse connectivity move.",
                edge=list(new_edge),coordination=coord,
            ),
            evidence(
                "qccg-cdt-move46-wrong-time-control",
                "CDT_MOVE46_TIME_SKIP_NEGATIVE_CONTROL",
                "PASS" if wrong_time_detected else "FAIL",
                "Moving one spatial vertex outside the adjacent slice structure is detected as an invalid causal configuration.",
                bad_causal=causal(BAD_FINAL),
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("CDT-style 4->6 move audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
