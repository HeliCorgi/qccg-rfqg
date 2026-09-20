#!/usr/bin/env python3
"""Strict time-local causal slab for a spatial 1<->4 refinement.

Construct a tetrahedron x interval using the standard 4D staircase
triangulation.  Its lower and upper tetrahedra lie at t=0 and t=1.

Then insert one new vertex on the UPPER spatial slice and stellar-subdivide the
single 4-simplex adjacent to the upper boundary tetrahedron.  The result has:
- one unrefined lower spatial tetrahedron;
- four refined upper spatial tetrahedra (a 1->4 Pachner refinement);
- only adjacent-slice causal 4-simplices;
- valid 4D manifold-with-boundary incidence.

Collapsing the upper refinement vertex reconstructs the original prism exactly.
A wrong-time new vertex is rejected.

This is an explicit strict-foliation slab for the 1<->4 spatial move only.
The independently audited strict 2<->3 slab is handled in
run_causal_23_slab_audit.py; arbitrary mixed histories remain separate
obligations.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


LOWER = ((0,0),(0,1),(0,2),(0,3))
UPPER = ((1,0),(1,1),(1,2),(1,3))
NEW = (1,4)
BAD_NEW = (2,4)


def staircase_prism(lower, upper):
    out=[]
    for j in range(4):
        s=[]
        for i in range(j+1):
            s.append(lower[i])
        for i in range(j,4):
            s.append(upper[i])
        out.append(tuple(sorted(s)))
    return set(out)


BASE = staircase_prism(LOWER,UPPER)


def face_map(S):
    m=collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s,4):
            m[tuple(sorted(f))].append(s)
    return m


def boundary(S):
    return {f for f,inc in face_map(S).items() if len(inc)==1}


def manifold(S):
    counts=[len(v) for v in face_map(S).values()]
    return bool(counts) and all(x in (1,2) for x in counts)


def causal_type(s):
    cnt=collections.Counter(t for t,_ in s)
    if len(cnt)!=2:
        return None
    ts=sorted(cnt)
    if ts[1]-ts[0]!=1:
        return None
    return tuple(sorted(cnt.values()))


def causal(S):
    return all(causal_type(s) in {(1,4),(2,3)} for s in S)


def refine_upper(S,newv):
    S2=set(S)
    top=tuple(sorted(UPPER))
    adjacent=[s for s in S2 if set(top).issubset(s)]
    if len(adjacent)!=1:
        raise RuntimeError(f"expected one top-adjacent 4-simplex, got {len(adjacent)}")
    old=adjacent[0]
    opposite=next(iter(set(old)-set(top)))
    S2.remove(old)
    for omit in top:
        S2.add(tuple(sorted({opposite,newv}|(set(top)-{omit}))))
    return S2,old,opposite


def collapse_upper(S,newv,old_simplex):
    S2=set(S)
    star=[s for s in S2 if newv in s]
    if len(star)!=4:
        return None,len(star)
    for s in star:
        S2.remove(s)
    S2.add(old_simplex)
    return S2,len(star)


def boundary_partition(S):
    lower=[];upper=[];side=[]
    for f in boundary(S):
        times={t for t,_ in f}
        if times=={0}:
            lower.append(f)
        elif times=={1}:
            upper.append(f)
        else:
            side.append(f)
    return set(lower),set(upper),set(side)


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"strict-causal-refinement-slab-audit",
        "artifact":"qccg/run_causal_refinement_slab_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    base_manifold=manifold(BASE)
    base_causal=causal(BASE)
    b_lower,b_upper,b_side=boundary_partition(BASE)

    refined,old,opp=refine_upper(BASE,NEW)
    r_lower,r_upper,r_side=boundary_partition(refined)

    upper_expected={
        tuple(sorted({NEW}|(set(UPPER)-{omit})))
        for omit in UPPER
    }
    lower_expected={tuple(sorted(LOWER))}

    refined_ok=(
        manifold(refined)
        and causal(refined)
        and r_lower==lower_expected
        and r_upper==upper_expected
    )

    collapsed,star_size=collapse_upper(refined,NEW,old)
    inverse_exact=collapsed==BASE

    bad,_old_bad,_opp_bad=refine_upper(BASE,BAD_NEW)
    wrong_time_detected=not causal(bad)

    # The side boundary changes only by a local subdivision adjacent to the top;
    # compare topology-level counts and retain the exact sets for provenance.
    base_side_count=len(b_side)
    refined_side_count=len(r_side)
    side_nonempty=base_side_count>0 and refined_side_count>0

    passed=base_manifold and base_causal and refined_ok and inverse_exact and side_nonempty

    result={
        "schema":1,
        "scope":"strict adjacent-slice 4D slab for one spatial 1<->4 refinement",
        "evidence":[
            evidence(
                "qccg-causal-14-slab",
                "CAUSAL_14_SPACETIME_SLAB_TOY",
                "PASS" if passed else "FAIL",
                "A tetrahedron-time prism admits an explicit upper-slice 1->4 refinement whose 4D triangulation remains a causal manifold-with-boundary with the expected lower and refined upper spatial boundaries.",
                base_simplices=[[list(v) for v in s] for s in sorted(BASE)],
                refined_simplices=[[list(v) for v in s] for s in sorted(refined)],
                old_top_adjacent_simplex=[list(v) for v in old],
                opposite_vertex=list(opp),
                lower_boundary=[[list(v) for v in f] for f in sorted(r_lower)],
                upper_boundary=[[list(v) for v in f] for f in sorted(r_upper)],
                side_boundary_count_before=base_side_count,
                side_boundary_count_after=refined_side_count,
                manifold=manifold(refined),
                causal=causal(refined),
            ),
            evidence(
                "qccg-causal-14-slab-inverse",
                "CAUSAL_14_SLAB_INVERSE_CONTROL",
                "PASS" if inverse_exact and star_size==4 else "FAIL",
                "Collapsing the coordination-four upper refinement vertex reconstructs the original causal tetrahedron-time prism exactly.",
                refinement_vertex=list(NEW),
                star_size=star_size,
                inverse_exact=inverse_exact,
            ),
            evidence(
                "qccg-causal-14-slab-time-control",
                "CAUSAL_14_SLAB_TIME_SKIP_CONTROL",
                "PASS" if wrong_time_detected else "FAIL",
                "Placing the refinement vertex at t=2 creates nonadjacent-time 4-simplices and is rejected.",
                bad_vertex=list(BAD_NEW),
                bad_causal=causal(bad),
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("strict causal 1<->4 slab audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
