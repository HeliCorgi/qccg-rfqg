#!/usr/bin/env python3
"""Embed strict local Pachner slabs into complete closed S^3 slice transitions.

Two constructive audits are performed.

1) Closed-slice 1->4:
   Start from the boundary of a 4-simplex (a triangulated S^3).  Triangulate
   every tetrahedron x interval by the same global staircase rule, then
   stellar-subdivide the upper boundary tetrahedron of one chosen prism.
   The resulting 4D complex must have boundary ONLY
      lower S^3  +  upper 1->4-refined S^3.

2) Closed-slice 2->3:
   Start from an S^3 obtained by one 1->4 refinement and relabel it so a valid
   2->3 pair is exactly 0123,0124.  Build the product slab for the whole S^3,
   remove the product subcomplex of that pair, and replace it by the explicit
   nine-simplex strict 2->3 slab.  Again, the only 4D boundary must be the
   complete lower and upper S^3 slices.

Thus the local slabs are shown to match the standard lateral prism
triangulation of their complements in explicit closed spatial universes.
This remains finite constructive evidence, not a theorem for arbitrary
triangulations/histories.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


def V(t,i): return (t,i)


def boundary_4simplex_s3():
    verts=set(range(5))
    return {tuple(sorted(verts-{omit})) for omit in verts}


def refine14(S,tet,newv):
    S2=set(S);S2.remove(tuple(sorted(tet)));T=set(tet)
    for omit in tet:
        S2.add(tuple(sorted({newv}|(T-{omit}))))
    return S2


def staircase_tet_prism(tet):
    v=sorted(tet); out=set()
    for j in range(4):
        s=[V(0,v[i]) for i in range(j+1)]
        s += [V(1,v[i]) for i in range(j,4)]
        out.add(tuple(sorted(s)))
    return out


def product_slab(S):
    return set().union(*(staircase_tet_prism(tet) for tet in S))


def face_map(S):
    m=collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s,4):
            m[tuple(sorted(f))].append(s)
    return m


def boundary(S):
    return {f for f,inc in face_map(S).items() if len(inc)==1}


def manifold(S):
    vals=[len(x) for x in face_map(S).values()]
    return bool(vals) and all(n in (1,2) for n in vals)


def causal(S):
    for s in S:
        cnt=collections.Counter(t for t,_ in s)
        if len(cnt)!=2:
            return False
        ts=sorted(cnt)
        if ts!=[0,1]:
            return False
        if tuple(sorted(cnt.values())) not in {(1,4),(2,3)}:
            return False
    return True


def dual_connected(S):
    sims=list(S);idx={s:i for i,s in enumerate(sims)}
    adj=[set() for _ in sims]
    for inc in face_map(S).values():
        if len(inc)==2:
            a,b=inc;ia,ib=idx[a],idx[b]
            adj[ia].add(ib);adj[ib].add(ia)
    seen=set();stack=[0] if sims else []
    while stack:
        i=stack.pop()
        if i in seen:continue
        seen.add(i);stack.extend(adj[i]-seen)
    return len(seen)==len(sims)


def spatial_boundary(S,t):
    return {
        tuple(sorted(v for _tt,v in f))
        for f in boundary(S)
        if {tt for tt,_v in f}=={t}
    }


def mixed_boundary(S):
    return {
        f for f in boundary(S)
        if len({tt for tt,_v in f})>1
    }


def closed14():
    lower=boundary_4simplex_s3()
    moving=(0,1,2,3)
    upper=refine14(lower,moving,5)

    slab=product_slab(lower)
    top=tuple(sorted(V(1,i) for i in moving))
    adjacent=[s for s in slab if set(top).issubset(s)]
    if len(adjacent)!=1:
        raise RuntimeError("1->4 closed slab: top facet not uniquely adjacent")
    old=adjacent[0]
    opp=next(iter(set(old)-set(top)))
    slab.remove(old)
    newv=V(1,5)
    for omit in top:
        slab.add(tuple(sorted({opp,newv}|(set(top)-{omit}))))

    ok=(
        manifold(slab) and causal(slab) and dual_connected(slab)
        and spatial_boundary(slab,0)==lower
        and spatial_boundary(slab,1)==upper
        and not mixed_boundary(slab)
    )
    return slab,lower,upper,ok


STRICT23={
    (0,1,2,3,8),
    (0,1,2,4,9),
    (0,1,2,7,8),
    (0,1,2,7,9),
    (0,1,6,8,9),
    (0,1,7,8,9),
    (0,5,6,8,9),
    (0,5,7,8,9),
    (1,6,7,8,9),
}


def map23():
    out=set()
    for s in STRICT23:
        out.add(tuple(sorted(V(0,v) if v<5 else V(1,v-5) for v in s)))
    return out


def closed23():
    # Closed S^3 with the valid pair 0123,0124; obtained from a boundary-S^3
    # by one 1->4 refinement and a gauge relabelling.
    lower={
        (0,1,2,3),(0,1,2,4),
        (0,1,3,5),(0,1,4,5),
        (0,2,3,5),(0,2,4,5),
        (1,2,3,5),(1,2,4,5),
    }
    pair={(0,1,2,3),(0,1,2,4)}
    replacement={(1,2,3,4),(0,2,3,4),(0,1,3,4)}
    upper=(lower-pair)|replacement

    slab=product_slab(lower)
    pair_product=set().union(*(staircase_tet_prism(t) for t in pair))
    # Every removed simplex must actually occur in the full product slab.
    if not pair_product.issubset(slab):
        raise RuntimeError("2->3 closed slab: pair product not contained")
    slab-=pair_product
    slab|=map23()

    ok=(
        manifold(slab) and causal(slab) and dual_connected(slab)
        and spatial_boundary(slab,0)==lower
        and spatial_boundary(slab,1)==upper
        and not mixed_boundary(slab)
    )
    return slab,lower,upper,ok


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"closed-slice-causal-slab-audit",
        "artifact":"qccg/run_closed_slice_slab_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    s14,l14,u14,ok14=closed14()
    s23,l23,u23,ok23=closed23()

    # Negative control: remove one local 2->3 replacement simplex.  A closed
    # slice slab must then acquire a spurious mixed-time boundary or mismatch.
    bad=set(s23);bad.remove(next(iter(map23())))
    bad_detected=(
        spatial_boundary(bad,0)!=l23
        or spatial_boundary(bad,1)!=u23
        or bool(mixed_boundary(bad))
        or not manifold(bad)
    )

    result={
        "schema":1,
        "scope":"explicit closed-S3 adjacent-slice embeddings of local 1->4 and 2->3 Pachner slabs",
        "evidence":[
            evidence(
                "qccg-closed-slice-14-slab",
                "CLOSED_SLICE_14_CAUSAL_SLAB_TOY",
                "PASS" if ok14 else "FAIL",
                "A complete closed S^3 slice-to-slice product slab remains a causal 4D manifold after an upper-boundary 1->4 refinement; its only boundary components are the full lower and refined upper S^3 slices.",
                n_4simplices=len(s14),
                lower_tetrahedra=len(l14),
                upper_tetrahedra=len(u14),
                mixed_boundary_count=len(mixed_boundary(s14)),
                manifold=manifold(s14),causal=causal(s14),dual_connected=dual_connected(s14),
            ),
            evidence(
                "qccg-closed-slice-23-slab",
                "CLOSED_SLICE_23_CAUSAL_SLAB_TOY",
                "PASS" if ok23 else "FAIL",
                "Replacing the product subcomplex of a valid 2->3 pair inside a closed S^3 product slab by the explicit nine-simplex strict slab yields a causal 4D manifold whose only boundaries are the complete lower and 2->3-transformed upper S^3 slices.",
                n_4simplices=len(s23),
                lower_tetrahedra=len(l23),
                upper_tetrahedra=len(u23),
                mixed_boundary_count=len(mixed_boundary(s23)),
                manifold=manifold(s23),causal=causal(s23),dual_connected=dual_connected(s23),
            ),
            evidence(
                "qccg-closed-slice-slab-removal-control",
                "CLOSED_SLICE_SLAB_COMPLETION_CONTROL",
                "PASS" if bad_detected else "FAIL",
                "Removing one required strict 2->3 replacement simplex creates a boundary/incidence defect in the completed closed-slice slab and is detected.",
                bad_mixed_boundary_count=len(mixed_boundary(bad)),
                bad_manifold=manifold(bad),
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("closed-slice causal slab audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
