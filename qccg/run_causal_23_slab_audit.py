#!/usr/bin/env python3
"""Strict adjacent-slice causal slab for a spatial 2<->3 Pachner move.

Five spatial vertices are copied to a lower slice t=0 and an upper slice t=1.
The lower spatial 3-ball consists of two tetrahedra sharing triangle (0,1,2):
    0123, 0124.
The upper 3-ball is the 2->3 Pachner replacement:
    0134, 0234, 1234.

The common spatial 2-sphere boundary is connected across time by the standard
triangle x interval staircase triangulation.  A finite exact incidence search
over allowed (4,1)/(3,2)/(2,3)/(1,4) 4-simplices found the nine-simplex slab
verified below.

The committed audit does not depend on the search solver: it verifies directly
that the explicit nine-simplex complex
- has exactly the prescribed lower/upper/lateral boundary;
- has incidence two on every internal tetrahedron;
- uses only adjacent time slices;
- is connected;
- reverses to a valid strict 3->2 slab under t=0 <-> t=1;
- fails if one required 4-simplex is removed.

This closes the finite strict-foliation existence problem for the local
2<->3 spatial Pachner transition.  Arbitrary global compositions remain a
separate gate.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


def L(i): return i
def U(i): return 5+i

TIMES={i:0 for i in range(5)}
TIMES.update({5+i:1 for i in range(5)})

OLD=((0,1,2,3),(0,1,2,4))
NEW=((0,1,3,4),(0,2,3,4),(1,2,3,4))

# Explicit solution, lower vertices 0..4 and upper copies 5..9.
SLAB={
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


def boundary_triangles(tets):
    c=collections.Counter()
    for tet in tets:
        for tri in itertools.combinations(tet,3):
            c[tuple(sorted(tri))]+=1
    return {tri for tri,n in c.items() if n==1}


def triangle_prism(tri):
    v=sorted(tri)
    out=set()
    for j in range(3):
        tet=[L(v[i]) for i in range(j+1)]
        tet += [U(v[i]) for i in range(j,3)]
        out.add(tuple(sorted(tet)))
    return out


COMMON_2SPHERE=boundary_triangles(OLD)
assert COMMON_2SPHERE==boundary_triangles(NEW)

LOWER_BOUNDARY={tuple(sorted(L(i) for i in tet)) for tet in OLD}
UPPER_BOUNDARY={tuple(sorted(U(i) for i in tet)) for tet in NEW}
LATERAL_BOUNDARY=set().union(*(triangle_prism(tri) for tri in COMMON_2SPHERE))
TARGET_BOUNDARY=LOWER_BOUNDARY|UPPER_BOUNDARY|LATERAL_BOUNDARY


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
    vals=[len(inc) for inc in face_map(S).values()]
    return bool(vals) and all(n in (1,2) for n in vals)


def causal_type(s):
    c=collections.Counter(TIMES[v] for v in s)
    if len(c)!=2:
        return None
    ts=sorted(c)
    if ts!=[0,1]:
        return None
    return tuple(sorted(c.values()))


def causal(S):
    return all(causal_type(s) in {(1,4),(2,3)} for s in S)


def dual_connected(S):
    sims=list(S); idx={s:i for i,s in enumerate(sims)}
    adj=[set() for _ in sims]
    for inc in face_map(S).values():
        if len(inc)==2:
            a,b=inc
            ia,ib=idx[a],idx[b]
            adj[ia].add(ib);adj[ib].add(ia)
    seen=set(); stack=[0] if sims else []
    while stack:
        i=stack.pop()
        if i in seen: continue
        seen.add(i); stack.extend(adj[i]-seen)
    return len(seen)==len(sims)


def swap_time_vertex(v):
    return v+5 if v<5 else v-5


def swap_time_complex(S):
    return {tuple(sorted(swap_time_vertex(v) for v in s)) for s in S}


def swapped_target():
    return {
        tuple(sorted(swap_time_vertex(v) for v in f))
        for f in TARGET_BOUNDARY
    }


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"strict-causal-23-slab-audit",
        "artifact":"qccg/run_causal_23_slab_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    fmap=face_map(SLAB)
    bnd=boundary(SLAB)
    ints=internal(SLAB)

    boundary_exact=bnd==TARGET_BOUNDARY
    incidence_ok=manifold(SLAB)
    causal_ok=causal(SLAB)
    connected=dual_connected(SLAB)

    # Partition observed boundary for explicit provenance.
    lower_obs={f for f in bnd if all(v<5 for v in f)}
    upper_obs={f for f in bnd if all(v>=5 for v in f)}
    lateral_obs=bnd-lower_obs-upper_obs

    # Reverse the entire slab.  It must now run NEW -> OLD with the same
    # manifold/causal properties and the time-swapped boundary.
    rev=swap_time_complex(SLAB)
    reverse_ok=(
        manifold(rev)
        and causal(rev)
        and dual_connected(rev)
        and boundary(rev)==swapped_target()
    )

    # Negative control: removing any essential simplex creates an incorrect
    # boundary/internal incidence pattern.  Check every single-simplex deletion.
    removal_rows=[]
    all_removals_detected=True
    for s in sorted(SLAB):
        bad=set(SLAB);bad.remove(s)
        wrong=(boundary(bad)!=TARGET_BOUNDARY) or (not manifold(bad))
        all_removals_detected=all_removals_detected and wrong
        removal_rows.append({
            "removed":list(s),
            "detected":wrong,
            "boundary_matches_target":boundary(bad)==TARGET_BOUNDARY,
            "manifold":manifold(bad),
        })

    passed=boundary_exact and incidence_ok and causal_ok and connected
    result={
        "schema":1,
        "scope":"explicit strict adjacent-slice 4D slab for spatial 2<->3 Pachner transition",
        "evidence":[
            evidence(
                "qccg-causal-23-slab",
                "CAUSAL_23_SPACETIME_SLAB_REALIZATION",
                "PASS" if passed else "FAIL",
                "An explicit nine-simplex adjacent-slice 4D complex realizes the spatial 2->3 Pachner transition with exactly the prescribed lower, upper and lateral boundary and incidence-two internal tetrahedra.",
                slab_4simplices=[list(s) for s in sorted(SLAB)],
                n_4simplices=len(SLAB),
                causal_types={str(t):sum(causal_type(s)==t for s in SLAB) for t in ((1,4),(2,3))},
                lower_boundary=[list(f) for f in sorted(lower_obs)],
                upper_boundary=[list(f) for f in sorted(upper_obs)],
                lateral_boundary_count=len(lateral_obs),
                target_lateral_count=len(LATERAL_BOUNDARY),
                internal_tetrahedra_count=len(ints),
                boundary_exact=boundary_exact,
                manifold=incidence_ok,
                dual_connected=connected,
            ),
            evidence(
                "qccg-causal-23-slab-inverse",
                "CAUSAL_23_SLAB_INVERSE_CONTROL",
                "PASS" if reverse_ok else "FAIL",
                "Swapping the two time slices turns the same nine-simplex complex into a valid strict 3->2 slab with the time-reversed prescribed boundary.",
                reverse_boundary_exact=boundary(rev)==swapped_target(),
                reverse_manifold=manifold(rev),
                reverse_causal=causal(rev),
                reverse_connected=dual_connected(rev),
            ),
            evidence(
                "qccg-causal-23-slab-removal-control",
                "CAUSAL_23_SLAB_COMPLETENESS_CONTROL",
                "PASS" if all_removals_detected else "FAIL",
                "Deleting any one of the nine required 4-simplices is detected through boundary or manifold-incidence failure.",
                rows=removal_rows,
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("strict causal 2<->3 slab audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
