#!/usr/bin/env python3
"""Strict composition toy: spatial 2->3 followed by 3->2.

Use the explicit nine-simplex strict causal 2->3 slab on t=0->1, and glue its
time-reversed slab on t=1->2 along the SAME intermediate three-tetrahedron
spatial triangulation.

The union is audited as one 4D simplicial manifold-with-boundary:
- every 4-simplex spans adjacent times only;
- all tetrahedral incidences are 1 (external boundary) or 2 (internal);
- each intermediate t=1 spatial tetrahedron has incidence exactly two;
- the dual 4-simplex graph is connected;
- the only purely spatial external boundaries are the old 2-tetrahedron
  triangulations at t=0 and t=2.

A bad control gives the second slab a distinct copy of the t=1 vertices.  The
middle triangulations then fail to glue and the union becomes disconnected /
retains spurious intermediate boundary.

This is a finite strict-foliation composition proof for one round trip, not an
arbitrary mixed-history theorem.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path


OLD=((0,1,2,3),(0,1,2,4))
NEW=((0,1,3,4),(0,2,3,4),(1,2,3,4))

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


def swap(v):
    return v+5 if v<5 else v-5


REV={tuple(sorted(swap(v) for v in s)) for s in SLAB}


def map_slab(S,t0,t1,lower_offset=0):
    out=set()
    for s in S:
        mapped=[]
        for v in s:
            if v<5:
                mapped.append((t0,lower_offset+v))
            else:
                mapped.append((t1,v-5))
        out.add(tuple(sorted(mapped)))
    return out


FIRST=map_slab(SLAB,0,1)
SECOND=map_slab(REV,1,2)
GOOD=FIRST|SECOND

# Bad gluing: second lower vertices have distinct spatial IDs, so no middle
# tetrahedron is identified with the first slab.
SECOND_BAD=map_slab(REV,1,2,lower_offset=100)
BAD=FIRST|SECOND_BAD


def face_map(S):
    m=collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s,4):
            m[tuple(sorted(f))].append(s)
    return m


def manifold(S):
    vals=[len(x) for x in face_map(S).values()]
    return bool(vals) and all(n in (1,2) for n in vals)


def causal(S):
    for s in S:
        cnt=collections.Counter(t for t,_ in s)
        if len(cnt)!=2:
            return False
        ts=sorted(cnt)
        if ts[1]-ts[0]!=1:
            return False
        if tuple(sorted(cnt.values())) not in {(1,4),(2,3)}:
            return False
    return True


def dual_connected(S):
    sims=list(S);idx={s:i for i,s in enumerate(sims)}
    adj=[set() for _ in sims]
    for inc in face_map(S).values():
        if len(inc)==2:
            a,b=inc
            ia,ib=idx[a],idx[b]
            adj[ia].add(ib);adj[ib].add(ia)
    seen=set();stack=[0] if sims else []
    while stack:
        i=stack.pop()
        if i in seen: continue
        seen.add(i);stack.extend(adj[i]-seen)
    return len(seen)==len(sims)


def spatial_faces_on_time(S,t):
    return {
        f for f,inc in face_map(S).items()
        if len(inc)==1 and {x[0] for x in f}=={t}
    }


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"strict-slab-composition-audit",
        "artifact":"qccg/run_strict_slab_composition_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    fm=face_map(GOOD)
    middle={
        tuple(sorted((1,i) for i in tet))
        for tet in NEW
    }
    middle_inc={str(f):len(fm.get(f,[])) for f in sorted(middle)}
    middle_internal=all(len(fm.get(f,[]))==2 for f in middle)

    expected_t0={tuple(sorted((0,i) for i in tet)) for tet in OLD}
    expected_t2={tuple(sorted((2,i) for i in tet)) for tet in OLD}
    t0=spatial_faces_on_time(GOOD,0)
    t1=spatial_faces_on_time(GOOD,1)
    t2=spatial_faces_on_time(GOOD,2)

    good_pass=(
        manifold(GOOD)
        and causal(GOOD)
        and dual_connected(GOOD)
        and middle_internal
        and t0==expected_t0
        and not t1
        and t2==expected_t2
    )

    bad_fm=face_map(BAD)
    bad_middle_inc=[len(bad_fm.get(f,[])) for f in middle]
    bad_detected=(
        not dual_connected(BAD)
        or spatial_faces_on_time(BAD,1)
        or any(n!=2 for n in bad_middle_inc)
    )

    result={
        "schema":1,
        "scope":"finite strict three-slice composition of explicit 2->3 and 3->2 causal slabs",
        "evidence":[
            evidence(
                "qccg-strict-23-roundtrip-composition",
                "STRICT_23_ROUNDTRIP_COMPOSITION_TOY",
                "PASS" if good_pass else "FAIL",
                "The explicit strict 2->3 slab and its time reverse glue on the same intermediate triangulation to form one connected three-slice causal 4D manifold-with-boundary; the t=1 spatial tetrahedra become internal.",
                n_4simplices=len(GOOD),
                middle_tetrahedron_incidences=middle_inc,
                t0_spatial_boundary=[[list(v) for v in f] for f in sorted(t0)],
                t1_spatial_boundary=[[list(v) for v in f] for f in sorted(t1)],
                t2_spatial_boundary=[[list(v) for v in f] for f in sorted(t2)],
                manifold=manifold(GOOD),
                causal=causal(GOOD),
                dual_connected=dual_connected(GOOD),
            ),
            evidence(
                "qccg-strict-slab-misglue-control",
                "STRICT_SLAB_GLUE_NEGATIVE_CONTROL",
                "PASS" if bad_detected else "FAIL",
                "Using a distinct copy of the intermediate slice vertices prevents the two strict slabs from composing and is detected through disconnection or spurious t=1 boundary/incidence.",
                bad_dual_connected=dual_connected(BAD),
                bad_t1_spatial_boundary_count=len(spatial_faces_on_time(BAD,1)),
                bad_middle_incidences=bad_middle_inc,
            ),
            evidence(
                "qccg-arbitrary-strict-composition-open",
                "QCCG_STRICT_FOLIATION_COMPOSITION",
                "OPEN",
                "One nontrivial strict 2->3->2 round trip composes exactly, but arbitrary mixed 1<->4 and 2<->3 histories have not yet been proven to admit one globally consistent strict foliation without additional Pachner relations/coherence conditions.",
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("strict slab composition audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
