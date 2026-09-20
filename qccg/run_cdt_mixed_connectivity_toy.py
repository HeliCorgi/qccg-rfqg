#!/usr/bin/env python3
"""Finite mixed-CDT-move fixed-boundary connectivity audit.

Three independently audited local 4D causal move patches are used:
- vertex-changing 2<->8;
- connectivity-changing 4<->6;
- self-dual 3<->3.

The patches are glued successively along boundary tetrahedra with matching
integer time patterns.  The script searches for a gluing for which all 2^3
local-move configurations are:
- connected 4D simplicial manifolds-with-boundary;
- causal (all 4-simplices use adjacent time slices);
- identical on the external tetrahedral boundary.

The resulting 8-state graph is a genuine finite fixed-boundary sector with
three move types.  Connectivity of this finite sector is checked, together
with a negative control where one move family is removed and the state graph
splits.

This is finite-sector connectivity evidence only, not scalable ergodicity of
the full CDT/QCCG triangulation space.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import sympy as sp


# Patch A: 2 <-> 8 vertex-changing move.
A_TIMES = {0:0,1:0,2:0,3:0,4:0,5:-1,6:1}
A0 = {
    tuple(sorted((0,1,2,3,5))),
    tuple(sorted((0,1,2,3,6))),
}
def a_final():
    out=set()
    base={0,1,2,3}
    for omit in (0,1,2,3):
        tet=tuple(sorted({4}|(base-{omit})))
        out.add(tuple(sorted((*tet,5))))
        out.add(tuple(sorted((*tet,6))))
    return out
A1=a_final()

# Patch B: 4 <-> 6 connectivity move.
B_TIMES={1:-1,2:0,3:0,4:0,5:0,6:0,7:1}
B0={
    tuple(sorted((1,2,3,4,5))),
    tuple(sorted((2,3,4,5,7))),
    tuple(sorted((1,3,4,5,6))),
    tuple(sorted((3,4,5,6,7))),
}
B1={
    tuple(sorted((1,2,3,4,6))),
    tuple(sorted((2,3,4,6,7))),
    tuple(sorted((1,2,3,5,6))),
    tuple(sorted((2,3,5,6,7))),
    tuple(sorted((1,2,4,5,6))),
    tuple(sorted((2,4,5,6,7))),
}

# Patch C: 3 <-> 3 self-dual move.
C_TIMES={1:0,2:0,3:1,4:0,5:0,6:1}
C0={
    tuple(sorted((1,2,4,5,6))),
    tuple(sorted((1,3,4,5,6))),
    tuple(sorted((2,3,4,5,6))),
}
C1={
    tuple(sorted((1,2,3,4,5))),
    tuple(sorted((1,2,3,4,6))),
    tuple(sorted((1,2,3,5,6))),
}


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


def dual_connected(S):
    sims=list(S)
    idx={s:i for i,s in enumerate(sims)}
    adj=[set() for _ in sims]
    for inc in face_map(S).values():
        if len(inc)==2:
            a,b=inc
            i,j=idx[a],idx[b]
            adj[i].add(j);adj[j].add(i)
    seen=set();stack=[0] if sims else []
    while stack:
        i=stack.pop()
        if i in seen: continue
        seen.add(i);stack.extend(adj[i]-seen)
    return len(seen)==len(sims)


def simplex_causal(s,times):
    vals=collections.Counter(times[v] for v in s)
    if len(vals)!=2:
        return False
    ts=sorted(vals)
    return ts[1]-ts[0]==1 and tuple(sorted(vals.values())) in {(1,4),(2,3)}


def causal(S,times):
    return all(simplex_causal(s,times) for s in S)


def face_time_pattern(face,times):
    return tuple(sorted(times[v] for v in face))


def map_patch(states,times, face_local, face_global, global_times, shift, next_label):
    """Map one patch into global labels, identifying one boundary tetrahedron."""
    local_by_time=collections.defaultdict(list)
    global_by_time=collections.defaultdict(list)
    for v in face_local:
        local_by_time[times[v]+shift].append(v)
    for v in face_global:
        global_by_time[global_times[v]].append(v)
    if set(local_by_time)!=set(global_by_time):
        return None
    if any(len(local_by_time[t])!=len(global_by_time[t]) for t in local_by_time):
        return None

    classes=sorted(local_by_time)
    perm_lists=[]
    for t in classes:
        loc=sorted(local_by_time[t])
        glob=sorted(global_by_time[t])
        perm_lists.append((loc,list(itertools.permutations(glob))))

    all_local=sorted({v for S in states for s in S for v in s})
    for choices in itertools.product(*[x[1] for x in perm_lists]):
        mp={}
        for (loc,_),perm in zip(perm_lists,choices):
            for lv,gv in zip(loc,perm):
                mp[lv]=gv
        fresh=next_label
        times2=dict(global_times)
        for lv in all_local:
            if lv in mp: continue
            mp[lv]=fresh
            times2[fresh]=times[lv]+shift
            fresh+=1
        mapped=[]
        for S in states:
            mapped.append({
                tuple(sorted(mp[v] for v in s))
                for s in S
            })
        yield mapped,times2,fresh,mp


def audit_states(states,times):
    bs=[boundary(S) for S in states]
    same_boundary=all(b==bs[0] for b in bs[1:])
    rows=[]
    for S in states:
        rows.append({
            "n_4simplices":len(S),
            "manifold":manifold(S),
            "causal":causal(S,times),
            "dual_connected":dual_connected(S),
        })
    ok=same_boundary and all(r["manifold"] and r["causal"] and r["dual_connected"] for r in rows)
    return ok,bs[0],rows


def glue_second(base_states,base_times,patch_states,patch_times,next_label):
    common_boundary=set.intersection(*[boundary(S) for S in base_states])
    patch_boundary=set.intersection(*[boundary(S) for S in patch_states])
    for fg in sorted(common_boundary):
        pg=face_time_pattern(fg,base_times)
        for fl in sorted(patch_boundary):
            for shift in range(-2,3):
                if tuple(t+shift for t in face_time_pattern(fl,patch_times))!=pg:
                    continue
                for mapped,times2,next2,mp in map_patch(
                    patch_states,patch_times,fl,fg,base_times,shift,next_label
                ):
                    combos=[]
                    for A in base_states:
                        for B in mapped:
                            combos.append(set(A)|set(B))
                    ok,bnd,rows=audit_states(combos,times2)
                    if ok:
                        return combos,times2,next2,{
                            "global_face":list(fg),
                            "local_face":list(fl),
                            "shift":shift,
                            "mapping":{str(k):v for k,v in mp.items()},
                            "rows":rows,
                        }
    return None


def graph_connected(n,edges):
    adj=[set() for _ in range(n)]
    for a,b,_typ in edges:
        adj[a].add(b);adj[b].add(a)
    seen=set();stack=[0]
    while stack:
        v=stack.pop()
        if v in seen: continue
        seen.add(v);stack.extend(adj[v]-seen)
    return len(seen)==n


def laplacian(n,edges):
    L=sp.zeros(n)
    for a,b,_ in edges:
        L[a,a]+=1;L[b,b]+=1;L[a,b]-=1;L[b,a]-=1
    return L


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"mixed-cdt-fixed-boundary-connectivity",
        "artifact":"qccg/run_cdt_mixed_connectivity_toy.py",
        "note":note,"metadata":metadata,
    }


def main():
    # Patch A is the seed connected 4-ball pair.
    base_states=[A0,A1]
    base_times=dict(A_TIMES)
    ok0,b0,rows0=audit_states(base_states,base_times)
    next_label=max(base_times)+1

    gB=glue_second(base_states,base_times,[B0,B1],B_TIMES,next_label)
    if gB is None:
        raise SystemExit("no valid causal fixed-boundary gluing found for patch B")
    statesAB,timesAB,nextB,metaB=gB

    gC=glue_second(statesAB,timesAB,[C0,C1],C_TIMES,nextB)
    if gC is None:
        raise SystemExit("no valid causal fixed-boundary gluing found for patch C")
    states,timesABC,nextC,metaC=gC

    # Ordering from glue_second is product order:
    # AB states: A major, B minor; final: AB major, C minor.
    # index bits are (A,B,C).
    expected=8
    if len(states)!=expected:
        raise SystemExit(f"expected {expected} mixed states, got {len(states)}")

    edges=[]
    for idx in range(expected):
        for bit,typ in ((2,"2<->8"),(1,"4<->6"),(0,"3<->3")):
            j=idx^(1<<bit)
            if idx<j:
                edges.append((idx,j,typ))

    connected=graph_connected(expected,edges)
    L=laplacian(expected,edges)
    hermitian=L.H==L
    uniform_zero=L*sp.ones(expected,1)==sp.zeros(expected,1)
    nullity=len(L.nullspace())
    evals=[]
    for val,mult in L.eigenvals().items():
        evals.extend([float(sp.N(val))]*int(mult))
    psd=min(evals)>=-1e-12

    ok_states,boundary_global,state_rows=audit_states(states,timesABC)

    # Negative control: remove one move family.  The cube must split into two
    # disconnected components labelled by that frozen bit.
    reduced=[e for e in edges if e[2]!="3<->3"]
    reduced_connected=graph_connected(expected,reduced)
    reduced_nullity=len(laplacian(expected,reduced).nullspace())
    control=(not reduced_connected) and reduced_nullity==2

    passed=ok0 and ok_states and connected and hermitian and uniform_zero and nullity==1 and psd
    result={
        "schema":1,
        "scope":"finite connected fixed-boundary mixed 4D causal move sector; not scalable ergodicity",
        "evidence":[
            evidence(
                "qccg-cdt-mixed-fixed-boundary-connectivity",
                "CDT_MIXED_MOVE_FIXED_BOUNDARY_CONNECTIVITY_TOY",
                "PASS" if passed else "FAIL",
                "Three boundary-preserving causal move patches (2<->8, 4<->6, 3<->3) are glued into one connected 4D manifold-with-boundary; all eight local-move configurations share the same external boundary and form a connected move graph.",
                patch_b_gluing=metaB,
                patch_c_gluing=metaC,
                n_states=len(states),
                state_rows=state_rows,
                boundary_tetrahedra=len(boundary_global),
                move_edges=[{"a":a,"b":b,"type":t} for a,b,t in edges],
                move_graph_nullity=nullity,
                move_graph_eigenvalues=sorted(evals),
            ),
            evidence(
                "qccg-cdt-mixed-move-family-control",
                "CDT_MOVE_FAMILY_CONNECTIVITY_CONTROL",
                "PASS" if control else "FAIL",
                "Removing the self-dual 3<->3 move family disconnects the finite mixed sector into two components, showing that move-family completeness is operationally visible in the connectivity audit.",
                reduced_edges=[{"a":a,"b":b,"type":t} for a,b,t in reduced],
                reduced_connected=reduced_connected,
                reduced_nullity=reduced_nullity,
            ),
            evidence(
                "qccg-cdt-scalable-connectivity-open-mixed-sector",
                "QCCG_CDT_SCALABLE_CONNECTIVITY",
                "OPEN",
                "Finite mixed fixed-boundary connectivity is established only for the audited glued sector. Connectivity/ergodicity has not been demonstrated across increasing generic causal triangulation sectors.",
                next_step=(
                    "Generate fixed-boundary causal triangulation sectors at increasing volume, "
                    "enumerate/apply the realized local move templates generically, and track "
                    "component counts plus candidate conserved invariants."
                ),
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("mixed CDT connectivity toy failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
