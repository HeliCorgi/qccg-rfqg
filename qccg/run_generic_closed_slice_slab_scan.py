#!/usr/bin/env python3
"""Generic closed-slice strict-slab scan for QCCG local Pachner moves.

This stress-tests constructive builders beyond one hand-picked triangulation.

For several deterministic random seeds:
1. start from the boundary of a 4-simplex (S^3);
2. grow a closed triangulation by a mix of valid spatial 1->4 and 2->3 moves;
3. build a complete closed-slice causal slab for a random 1->4 move;
4. build a complete closed-slice causal slab for a valid 2->3 move.

The 2->3 builder explicitly quotients vertex labels: it relabels the shared
triangle to canonical vertices 0,1,2, the two opposite vertices to 3,4 and all
spectator vertices above 4 before inserting the universal nine-simplex strict
slab.  This makes the construction invariant under arbitrary input labels.

A negative control deliberately assigns canonical roles by numeric order rather
than Pachner incidence; on a permuted example it fails the closed-slice audit.

This provides finite randomized evidence for a generic local builder.  It does
not prove arbitrary global history composition.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
from pathlib import Path


SEEDS=(11,29,47,83,131)
GROW_STEPS=9


def V(t,i): return (t,i)


def boundary_s3():
    vs=set(range(5))
    return {tuple(sorted(vs-{omit})) for omit in vs}


def face_map3(S):
    m=collections.defaultdict(list)
    for tet in S:
        for tri in itertools.combinations(tet,3):
            m[tuple(sorted(tri))].append(tet)
    return m


def edges3(S):
    return {tuple(sorted(e)) for tet in S for e in itertools.combinations(tet,2)}


def move14(S,tet,newv):
    S2=set(S);S2.remove(tet);T=set(tet)
    for omit in tet:
        S2.add(tuple(sorted({newv}|(T-{omit}))))
    return S2


def candidates23(S):
    es=edges3(S);out=[]
    for face,pair in face_map3(S).items():
        if len(pair)!=2: continue
        d=next(iter(set(pair[0])-set(face)))
        e=next(iter(set(pair[1])-set(face)))
        if tuple(sorted((d,e))) in es: continue
        new=tuple(
            tuple(sorted({d,e}|(set(face)-{omit})))
            for omit in face
        )
        if all(t not in S for t in new):
            out.append((tuple(face),tuple(pair),d,e,new))
    return out


def apply23(S,c):
    face,pair,d,e,new=c
    S2=set(S)
    for t in pair:S2.remove(t)
    S2.update(new)
    return S2


def staircase(tet):
    v=sorted(tet);out=set()
    for j in range(4):
        s=[V(0,v[i]) for i in range(j+1)]
        s += [V(1,v[i]) for i in range(j,4)]
        out.add(tuple(sorted(s)))
    return out


def product_slab(S):
    return set().union(*(staircase(t) for t in S))


def face_map4(S):
    m=collections.defaultdict(list)
    for s in S:
        for f in itertools.combinations(s,4):
            m[tuple(sorted(f))].append(s)
    return m


def boundary4(S):
    return {f for f,inc in face_map4(S).items() if len(inc)==1}


def manifold4(S):
    vals=[len(x) for x in face_map4(S).values()]
    return bool(vals) and all(n in (1,2) for n in vals)


def causal4(S):
    for s in S:
        cnt=collections.Counter(t for t,_ in s)
        if len(cnt)!=2 or sorted(cnt)!=[0,1]:
            return False
        if tuple(sorted(cnt.values())) not in {(1,4),(2,3)}:
            return False
    return True


def dual_connected(S):
    sims=list(S);idx={s:i for i,s in enumerate(sims)}
    adj=[set() for _ in sims]
    for inc in face_map4(S).values():
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
        for f in boundary4(S)
        if {tt for tt,_ in f}=={t}
    }


def mixed_boundary(S):
    return {f for f in boundary4(S) if len({t for t,_ in f})>1}


STRICT23={
    (0,1,2,3,8),(0,1,2,4,9),(0,1,2,7,8),
    (0,1,2,7,9),(0,1,6,8,9),(0,1,7,8,9),
    (0,5,6,8,9),(0,5,7,8,9),(1,6,7,8,9),
}
OLD={(0,1,2,3),(0,1,2,4)}
NEW={(1,2,3,4),(0,2,3,4),(0,1,3,4)}


def mapped_strict23():
    return {
        tuple(sorted(V(0,v) if v<5 else V(1,v-5) for v in s))
        for s in STRICT23
    }


def audit_closed(slab,lower,upper):
    return (
        manifold4(slab) and causal4(slab) and dual_connected(slab)
        and spatial_boundary(slab,0)==lower
        and spatial_boundary(slab,1)==upper
        and not mixed_boundary(slab)
    )


def build14(S,tet,newv):
    upper=move14(S,tet,newv)
    slab=product_slab(S)
    top=tuple(sorted(V(1,i) for i in tet))
    adj=[s for s in slab if set(top).issubset(s)]
    if len(adj)!=1:return None,upper
    old=adj[0];opp=next(iter(set(old)-set(top)))
    slab.remove(old)
    nv=V(1,newv)
    for omit in top:
        slab.add(tuple(sorted({opp,nv}|(set(top)-{omit}))))
    return slab,upper


def canonicalize23(S,c):
    face,pair,d,e,_new=c
    f=sorted(face)
    # Canonical role is incidence-based, not numeric-order-based.
    mp={f[0]:0,f[1]:1,f[2]:2,d:3,e:4}
    rest=sorted({v for tet in S for v in tet}-set(mp))
    for i,v in enumerate(rest,start=5):mp[v]=i
    Sr={tuple(sorted(mp[v] for v in tet)) for tet in S}
    return Sr,mp


def build23(S,c):
    Sr,mp=canonicalize23(S,c)
    lower=Sr
    slab=product_slab(lower)
    pair=OLD
    pair_product=set().union(*(staircase(t) for t in pair))
    if not pair_product.issubset(slab):
        return None,None,mp
    slab-=pair_product
    slab|=mapped_strict23()
    upper=(lower-OLD)|NEW
    return slab,upper,mp


def grow(seed):
    rng=random.Random(seed)
    S=boundary_s3();nextv=5
    rows=[]
    for step in range(GROW_STEPS):
        cs=candidates23(S)
        if cs and rng.random()<0.45:
            c=rng.choice(cs)
            S=apply23(S,c)
            rows.append("23")
        else:
            tet=rng.choice(sorted(S))
            S=move14(S,tet,nextv);nextv+=1
            rows.append("14")
    return S,nextv,rows


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"generic-closed-slice-slab-scan",
        "artifact":"qccg/run_generic_closed_slice_slab_scan.py",
        "note":note,"metadata":metadata,
    }


def main():
    rows=[];all14=True;all23=True
    saved_control=None
    for seed in SEEDS:
        S,nextv,history=grow(seed)

        tet=random.Random(seed+1000).choice(sorted(S))
        slab14,upper14=build14(S,tet,nextv)
        ok14=slab14 is not None and audit_closed(slab14,S,upper14)
        all14=all14 and ok14

        cs=candidates23(S)
        if not cs:
            # One deterministic 1->4 refinement always creates candidate 2->3
            # pairs against neighbouring tetrahedra.
            tet0=sorted(S)[0]
            S=move14(S,tet0,nextv);nextv+=1
            cs=candidates23(S)
        c23=sorted(cs,key=lambda x:(x[0],x[2],x[3]))[0]
        slab23,upper23,mp=build23(S,c23)
        Sr={tuple(sorted(mp[v] for v in tet)) for tet in S}
        ok23=slab23 is not None and audit_closed(slab23,Sr,upper23)
        all23=all23 and ok23

        rows.append({
            "seed":seed,"history":history,
            "n_tetrahedra":len(S),"n_vertices":len({v for t in S for v in t}),
            "move14_ok":ok14,"move23_ok":ok23,
            "canonical23_map":{str(k):v for k,v in sorted(mp.items())},
        })
        if saved_control is None:
            saved_control=(S,c23)

    # Negative control: assign the five move vertices canonical roles by their
    # raw numeric ordering instead of by shared-face/opposite-vertex incidence.
    S,c=saved_control
    face,pair,d,e,_new=c
    five=sorted(set(face)|{d,e})
    wrong={v:i for i,v in enumerate(five)}
    rest=sorted({v for tet in S for v in tet}-set(wrong))
    for i,v in enumerate(rest,start=5):wrong[v]=i
    Sw={tuple(sorted(wrong[v] for v in tet)) for tet in S}
    bad=product_slab(Sw)
    pairprod=set().union(*(staircase(t) for t in OLD))
    if pairprod.issubset(bad):
        bad-=pairprod;bad|=mapped_strict23()
        badupper=(Sw-OLD)|NEW
        wrong_accidentally_works=audit_closed(bad,Sw,badupper)
    else:
        wrong_accidentally_works=False
    control=not wrong_accidentally_works

    result={
        "schema":1,
        "scope":"deterministic randomized closed-S3 stress scan of constructive local 1->4 and 2->3 strict slab builders",
        "evidence":[
            evidence(
                "qccg-generic-closed-slice-slab-scan",
                "GENERIC_LOCAL_SLAB_BUILDER_SCAN",
                "PASS" if all14 and all23 else "FAIL",
                "Across several grown closed S3 triangulations, constructive 1->4 and incidence-canonicalized 2->3 builders produce complete adjacent-slice causal 4D slabs with exactly the intended lower and upper spatial boundaries.",
                seeds=list(SEEDS),grow_steps=GROW_STEPS,rows=rows,
            ),
            evidence(
                "qccg-slab-label-gauge-control",
                "LOCAL_SLAB_LABEL_GAUGE_CONTROL",
                "PASS" if control else "FAIL",
                "A numeric-order role assignment is rejected on a relabelled 2->3 move, while incidence-based canonical roles are used by the passing builder; vertex labels remain gauge.",
                wrong_mapping={str(k):v for k,v in sorted(wrong.items())},
                wrong_accidentally_works=wrong_accidentally_works,
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("generic closed-slice slab scan failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
