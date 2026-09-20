#!/usr/bin/env python3
"""Search for simple conserved f-vector invariants of the realized causal move set.

Uses the explicit audited local move realizations:
- causal 2->4 Pachner move on the foliated cylinder;
- CDT-style 2->8 vertex move;
- CDT-style 4->6 connectivity move;
- CDT-style 3->3 self-dual move.

For each move we compute Δ(f0,...,f4), the rational linear invariants, and
component-wise modular invariants from gcds of the move deltas.

A fixed-boundary 4-manifold necessarily satisfies the homogeneous incidence
identity
    2 Δf3 = 5 Δf4,
and the 4D Dehn--Sommerville move relation
    2 Δf1 - 3 Δf2 + 4 Δf3 - 5 Δf4 = 0.
Hence Δf4 is even, Δf3 is divisible by five, and Δf2 is also even.
These apparent modular invariants are manifold-relation consequences, not
evidence of move-set nonergodicity.

Any additional component-wise modulus not implied by the audited manifold
relations is treated as an unexpected candidate obstruction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from functools import reduce
from pathlib import Path

import sympy as sp

import run_causal_move_dynamics_toy as m24
import run_cdt_move4_audit as m28
import run_cdt_move46_audit as m46
import run_cdt_move33_audit as m33


def fvector(S):
    vals=[]
    for size in range(1,6):
        subs=set()
        for s in S:
            subs.update(tuple(sorted(x)) for x in itertools.combinations(s,size))
        vals.append(len(subs))
    return vals


def delta(A,B):
    fa,fb=fvector(A),fvector(B)
    return [b-a for a,b in zip(fa,fb)]


def first_valid_24():
    S0,spatial=m24.build()
    for cand in m24.candidates_2_to_4(S0):
        S1=m24.apply_2_to_4(S0,cand)
        if m24.audit(S1,spatial)["valid"]:
            return S0,S1,cand
    raise RuntimeError("no valid causal 2->4 move found")


def gcd_abs(xs):
    vals=[abs(int(x)) for x in xs if int(x)!=0]
    return 0 if not vals else reduce(math.gcd,vals)


def boundary_tetrahedra(S):
    return {f for f,inc in m28.face_map(S).items() if len(inc)==1}


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"cdt-move-invariant-audit",
        "artifact":"qccg/run_cdt_move_invariant_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    A24,B24,c24=first_valid_24()
    moves={
        "2->4":(A24,B24),
        "2->8":(m28.INITIAL,m28.FINAL),
        "4->6":(m46.INITIAL,m46.FINAL),
        "3->3":(m33.INITIAL,m33.FINAL),
    }
    deltas={name:delta(A,B) for name,(A,B) in moves.items()}

    M=sp.Matrix.hstack(*[sp.Matrix(deltas[name]) for name in moves])
    null=[list(map(int,v)) for v in M.T.nullspace()]

    euler=sp.Matrix([1,-1,1,-1,1])
    euler_ok=all(int((euler.T*sp.Matrix(d))[0])==0 for d in deltas.values())

    incidence_ok=all(2*d[3]-5*d[4]==0 for d in deltas.values())
    dehn_sommerville_ok=all(
        2*d[1]-3*d[2]+4*d[3]-5*d[4]==0
        for d in deltas.values()
    )

    boundary_rows={}
    boundary_preserved=True
    for name,(A,B) in moves.items():
        ba=boundary_tetrahedra(A)
        bb=boundary_tetrahedra(B)
        same=(ba==bb)
        boundary_rows[name]={
            "before":len(ba),"after":len(bb),"same_exact_boundary":same,
        }
        boundary_preserved=boundary_preserved and same

    component_gcds=[
        gcd_abs([deltas[name][i] for name in moves])
        for i in range(5)
    ]
    modular={
        f"f{i}":g for i,g in enumerate(component_gcds) if g>1
    }
    expected_boundary_modular={
        # Δf4 even and Δf3 multiple of 5 from 2Δf3=5Δf4.
        "f3":5,
        "f4":2,
        # Mod 2, Dehn--Sommerville gives Δf2 ≡ Δf4 (mod 2),
        # hence Δf2 is even because Δf4 is even.
        "f2":2,
    }
    unexpected={
        k:v for k,v in modular.items()
        if expected_boundary_modular.get(k)!=v
    }

    # Explicitly verify the parity/mod-5 changes are consequences of incidence.
    boundary_moduli_explained=(
        incidence_ok
        and dehn_sommerville_ok
        and component_gcds[2]==2
        and component_gcds[3]==5
        and component_gcds[4]==2
    )

    passed=(
        euler_ok
        and incidence_ok
        and dehn_sommerville_ok
        and boundary_preserved
        and boundary_moduli_explained
        and not unexpected
    )

    result={
        "schema":1,
        "scope":"simple f-vector invariant search for the realized finite causal move templates",
        "evidence":[
            evidence(
                "qccg-cdt-move-invariant-audit",
                "CDT_MOVE_INVARIANT_AUDIT",
                "PASS" if passed else "FAIL",
                "The audited move deltas preserve Euler, fixed-boundary codimension-one incidence, and the 4D Dehn--Sommerville move relation. The visible f2 mod 2, f3 mod 5 and f4 mod 2 invariants are implied by those manifold relations; no additional component-wise modulus is found.",
                deltas=deltas,
                matrix=[[int(x) for x in row] for row in M.tolist()],
                matrix_rank=int(M.rank()),
                rational_left_nullspace=null,
                euler_invariant=euler_ok,
                fixed_boundary_incidence=incidence_ok,
                dehn_sommerville_relation=dehn_sommerville_ok,
                boundary_rows=boundary_rows,
                component_gcds=component_gcds,
                modular_invariants=modular,
                expected_boundary_modular=expected_boundary_modular,
                unexpected_component_moduli=unexpected,
            ),
            evidence(
                "qccg-cdt-boundary-parity-control",
                "CDT_BOUNDARY_MODULAR_INVARIANTS_EXPLAINED",
                "PASS" if boundary_moduli_explained else "FAIL",
                "The apparent N4 parity and N3 mod-5 invariants follow from 2 Δf3 = 5 Δf4, while N2 parity follows from that incidence relation together with the 4D Dehn--Sommerville move relation; none is an extra move-set conservation law.",
                component_gcds=component_gcds,
            ),
            evidence(
                "qccg-cdt-generic-invariant-search-open",
                "QCCG_CDT_GENERIC_INVARIANT_SEARCH",
                "OPEN",
                "The f-vector audit excludes only simple component-wise modular obstructions. Nonlocal/topological invariants of generic causal triangulations still require scalable state-graph sampling.",
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("CDT move invariant audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
