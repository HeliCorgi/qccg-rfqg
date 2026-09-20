#!/usr/bin/env python3
"""Ward-selection audit for the relative coefficients of the cubic spin-2 vertex.

The existing QCCG tensor-cubic construction inserts the continuum
Einstein-Hilbert Gamma-Gamma cubic structure.  Here we weaken that input by
splitting the cubic coefficient into two independent structures

    alpha * eta^{mn} B12_mn + beta * P1^{mn} B11_mn

and ask whether on-shell pure-gauge replacement conditions select a unique
ratio alpha:beta.

This audit is now a negative-control test.  If each structure separately
passes the selected on-shell three-point pure-gauge replacement test, then
that test cannot determine the relative coefficient.  In that case the audit
passes by detecting underdetermination and requires a stronger selector such
as off-shell Noether consistency or four-point factorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

I = sp.I
eta = sp.diag(1,-1,-1,-1)

TLAM = (sp.Integer(1), sp.Integer(0))
LAMBDA_CASES = (
    ((1,0),(0,1),(-1,-1)),
    ((1,1),(1,-2),(-2,1)),
    ((2,1),(-1,2),(-1,-3)),
)


def angle(a,b):
    return a[0]*b[1]-a[1]*b[0]


def square(a,b):
    return a[0]*b[1]-a[1]*b[0]


def vec_from_bisp(M):
    return sp.Matrix([
        sp.simplify((M[0,0]+M[1,1])/2),
        sp.simplify((M[0,1]+M[1,0])/2),
        sp.simplify((M[1,0]-M[0,1])/(2*I)),
        sp.simplify((M[0,0]-M[1,1])/2),
    ])


def momentum(lam,tlam):
    return vec_from_bisp(sp.Matrix(lam)*sp.Matrix(tlam).T)


def pol_plus(lam,tlam,tq=(0,1)):
    return vec_from_bisp(sp.Matrix(lam)*sp.Matrix(tq).T/square(tlam,tq))


def pol_minus(lam,tlam,q):
    return vec_from_bisp(sp.Matrix(q)*sp.Matrix(tlam).T/angle(q,lam))


def lower(v):
    return eta*v


def graviton_pol(e):
    ec=lower(e)
    return ec*ec.T


def choose_minus_ref(lam):
    for q in ((1,0),(0,1),(1,1)):
        if angle(q,lam)!=0:
            return q
    raise RuntimeError("no negative-helicity reference")


def physical_eps(lams):
    out=[]
    for i,lam in enumerate(lams):
        if i<2:
            e=pol_plus(lam,TLAM)
        else:
            e=pol_minus(lam,TLAM,choose_minus_ref(lam))
        out.append(graviton_pol(e))
    return out


def gauge_pol(k,xi):
    kc=lower(k)
    xc=lower(sp.Matrix(xi))
    return kc*xc.T+xc*kc.T


def cubic_components(eps_list,ks):
    """Return coefficients (A,B) for eta*b12 and P1*b11 structures."""
    ts=sp.symbols("t1 t2 t3")
    h=sp.zeros(4)
    dh=[sp.zeros(4) for _ in range(4)]
    for tv,eps,k in zip(ts,eps_list,ks):
        h += tv*eps
        kc=lower(k)
        for a in range(4):
            dh[a] += I*tv*kc[a]*eps

    hup=eta*h*eta
    htr=sp.trace(eta*h)
    P1=sp.Rational(1,2)*htr*eta-hup

    G1=[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    G2=[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for m in range(4):
            for n in range(4):
                g1=0
                g2=0
                for s in range(4):
                    X=dh[m][s,n]+dh[n][s,m]-dh[s][m,n]
                    g1 += sp.Rational(1,2)*eta[r,s]*X
                    g2 += -sp.Rational(1,2)*hup[r,s]*X
                G1[r][m][n]=sp.expand(g1)
                G2[r][m][n]=sp.expand(g2)

    A=0
    B=0
    for m in range(4):
        for n in range(4):
            b11=0
            b12=0
            for r in range(4):
                for s in range(4):
                    b11 += G1[r][m][s]*G1[s][n][r]-G1[r][m][n]*G1[s][r][s]
                    b12 += (
                        G1[r][m][s]*G2[s][n][r]
                        +G2[r][m][s]*G1[s][n][r]
                        -G1[r][m][n]*G2[s][r][s]
                        -G2[r][m][n]*G1[s][r][s]
                    )
            A += eta[m,n]*b12
            B += P1[m,n]*b11

    def coeff(expr):
        p=sp.expand(expr)
        return sp.simplify(p.coeff(ts[0],1).coeff(ts[1],1).coeff(ts[2],1))

    return coeff(A),coeff(B)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id":eid,
        "obligation":obligation,
        "status":status,
        "engine":"sympy-cubic-ward-coefficient-selection",
        "artifact":"qccg/run_cubic_coefficient_selection_audit.py",
        "note":note,
        "metadata":metadata,
    }


def main():
    rows=[]
    ward_rows=[]
    phys_rows=[]
    xis=((0,1,1,0),(1,0,1,1),(1,1,0,-1))

    for case_id,lams0 in enumerate(LAMBDA_CASES,start=1):
        lams=tuple(tuple(sp.Integer(x) for x in lam) for lam in lams0)
        ks=tuple(momentum(lam,TLAM) for lam in lams)
        eps=physical_eps(lams)
        Aphys,Bphys=cubic_components(eps,ks)
        phys_rows.append({
            "case":case_id,
            "A":str(Aphys),"B":str(Bphys),
            "EH_sum":str(sp.simplify(Aphys+Bphys)),
        })
        for leg in range(3):
            eg=list(eps)
            eg[leg]=gauge_pol(ks[leg],xis[leg])
            Ag,Bg=cubic_components(eg,ks)
            ward_rows.append((Ag,Bg))
            rows.append({
                "case":case_id,"leg":leg+1,
                "A_gauge":str(Ag),"B_gauge":str(Bg),
                "EH_sum":str(sp.simplify(Ag+Bg)),
            })

    M=sp.Matrix([[a,b] for a,b in ward_rows])
    null=M.nullspace()
    rank=int(M.rank())

    gauge_eh_zero=all(sp.simplify(a+b)==0 for a,b in ward_rows)
    phys_nonzero=all(sp.simplify(sp.sympify(r["EH_sum"]))!=0 for r in phys_rows)

    alpha_alone_gauge_zero=all(a==0 for a,_b in ward_rows)
    beta_alone_gauge_zero=all(b==0 for _a,b in ward_rows)
    underdetermined=(rank==0 and len(null)==2 and alpha_alone_gauge_zero and beta_alone_gauge_zero)

    result={
        "schema":1,
        "scope":"two-parameter cubic Gamma-Gamma ansatz Ward selection; not microscopic RG derivation",
        "evidence":[
            evidence(
                "qccg-cubic-onshell-underdetermination",
                "CUBIC_ONSHELL_WARD_UNDERDETERMINATION_DETECTED",
                "PASS" if underdetermined and gauge_eh_zero and phys_nonzero else "FAIL",
                "The audited on-shell three-point pure-gauge conditions have rank zero on the two cubic structures: each structure separately passes the gauge-replacement test, so this test cannot select the Einstein-Hilbert relative coefficient.",
                matrix_rank=rank,
                ward_matrix=[[str(x) for x in row] for row in M.tolist()],
                nullspace=[ [str(x) for x in v] for v in null ],
                alpha_alone_gauge_zero=alpha_alone_gauge_zero,
                beta_alone_gauge_zero=beta_alone_gauge_zero,
                ward_rows=rows,
                physical_rows=phys_rows,
            ),
            evidence(
                "qccg-cubic-stronger-selector-open",
                "CUBIC_OFFSHELL_OR_FOURPOINT_SELECTOR",
                "OPEN",
                "The relative cubic coefficients require a stronger consistency condition than on-shell three-point gauge replacement, such as off-shell Noether consistency with the quadratic action or four-point factorization/unitarity.",
            ),
            evidence(
                "qccg-cubic-larger-basis-open",
                "QCCG_CUBIC_LARGER_BASIS_UNIQUENESS",
                "OPEN",
                "A complete independent local two-derivative spin-2 cubic operator basis has not been enumerated and reduced modulo field redefinitions.",
            ),
        ],
    }

    p=Path(args.out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"]):
        raise SystemExit("cubic on-shell underdetermination audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
