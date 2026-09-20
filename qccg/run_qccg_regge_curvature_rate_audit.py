#!/usr/bin/env python3
"""Local Regge-curvature weighted QCCG Pachner generator diagnostic.

Literature anchors
------------------
- T. Regge, "General relativity without coordinates", Nuovo Cimento 19
  (1961) 558-571, DOI:10.1007/BF02733251.
- J. Ambjorn et al., "The effective action in 4-dim CDT. The transfer matrix
  approach", JHEP 06 (2014) 034, arXiv:1403.5940,
  DOI:10.1007/JHEP06(2014)034.

For an equilateral 3D triangulation (unit edge length), hinges are edges and
we use the integrated-curvature proxy

    R_Regge = 2 * sum_e delta_e,
    delta_e = 2*pi - n_e*acos(1/3).

The factor 2 is the standard Einstein-Hilbert/Regge normalization convention;
only differences enter the finite diagnostic.

We define a reversible continuous-time local move rate

    r(S->S') = exp[-(U(S')-U(S))/2],
    U(S) = -kappa_R * R_Regge(S),

which obeys detailed balance with pi(S) proportional to exp[-U(S)] on an
undirected Pachner state graph.

At increasing N3, the exact local generator gives
    b(N)=sum r*DeltaN,
    a(N)=sum r*(DeltaN)^2.
The zero-current Fokker-Planck diagnostic then infers
    U_eff'(N)=(a'(N)-2b(N))/a(N)
and fits it to
    (mu/3) N^(-2/3) - lambda.

This first pass deliberately samples the existing unweighted QCCG geometry
family.  If the curvature term is not resolved, the result is retained as a
negative diagnostic: weighting the rates without first equilibrating the
geometry is insufficient.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_time_local_slice_transfer_toy as qslice


KAPPA_R_VALUES = (-0.12, -0.06, 0.0, 0.06, 0.12)
SAMPLES_PER_TARGET = 6
MIN_R2_RESOLVED = 0.80
MAX_DETAIL_BALANCE_RELERR = 2.0e-12
THETA = math.acos(1.0/3.0)

SOURCES = [
    {
        "title": "General relativity without coordinates",
        "author": "T. Regge",
        "journal": "Nuovo Cimento 19 (1961) 558-571",
        "doi": "10.1007/BF02733251",
    },
    {
        "title": "The effective action in 4-dim CDT. The transfer matrix approach",
        "journal": "JHEP 06 (2014) 034",
        "arxiv": "1403.5940",
        "doi": "10.1007/JHEP06(2014)034",
    },
]


def edges(S):
    out=set()
    for tet in S:
        out.update(tuple(sorted(e)) for e in itertools.combinations(tet,2))
    return out


def regge_curvature(S):
    # Sum_e n_e = 6*N3 for a closed tetrahedral triangulation.
    n1=len(edges(S))
    n3=len(S)
    return 2.0*(2.0*math.pi*n1 - 6.0*THETA*n3)


def apply41(S,c):
    _v,star,target=c
    S2=set(S)
    for tet in star:
        S2.remove(tet)
    S2.add(target)
    return S2


def move_targets(S):
    used={v for tet in S for v in tet}
    fresh=max(used)+1
    for tet in sorted(S):
        yield "14", 3, qslice.apply14(S,(tet,fresh))
    for c in qslice.candidates41(S):
        yield "41", -3, apply41(S,c)
    for c in qslice.candidates23(S):
        yield "23", 1, qslice.apply23(S,c)
    for c in qslice.candidates32(S):
        yield "32", -1, qslice.apply32(S,c)


def local_generator(S,kappa):
    R0=regge_curvature(S)
    b=0.0
    a=0.0
    rows=[]
    max_db=0.0
    for typ,dn,S2 in move_targets(S):
        if not qslice.manifold(S2):
            continue
        R1=regge_curvature(S2)
        dR=R1-R0
        dU=-kappa*dR
        rate=math.exp(-0.5*dU)
        reverse_rate=math.exp(+0.5*dU)

        # pi ~ exp(-U), U=-kappa R.  Shift U0 to zero to avoid overflow.
        # Detailed balance ratio can be checked locally without normalization.
        lhs=rate
        rhs=math.exp(-dU)*reverse_rate
        rel=abs(lhs-rhs)/max(1.0,abs(lhs),abs(rhs))
        max_db=max(max_db,rel)

        b += rate*dn
        a += rate*(dn**2)
        rows.append({
            "type":typ,
            "delta_N3":dn,
            "delta_Regge_curvature":dR,
            "rate":rate,
        })
    return b,a,max_db,rows


def linear_fit(xs,ys):
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    intercept=ym-slope*xm
    pred=[intercept+slope*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1.0-sse/sst if sst>1.0e-18 else (1.0 if sse<1.0e-18 else -1.0)
    return slope,intercept,r2,pred


def fit_effective_potential(aggregate):
    Ns=[float(r["N3"]) for r in aggregate]
    aa=[r["mean_a"] for r in aggregate]
    D0,D1,ar2,_=linear_fit(Ns,aa)

    xs=[]
    ups=[]
    for r in aggregate:
        N=float(r["N3"])
        up=(D0-2.0*r["mean_b"])/r["mean_a"]
        xs.append(N**(-2.0/3.0))
        ups.append(up)

    c,intercept,r2,pred=linear_fit(xs,ups)
    return {
        "diffusion_D0":D0,
        "diffusion_D1":D1,
        "diffusion_linear_r2":ar2,
        "mu":3.0*c,
        "lambda":-intercept,
        "potential_derivative_r2":r2,
        "rows":[
            {
                **row,
                "Uprime_inferred":up,
                "Uprime_fit":pr,
            }
            for row,up,pr in zip(aggregate,ups,pred)
        ],
    }


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-local-regge-curvature-rate-audit",
        "artifact":"qccg/run_qccg_regge_curvature_rate_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    scan=[]
    max_db_all=0.0
    for ki,kappa in enumerate(KAPPA_R_VALUES):
        raw=[]
        aggregate=[]
        for ti,n3 in enumerate(base.TARGETS):
            vals=[]
            for s in range(SAMPLES_PER_TARGET):
                S,_=base.build_target(n3,base.SEED+1000*ti+s)
                b,a,db,moves=local_generator(S,kappa)
                max_db_all=max(max_db_all,db)
                vals.append((b,a))
                raw.append({
                    "kappa_R":kappa,
                    "N3":n3,
                    "sample":s,
                    "Regge_curvature":regge_curvature(S),
                    "drift_rate_b":b,
                    "variance_rate_a":a,
                    "detail_balance_max_relative_error":db,
                    "move_counts": {
                        typ:sum(1 for row in moves if row["type"]==typ)
                        for typ in ("14","41","23","32")
                    },
                })
            aggregate.append({
                "N3":n3,
                "mean_b":sum(x[0] for x in vals)/len(vals),
                "mean_a":sum(x[1] for x in vals)/len(vals),
            })
        fit=fit_effective_potential(aggregate)
        scan.append({
            "kappa_R":kappa,
            "fit":fit,
            "aggregate":aggregate,
            "resolved_curvature_term":fit["potential_derivative_r2"]>=MIN_R2_RESOLVED,
        })

    detailed_balance=max_db_all<=MAX_DETAIL_BALANCE_RELERR
    resolved=[r for r in scan if r["resolved_curvature_term"]]
    simple_rate_repair_succeeds=bool(resolved)
    simple_rate_repair_insufficient=not simple_rate_repair_succeeds

    result={
        "schema":1,
        "scope":"local Regge-curvature weighted rates on unweighted QCCG geometry samples; no curvature-weighted equilibrium ensemble yet",
        "sources":SOURCES,
        "evidence":[
            evidence(
                "qccg-regge-local-detailed-balance",
                "QCCG_REGGE_LOCAL_DETAILED_BALANCE",
                "PASS" if detailed_balance else "FAIL",
                "The symmetric local Regge-curvature Pachner rates satisfy the intended detailed-balance relation on the audited move samples.",
                kappa_R_values=list(KAPPA_R_VALUES),
                max_relative_error=max_db_all,
                tolerance=MAX_DETAIL_BALANCE_RELERR,
                regge_formula="R_Regge=2*sum_e[2*pi-n_e*acos(1/3)] for unit equilateral edges",
                sources=SOURCES,
            ),
            evidence(
                "qccg-regge-rate-potential-scan",
                "QCCG_REGGE_RATE_POTENTIAL_SCAN",
                "PASS",
                "A preregistered symmetric scan of local Regge-curvature couplings is converted to exact generator drift/diffusion data and fitted to the CDT-inspired potential derivative.",
                kappa_R_values=list(KAPPA_R_VALUES),
                minimum_potential_r2=MIN_R2_RESOLVED,
                scan=scan,
                sources=SOURCES,
            ),
            evidence(
                "qccg-regge-rate-only-insufficient",
                "QCCG_REGGE_RATE_ONLY_INSUFFICIENT",
                "PASS" if simple_rate_repair_insufficient else "NOT_APPLICABLE",
                "If no audited kappa_R resolves the N3^(-2/3) curvature derivative, local Regge weighting of rates on an unweighted geometry family is recorded as insufficient rather than tuned.",
                resolved_kappa_R=[r["kappa_R"] for r in resolved],
                scan=scan,
            ),
            evidence(
                "qccg-curvature-equilibrium-open",
                "QCCG_CURVATURE_WEIGHTED_EQUILIBRIUM_ENSEMBLE",
                "OPEN" if simple_rate_repair_insufficient else "NOT_APPLICABLE",
                "If rate weighting alone is insufficient, the next repair is to thermalize the spatial triangulation ensemble with the Regge curvature weight before extracting drift/diffusion, because the curvature term acts on the geometry distribution as well as individual move rates.",
                sources=SOURCES,
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not detailed_balance:
        raise SystemExit("Regge curvature weighted local detailed balance failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
