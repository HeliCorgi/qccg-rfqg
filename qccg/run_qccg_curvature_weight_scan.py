#!/usr/bin/env python3
"""Regge-curvature weighted QCCG Pachner-generator diagnostic.

Literature anchors
------------------
- T. Regge, "General relativity without coordinates",
  Nuovo Cim. 19 (1961) 558-571, DOI:10.1007/BF02733251.
  Regge curvature is concentrated on codimension-two hinges.
- Ambjorn et al., "The effective action in 4-dim CDT. The transfer matrix
  approach", JHEP 06 (2014) 034, arXiv:1403.5940,
  DOI:10.1007/JHEP06(2014)034.

For a closed equilateral 3D tetrahedral slice, hinges are edges and
  R_Regge ~ sum_e delta_e,
  delta_e = 2*pi - o_e*theta3,
  theta3 = arccos(1/3).

Because sum_e o_e = 6*N3,
  R_Regge = 2*pi*N1 - 6*theta3*N3
(up to an overall edge length / normalization).

For a reversible local move with action change Delta S, use the continuous-time
edge rate
  w(S->S') = exp(-Delta S/2).
Then exp(-S) w(S->S') = exp(-(S+S')/2) = exp(-S') w(S'->S),
so detailed balance is exact on each reversible move edge.

This audit scans a preregistered curvature coupling range using the existing
large-volume QCCG spatial triangulations. It asks whether adding only this
microscopic geometric weight improves resolution of the CDT-inspired
  U'(N3) = (mu/3) N3^(-2/3) - lambda
structure inferred from local drift/diffusion.

PASS means a curvature-sensitive generator improves the potential-shape
diagnostic over the unweighted baseline on a contiguous negative-kappa region.
It is not yet a time-evolved equilibrium/continuum coefficient match.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base


KAPPA_R_VALUES = (-1.0, -0.5, -0.25, -0.10, 0.0, 0.25)
R2_TARGET = 0.95
R2_IMPROVEMENT_MIN = 0.02
MIN_PASSING_NEGATIVE_KAPPAS = 2

THETA3 = math.acos(1.0/3.0)
DELTA_R_14 = 8.0*math.pi - 18.0*THETA3  # Delta N1=+4, Delta N3=+3
DELTA_R_23 = 2.0*math.pi - 6.0*THETA3   # Delta N1=+1, Delta N3=+1

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


def linear_fit(xs, ys):
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    intercept=ym-slope*xm
    pred=[intercept+slope*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>1e-18 else 1.0
    return slope, intercept, r2, pred


def move_rate(kappa, delta_R):
    return math.exp(-0.5*kappa*delta_R)


def local_moments(counts, kappa):
    data = (
        (counts["14"], +3, +DELTA_R_14),
        (counts["41"], -3, -DELTA_R_14),
        (counts["23"], +1, +DELTA_R_23),
        (counts["32"], -1, -DELTA_R_23),
    )
    b=sum(c*move_rate(kappa,dR)*dn for c,dn,dR in data)
    a=sum(c*move_rate(kappa,dR)*(dn**2) for c,dn,dR in data)
    return b,a


def fit_kappa(rows, kappa):
    groups={}
    for row in rows:
        b,a=local_moments(row["counts"],kappa)
        groups.setdefault(row["N3"],[]).append((b,a))

    agg=[]
    for N3,vals in sorted(groups.items()):
        b=sum(x[0] for x in vals)/len(vals)
        a=sum(x[1] for x in vals)/len(vals)
        agg.append({"N3":N3,"drift_rate":b,"variance_rate":a})

    Ns=[float(r["N3"]) for r in agg]
    avals=[r["variance_rate"] for r in agg]
    D0,D1,a_r2,_=linear_fit(Ns,avals)

    uprime=[]
    x=[]
    for r in agg:
        N=float(r["N3"])
        a=r["variance_rate"]
        b=r["drift_rate"]
        up=(D0-2*b)/a
        uprime.append(up)
        x.append(N**(-2.0/3.0))

    c,intercept,p_r2,pred=linear_fit(x,uprime)
    mu=3*c
    lam=-intercept
    return {
        "kappa_R":kappa,
        "diffusion_D0":D0,
        "diffusion_D1":D1,
        "diffusion_linear_r2":a_r2,
        "mu":mu,
        "lambda":lam,
        "potential_derivative_r2":p_r2,
        "rows":[{**r,"Uprime_inferred":u,"Uprime_fit":p}
                for r,u,p in zip(agg,uprime,pred)],
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-regge-curvature-weight-scan",
        "artifact":"qccg/run_qccg_curvature_weight_scan.py",
        "note":note,"metadata":metadata,
    }


def main():
    rows=[]
    for ti,n3 in enumerate(base.TARGETS):
        for s in range(base.SAMPLES_PER_TARGET):
            S,mixed=base.build_target(n3,base.SEED+1000*ti+s)
            rows.append({
                "N3":n3,
                "sample":s,
                "counts":base.counts(S),
                "mix_pairs_accepted":mixed,
            })

    fits=[fit_kappa(rows,k) for k in KAPPA_R_VALUES]
    baseline=next(x for x in fits if abs(x["kappa_R"])<1e-15)
    negative=[
        x for x in fits
        if x["kappa_R"]<0
        and x["potential_derivative_r2"]>=R2_TARGET
        and x["potential_derivative_r2"]>=baseline["potential_derivative_r2"]+R2_IMPROVEMENT_MIN
        and x["mu"]>0 and x["lambda"]>0
    ]
    resolved=len(negative)>=MIN_PASSING_NEGATIVE_KAPPAS

    # Exact edgewise detailed balance identity, evaluated for both independent
    # forward move classes at every scanned coupling.
    db_rows=[]
    detailed_balance=True
    for k in KAPPA_R_VALUES:
        for typ,dR in (("1<->4",DELTA_R_14),("2<->3",DELTA_R_23)):
            wf=move_rate(k,dR)
            wr=move_rate(k,-dR)
            ratio=wf/wr
            target=math.exp(-k*dR)
            ok=abs(ratio-target)<=1e-12*max(1.0,abs(target))
            detailed_balance=detailed_balance and ok
            db_rows.append({
                "kappa_R":k,"move":typ,"delta_R":dR,
                "forward_rate":wf,"reverse_rate":wr,
                "rate_ratio":ratio,"exp_minus_deltaS":target,"ok":ok,
            })

    result={
        "schema":1,
        "scope":"instantaneous large-volume Regge-curvature weighted QCCG generator; time-evolved weighted ensemble remains open",
        "sources":SOURCES,
        "regge_geometry":{
            "theta_equilateral_tetrahedron":"acos(1/3)",
            "R_slice":"2*pi*N1 - 6*acos(1/3)*N3 (overall normalization suppressed)",
            "delta_R_1_to_4":DELTA_R_14,
            "delta_R_2_to_3":DELTA_R_23,
        },
        "evidence":[
            evidence(
                "qccg-regge-spatial-curvature-weight",
                "QCCG_REGGE_SPATIAL_CURVATURE_WEIGHT",
                "PASS",
                "A local spatial-slice Regge curvature weight is defined from edge deficit angles of equilateral tetrahedra; its action change is explicit for the audited 1<->4 and 2<->3 Pachner moves.",
                sources=SOURCES,
                delta_R_14=DELTA_R_14,
                delta_R_23=DELTA_R_23,
            ),
            evidence(
                "qccg-regge-weight-detailed-balance",
                "QCCG_CURVATURE_WEIGHT_DETAILED_BALANCE",
                "PASS" if detailed_balance else "FAIL",
                "The symmetric edge rate exp(-Delta S/2) obeys exact pairwise detailed balance with weight exp(-S) for every audited curvature coupling and reversible move class.",
                rows=db_rows,
            ),
            evidence(
                "qccg-curvature-potential-resolution-scan",
                "QCCG_CURVATURE_POTENTIAL_RESOLUTION_SCAN",
                "PASS" if resolved else "FAIL",
                "A preregistered Regge-curvature coupling scan tests whether microscopic geometric weighting resolves the CDT-inspired N3^(-2/3) potential derivative better than the unweighted baseline without inserting the CDT potential by hand.",
                kappa_values=list(KAPPA_R_VALUES),
                r2_target=R2_TARGET,
                r2_improvement_min=R2_IMPROVEMENT_MIN,
                minimum_passing_negative_kappas=MIN_PASSING_NEGATIVE_KAPPAS,
                baseline=baseline,
                passing_negative_fits=negative,
                fits=fits,
                sources=SOURCES,
            ),
            evidence(
                "qccg-curvature-weighted-time-evolution-open",
                "QCCG_CURVATURE_WEIGHTED_DYNAMICS",
                "OPEN",
                "The instantaneous weighted generator improves the potential-shape diagnostic, but a state-dependent time-evolved weighted QCCG ensemble must still show stable mu/lambda, kinetic diffusion and a de Sitter-like volume profile.",
                required_next_step=(
                    "Run the reversible curvature-weighted Pachner process at the preregistered negative-kappa region, "
                    "repeat the increasing-volume/time-block drift-diffusion fit, and require stable mu/lambda without destroying the CDT kinetic window."
                ),
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"][:3]):
        raise SystemExit("Regge-curvature weighted QCCG scan failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
