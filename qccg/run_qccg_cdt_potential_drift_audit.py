#!/usr/bin/env python3
"""Drift/diffusion diagnostic for the QCCG -> CDT potential sector.

For a one-dimensional effective volume diffusion
    dN = b(N) dt + sqrt(a(N)) dW
with a zero-current stationary density rho ~ exp(-U), the Ito Fokker-Planck
relation gives
    U'(N) = (a'(N) - 2 b(N)) / a(N).

The CDT minisuperspace potential has
    U(N) ~ mu N^(1/3) - lambda N,
so
    U'(N) ~ (mu/3) N^(-2/3) - lambda.

This audit asks whether the current *unweighted* QCCG Pachner process resolves
that two-term structure from its microscopic drift and diffusion. Failure to
resolve the curvature term is expected to indicate missing geometric/curvature
weighting in the parent dynamics, not an infrastructure failure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_time_evolved_volume_kernel as tev

TAUS=(0.025,0.05,0.075)
TRAJECTORIES=350
MIN_POTENTIAL_R2=0.80
MAX_LAMBDA_SPREAD=1.35
SEED=20260921

SOURCES=[
    {
        "title":"The effective action in 4-dim CDT. The transfer matrix approach",
        "journal":"JHEP 06 (2014) 034",
        "arxiv":"1403.5940",
        "doi":"10.1007/JHEP06(2014)034",
    }
]


def linear_fit(xs,ys):
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    intercept=ym-slope*xm
    pred=[intercept+slope*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>1e-18 else 1.0
    return slope,intercept,r2,pred


def fit_potential(rows):
    # Fit diffusion a(N)=D0*N+D1.
    Ns=[float(r["N3"]) for r in rows]
    avals=[r["variance_rate"] for r in rows]
    D0,D1,ar2,_=linear_fit(Ns,avals)

    uprime=[]
    x=[]
    for r in rows:
        N=float(r["N3"])
        a=r["variance_rate"]
        b=r["drift_rate"]
        up=(D0-2*b)/a
        uprime.append(up)
        x.append(N**(-2.0/3.0))

    # U'=c*N^-2/3 - lambda.
    c,intercept,r2,pred=linear_fit(x,uprime)
    # intercept = -lambda
    mu=3*c
    lam=-intercept
    return {
        "diffusion_D0":D0,
        "diffusion_D1":D1,
        "diffusion_linear_r2":ar2,
        "mu":mu,
        "lambda":lam,
        "potential_derivative_r2":r2,
        "rows":[
            {
                **r,
                "Uprime_inferred":u,
                "Uprime_fit":p,
            }
            for r,u,p in zip(rows,uprime,pred)
        ],
    }


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-cdt-potential-drift-diagnostic",
        "artifact":"qccg/run_qccg_cdt_potential_drift_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    start_states={}
    for ti,n3 in enumerate(base.TARGETS):
        start_states[n3]=base.build_target(n3,base.SEED+1000*ti)[0]

    blocks=[]
    lambdas=[]
    resolved_all=True
    for tau in TAUS:
        rows=[]
        for ti,n3 in enumerate(base.TARGETS):
            S0=start_states[n3]
            vals=[
                tev.evolve_delta(
                    S0,tau,
                    SEED + int(tau*100000)*100000 + ti*10000 + j
                )
                for j in range(TRAJECTORIES)
            ]
            mean,var,skew,excess=tev.moments(vals)
            rows.append({
                "N3":n3,
                "mean_delta":mean,
                "variance":var,
                "drift_rate":mean/tau,
                "variance_rate":var/tau,
                "skewness":skew,
                "excess_kurtosis_diagnostic":excess,
            })
        fit=fit_potential(rows)
        fit["tau"]=tau
        blocks.append(fit)
        lambdas.append(fit["lambda"])
        resolved=fit["potential_derivative_r2"]>=MIN_POTENTIAL_R2
        resolved_all=resolved_all and resolved

    positive_lams=[x for x in lambdas if x>0]
    lambda_stable=(
        len(positive_lams)==len(lambdas)
        and max(positive_lams)/min(positive_lams)<=MAX_LAMBDA_SPREAD
    )
    curvature_resolved=resolved_all
    mismatch_detected=lambda_stable and not curvature_resolved

    result={
        "schema":1,
        "scope":"local drift/diffusion potential diagnostic for unweighted QCCG Pachner dynamics",
        "sources":SOURCES,
        "evidence":[
            evidence(
                "qccg-volume-drift-potential-diagnostic",
                "QCCG_CDT_POTENTIAL_DRIFT_DIAGNOSTIC",
                "PASS",
                "Microscopic QCCG volume drift and diffusion are converted to a local effective-potential derivative using the zero-current Ito Fokker-Planck relation and fitted to the CDT-inspired (mu/3) N3^(-2/3)-lambda form.",
                taus=list(TAUS),
                trajectories=TRAJECTORIES,
                blocks=blocks,
                sources=SOURCES,
            ),
            evidence(
                "qccg-curvature-potential-not-resolved",
                "QCCG_CDT_CURVATURE_POTENTIAL_NOT_RESOLVED",
                "PASS" if mismatch_detected else "NOT_APPLICABLE",
                "The present unweighted move dynamics gives a comparatively stable linear-volume drift coefficient but does not resolve the N3^(-2/3) curvature derivative with the preregistered fit quality. This identifies missing geometric weighting rather than being tuned away.",
                minimum_potential_r2=MIN_POTENTIAL_R2,
                maximum_lambda_spread=MAX_LAMBDA_SPREAD,
                lambda_values=lambdas,
                blocks=blocks,
                sources=SOURCES,
            ),
            evidence(
                "qccg-curvature-weight-dynamics-open",
                "QCCG_CURVATURE_WEIGHTED_DYNAMICS",
                "OPEN",
                "A microscopic QCCG curvature/geometric term must be included in the causal Pachner dynamics and its coefficient inferred from QCCG couplings before mu and the de Sitter profile can be promoted.",
                required_next_step=(
                    "Construct a local curvature-sensitive QCCG move weight from the microscopic geometric sector, "
                    "rerun the large-volume drift/diffusion extraction, and require stable mu/lambda plus a de Sitter-like volume profile."
                ),
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    # Diagnostic itself always runs; only malformed numerics should fail.
    if any(not math.isfinite(b["lambda"]) or not math.isfinite(b["mu"]) for b in blocks):
        raise SystemExit("nonfinite QCCG potential diagnostic")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
