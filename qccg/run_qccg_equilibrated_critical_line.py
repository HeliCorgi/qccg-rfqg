#!/usr/bin/env python3
"""Recompute the QCCG curvature/volume critical line on equilibrated fixed-N3 geometries.

The earlier zero-drift line was obtained from deterministic/mildly mixed seed
triangulations.  After curvature-weighted conditional thermalization, the
large-volume move counts shift, so the old kappa_V no longer gives zero drift.

This audit:
1. fixes kappa_R to the preregistered curvature candidate (-1.0);
2. thermalizes independent fixed-volume geometry ensembles at N3=20,41,80,119;
3. solves kappa_V from mean b(N3)/N3=0 over N3>=80 using ONLY those
   equilibrated move counts;
4. re-evaluates the instantaneous coarse potential derivative.

The exact-N3 conditional geometry distribution is independent of kappa_V
because kappa_V*N3 is constant inside each volume sector.  Thus the conditional
samples can be used to solve the volume coupling after sampling.

This is a critical-line selection audit.  A separate fresh-seed time-evolved
confirmation is still required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_curvature_critical_line as oldcrit
import run_qccg_fixed_volume_curvature_thermalization as therm


KAPPA_R=-1.0
SEED=20270501
LARGE_N_MIN=80
ROOT_TOL=1e-10
POTENTIAL_R2_MIN=0.90


def local_moments(counts,kV):
    return oldcrit.local_moments(counts,KAPPA_R,kV)


def thermal_samples():
    # Use the old kappa_V only as a sampler aid. Conditional on exact N3,
    # the kappa_V term is constant and does not alter the target geometry law.
    old_rows=oldcrit.sample_rows()
    old_kv,_=oldcrit.solve_kappa_v(old_rows,KAPPA_R)
    out={}
    meta=[]
    for i,N in enumerate(base.TARGETS):
        t=therm.thermalize_target(N,KAPPA_R,old_kv,SEED+10000*i)
        if len(t["samples"])!=therm.SAMPLES_PER_TARGET:
            raise RuntimeError(f"failed fixed-volume sampling at N3={N}")
        out[N]=t["samples"]
        meta.append({
            "N3":N,
            "acceptance_fraction":t["acceptance_fraction"],
            "sample_stats":t["sample_stats"],
        })
    return out,meta,old_kv


def drift_density(samples,kV):
    vals=[]
    for N,Ss in samples.items():
        if N<LARGE_N_MIN:
            continue
        for S in Ss:
            b,_a=local_moments(base.counts(S),kV)
            vals.append(b/N)
    return sum(vals)/len(vals)


def solve_kv(samples):
    lo,hi=-4.0,4.0
    flo=drift_density(samples,lo)
    fhi=drift_density(samples,hi)
    if flo*fhi>0:
        raise RuntimeError(f"root not bracketed: {flo}, {fhi}")
    for _ in range(100):
        mid=0.5*(lo+hi)
        fm=drift_density(samples,mid)
        if abs(fm)<ROOT_TOL:
            return mid,fm
        if fm>0:
            lo=mid
        else:
            hi=mid
    mid=0.5*(lo+hi)
    return mid,drift_density(samples,mid)


def linear_fit(xs,ys):
    xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    intercept=ym-slope*xm
    pred=[intercept+slope*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>1e-18 else 1.0
    return slope,intercept,r2,pred


def fit_potential(samples,kV):
    agg=[]
    for N,Ss in sorted(samples.items()):
        vals=[local_moments(base.counts(S),kV) for S in Ss]
        agg.append({
            "N3":N,
            "drift_rate":sum(v[0] for v in vals)/len(vals),
            "variance_rate":sum(v[1] for v in vals)/len(vals),
        })

    Ns=[float(r["N3"]) for r in agg]
    aa=[r["variance_rate"] for r in agg]
    D0,D1,ar2,_=linear_fit(Ns,aa)

    x=[];up=[]
    for r in agg:
        N=float(r["N3"]);a=r["variance_rate"];b=r["drift_rate"]
        x.append(N**(-2.0/3.0))
        up.append((D0-2*b)/a)
    c,intercept,r2,pred=linear_fit(x,up)
    return {
        "diffusion_D0":D0,
        "diffusion_D1":D1,
        "diffusion_linear_r2":ar2,
        "mu":3*c,
        "lambda":-intercept,
        "potential_derivative_r2":r2,
        "rows":[{**r,"Uprime_inferred":u,"Uprime_fit":p}
                for r,u,p in zip(agg,up,pred)],
    }


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-equilibrated-critical-line",
        "artifact":"qccg/run_qccg_equilibrated_critical_line.py",
        "note":note,"metadata":metadata,
    }


def main():
    samples,thermal_meta,old_kv=thermal_samples()
    new_kv,res=solve_kv(samples)
    fit=fit_potential(samples,new_kv)

    shifted=abs(new_kv-old_kv)>1e-4
    passed=(
        abs(res)<1e-8
        and fit["potential_derivative_r2"]>=POTENTIAL_R2_MIN
        and fit["mu"]>0
        and fit["lambda"]>0
    )

    result={
        "schema":1,
        "scope":"equilibrated fixed-volume QCCG critical-line recomputation; time-evolved confirmation remains separate",
        "evidence":[
            evidence(
                "qccg-equilibrated-critical-line-shift",
                "QCCG_EQUILIBRATED_CRITICAL_LINE",
                "PASS" if passed else "FAIL",
                "The curvature/volume zero-drift line is recomputed using curvature-equilibrated exact-volume geometry ensembles; the resulting potential derivative retains the CDT-inspired curvature-plus-volume shape.",
                kappa_R=KAPPA_R,
                old_kappa_V=old_kv,
                equilibrated_kappa_V=new_kv,
                kappa_V_shift=new_kv-old_kv,
                zero_drift_residual=res,
                potential_fit=fit,
                potential_r2_min=POTENTIAL_R2_MIN,
                thermalization=thermal_meta,
            ),
            evidence(
                "qccg-critical-line-seed-bias-diagnostic",
                "QCCG_CRITICAL_LINE_SEED_BIAS_DETECTED",
                "PASS" if shifted else "NOT_APPLICABLE",
                "The zero-drift volume coupling changes after conditional geometry thermalization, showing that the previous seed-geometry critical line was not yet an equilibrium coarse critical line.",
                old_kappa_V=old_kv,
                equilibrated_kappa_V=new_kv,
                shift=new_kv-old_kv,
            ),
            evidence(
                "qccg-equilibrated-critical-time-confirm-open",
                "QCCG_EQUILIBRATED_CRITICAL_TIME_CONFIRMATION",
                "OPEN",
                "A fresh thermalization/dynamics seed must confirm the original preregistered kinetic and potential criteria at the equilibrated critical-line coupling.",
                selected_kappa_R=KAPPA_R,
                selected_kappa_V=new_kv,
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("equilibrated critical-line audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
