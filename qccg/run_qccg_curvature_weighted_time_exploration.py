#!/usr/bin/env python3
"""Exploratory time evolution on the Regge-curvature QCCG critical line.

This script intentionally does NOT promote
QCCG_CURVATURE_CRITICAL_TIME_CONFIRMATION.  It uses an independent seed to
measure the weighted state-dependent dynamics at the candidate selected by
run_qccg_curvature_critical_line.py and records the sampling distribution needed
to preregister a later confirmation test.

The local rates are
  w = exp[-(kappa_R Delta R + kappa_V Delta N3)/2]
for each reversible 1<->4 and 2<->3 Pachner edge.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_time_evolved_volume_kernel as tev
import run_qccg_curvature_critical_line as crit
import run_qccg_curvature_weight_scan as curv
import run_time_local_slice_transfer_toy as qslice


TAUS=(0.025,0.05)
TRAJECTORIES=300
SEED=20270117


def weighted_rates(kappa_R,kappa_V):
    return {
        "14":math.exp(-0.5*(kappa_R*curv.DELTA_R_14+kappa_V*3)),
        "41":math.exp(-0.5*(-kappa_R*curv.DELTA_R_14-kappa_V*3)),
        "23":math.exp(-0.5*(kappa_R*curv.DELTA_R_23+kappa_V)),
        "32":math.exp(-0.5*(-kappa_R*curv.DELTA_R_23-kappa_V)),
    }


def evolve_delta_weighted(S0,tmax,seed,kappa_R,kappa_V):
    rng=random.Random(seed)
    S=set(S0)
    rates=weighted_rates(kappa_R,kappa_V)
    t=0.0
    while t<tmax:
        c14,c41,c23,c32=tev.event_lists(S)
        cs={"14":c14,"41":c41,"23":c23,"32":c32}
        weights={k:len(cs[k])*rates[k] for k in cs}
        total=sum(weights.values())
        if total<=0:
            break
        dt=rng.expovariate(total)
        if t+dt>tmax:
            break
        u=rng.random()*total
        acc=0.0; typ=None
        for k in ("14","41","23","32"):
            acc += weights[k]
            if u<=acc:
                typ=k;break
        if typ is None:
            typ="32"

        lst=cs[typ]
        if not lst:
            t += dt
            continue
        c=rng.choice(lst)
        if typ=="14":
            used={v for tet in S for v in tet}
            S=qslice.apply14(S,(c,max(used)+1))
        elif typ=="41":
            S=tev.apply41(S,c)
        elif typ=="23":
            S=qslice.apply23(S,c)
        else:
            S=qslice.apply32(S,c)
        t += dt
    return len(S)-len(S0)


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


def fit_potential(rows):
    Ns=[float(r["N3"]) for r in rows]
    avals=[r["variance_rate"] for r in rows]
    D0,D1,ar2,_=linear_fit(Ns,avals)
    x=[];up=[]
    for r in rows:
        N=float(r["N3"]);a=r["variance_rate"];b=r["drift_rate"]
        x.append(N**(-2.0/3.0))
        up.append((D0-2*b)/a)
    c,intercept,r2,pred=linear_fit(x,up)
    return {
        "diffusion_D0":D0,"diffusion_D1":D1,"diffusion_linear_r2":ar2,
        "mu":3*c,"lambda":-intercept,"potential_derivative_r2":r2,
        "rows":[{**r,"Uprime_inferred":u,"Uprime_fit":p}
                for r,u,p in zip(rows,up,pred)],
    }


def logfit(rows):
    xs=[math.log(r["N3"]) for r in rows]
    ys=[math.log(r["variance"]) for r in rows]
    xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    p=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    a=ym-p*xm
    pred=[a+p*x for x in xs]
    sse=sum((y-pr)**2 for y,pr in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>1e-18 else 1.0
    return p,math.exp(a),r2


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-curvature-weighted-time-exploration",
        "artifact":"qccg/run_qccg_curvature_weighted_time_exploration.py",
        "note":note,"metadata":metadata,
    }


def main():
    local_rows=crit.sample_rows()
    scans=[]
    for kR in crit.KAPPA_R_VALUES:
        kV,res=crit.solve_kappa_v(local_rows,kR)
        fit=crit.fit_potential(local_rows,kR,kV)
        scans.append({"kappa_R":kR,"kappa_V_critical":kV,"residual":res,"fit":fit})
    candidates=[
        x for x in scans
        if x["fit"]["potential_derivative_r2"]>=crit.R2_SELECTION_TARGET
        and x["fit"]["mu"]>0
    ]
    if not candidates:
        raise SystemExit("no critical-line candidate available for exploration")
    selected=max(candidates,key=lambda x:x["fit"]["potential_derivative_r2"])
    kR=selected["kappa_R"];kV=selected["kappa_V_critical"]

    starts={}
    for ti,n3 in enumerate(base.TARGETS):
        starts[n3]=base.build_target(n3,base.SEED+1000*ti)[0]

    blocks=[]
    for tau in TAUS:
        rows=[]
        for ti,n3 in enumerate(base.TARGETS):
            vals=[
                evolve_delta_weighted(
                    starts[n3],tau,
                    SEED+int(tau*100000)*100000+ti*10000+j,
                    kR,kV
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
                "D_vol":var/(n3*tau),
                "skewness":skew,
                "excess_kurtosis_diagnostic":excess,
                "min_delta":min(vals),"max_delta":max(vals),
            })
        p,A,r2=logfit(rows)
        pot=fit_potential(rows)
        blocks.append({
            "tau":tau,
            "variance_exponent":p,
            "variance_amplitude":A,
            "variance_loglog_r2":r2,
            "D_vol_spread":max(r["D_vol"] for r in rows)/min(r["D_vol"] for r in rows),
            "potential":pot,
            "rows":rows,
        })

    finite=all(
        math.isfinite(v)
        for b in blocks
        for v in (
            b["variance_exponent"],b["variance_loglog_r2"],
            b["D_vol_spread"],b["potential"]["mu"],
            b["potential"]["lambda"],b["potential"]["potential_derivative_r2"]
        )
    )

    result={
        "schema":1,
        "scope":"exploratory independent-seed curvature-weighted QCCG time evolution; no confirmation promotion",
        "selected_from_pre_time_scan":selected,
        "rates":weighted_rates(kR,kV),
        "evidence":[
            evidence(
                "qccg-curvature-weighted-time-exploration",
                "QCCG_CURVATURE_WEIGHTED_TIME_EXPLORATION",
                "PASS" if finite else "FAIL",
                "Independent-seed state-dependent curvature-weighted Pachner dynamics is simulated on the preregistered critical-line candidate to estimate kinetic and potential fit distributions before confirmation thresholds are frozen.",
                trajectories=TRAJECTORIES,
                taus=list(TAUS),
                seed=SEED,
                selected_kappa_R=kR,
                selected_kappa_V=kV,
                rates=weighted_rates(kR,kV),
                blocks=blocks,
            ),
            evidence(
                "qccg-curvature-time-confirm-still-open",
                "QCCG_CURVATURE_CRITICAL_TIME_CONFIRMATION",
                "OPEN",
                "This is exploratory data. Confirmation thresholds and a fresh seed must be frozen before promoting the weighted critical dynamics.",
                exploratory_blocks=blocks,
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not finite:
        raise SystemExit("nonfinite curvature-weighted exploratory dynamics")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
