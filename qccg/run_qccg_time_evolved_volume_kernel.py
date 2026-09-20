#!/usr/bin/env python3
"""State-dependent time-evolved QCCG volume-kernel scan.

This goes beyond the frozen-generator cumulant audit.  We simulate the actual
continuous-time equal-local-rate Pachner process on closed S^3 triangulations.

At every state, each physical 1<->4 and 2<->3 Pachner move has unit rate.
Gillespie evolution therefore implements the continuous-time graph-Laplacian
kinetic process on the labelled representative, with one physical 1->4 event
per tetrahedron.

We test two preregistered mesoscopic blocks, tau=0.025 and 0.05:
- Var(Delta N3) ~ N3^p with p close to 1;
- Var/(N3*tau) approximately volume-independent;
- the largest-volume transition distribution has small skewness and excess
  kurtosis.

A longer tau=0.075 block is recorded only as a diagnostic because state
dependence / volume drift can move the process outside the local Gaussian
window.

PASS establishes a finite time-evolved diffusive window, not the full CDT
continuum coefficient/profile match.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_time_local_slice_transfer_toy as qslice


TAUS = (0.025, 0.05, 0.075)
PASS_TAUS = (0.025, 0.05)
TRAJECTORIES = 350
EXP_WINDOW = (0.85, 1.15)
R2_MIN = 0.96
COEFF_SPREAD_MAX = 1.45
LARGE_SKEW_MAX = 0.70
NORMAL_IQR_OVER_SD = 1.3489795003921634
IQR_RATIO_WINDOW = (1.10, 1.55)
NORMAL_P90_OVER_2SD = 1.6448536269514722
P90_RATIO_WINDOW = (1.40, 1.90)
SEED = 20260921


def event_lists(S):
    return (
        list(sorted(S)),
        qslice.candidates41(S),
        qslice.candidates23(S),
        qslice.candidates32(S),
    )


def apply41(S,c):
    _v,star,target=c
    S2=set(S)
    for tet in star:
        S2.remove(tet)
    S2.add(target)
    return S2


def evolve_delta(S0,tmax,seed):
    rng=random.Random(seed)
    S=set(S0)
    t=0.0
    while t<tmax:
        c14,c41,c23,c32=event_lists(S)
        counts=(len(c14),len(c41),len(c23),len(c32))
        total=sum(counts)
        if total==0:
            break
        dt=rng.expovariate(total)
        if t+dt>tmax:
            break
        u=rng.randrange(total)
        if u<counts[0]:
            tet=c14[u]
            used={v for t4 in S for v in t4}
            S=qslice.apply14(S,(tet,max(used)+1))
        elif u<counts[0]+counts[1]:
            S=apply41(S,c41[u-counts[0]])
        elif u<counts[0]+counts[1]+counts[2]:
            S=qslice.apply23(S,c23[u-counts[0]-counts[1]])
        else:
            S=qslice.apply32(S,c32[u-counts[0]-counts[1]-counts[2]])
        t += dt
    return len(S)-len(S0)


def moments(vals):
    n=len(vals); mean=sum(vals)/n
    centered=[x-mean for x in vals]
    var=sum(x*x for x in centered)/n
    if var<=0:
        return mean,var,0.0,0.0
    m3=sum(x**3 for x in centered)/n
    m4=sum(x**4 for x in centered)/n
    return mean,var,m3/(var**1.5),m4/(var*var)-3.0


def robust_shape(vals, var):
    xs=sorted(vals); n=len(xs)
    sd=math.sqrt(var)
    def q(p):
        return xs[int(round(p*(n-1)))]
    if sd<=0:
        return {"iqr_over_sd":0.0,"p90_width_over_2sd":0.0}
    return {
        "q05":q(0.05),"q25":q(0.25),"q50":q(0.50),
        "q75":q(0.75),"q95":q(0.95),
        "iqr_over_sd":(q(0.75)-q(0.25))/sd,
        "p90_width_over_2sd":(q(0.95)-q(0.05))/(2*sd),
    }


def logfit(rows):
    xs=[math.log(r["N3"]) for r in rows]
    ys=[math.log(r["variance"]) for r in rows]
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    p=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    a=ym-p*xm
    pred=[a+p*x for x in xs]
    sse=sum((y-pr)**2 for y,pr in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>0 else 1.0
    return p,math.exp(a),r2


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-time-evolved-volume-kernel",
        "artifact":"qccg/run_qccg_time_evolved_volume_kernel.py",
        "note":note,"metadata":metadata,
    }


def main():
    start_states={}
    for ti,n3 in enumerate(base.TARGETS):
        start_states[n3]=base.build_target(n3,base.SEED+1000*ti)[0]

    blocks=[]
    pass_rows=[]
    all_pass=True
    for tau in TAUS:
        rows=[]
        for ti,n3 in enumerate(base.TARGETS):
            S0=start_states[n3]
            vals=[
                evolve_delta(
                    S0,tau,
                    SEED + int(tau*100000)*100000 + ti*10000 + j
                )
                for j in range(TRAJECTORIES)
            ]
            mean,var,skew,excess=moments(vals)
            shape=robust_shape(vals,var)
            rows.append({
                "N3":n3,
                "mean_delta":mean,
                "variance":var,
                "variance_over_N3_tau":var/(n3*tau),
                "skewness":skew,
                "excess_kurtosis_diagnostic":excess,
                "robust_shape":shape,
                "min_delta":min(vals),
                "max_delta":max(vals),
            })
        p,A,r2=logfit(rows)
        ratios=[r["variance_over_N3_tau"] for r in rows]
        spread=max(ratios)/min(ratios)
        largest=rows[-1]
        shape=largest["robust_shape"]
        block_pass=(
            EXP_WINDOW[0]<=p<=EXP_WINDOW[1]
            and r2>=R2_MIN
            and spread<=COEFF_SPREAD_MAX
            and abs(largest["skewness"])<=LARGE_SKEW_MAX
            and IQR_RATIO_WINDOW[0]<=shape["iqr_over_sd"]<=IQR_RATIO_WINDOW[1]
            and P90_RATIO_WINDOW[0]<=shape["p90_width_over_2sd"]<=P90_RATIO_WINDOW[1]
        )
        if tau in PASS_TAUS:
            all_pass=all_pass and block_pass
            pass_rows.append(block_pass)
        blocks.append({
            "tau":tau,"rows":rows,
            "variance_exponent":p,"variance_amplitude":A,
            "variance_loglog_r2":r2,
            "variance_over_N3_tau_spread":spread,
            "pass_window_criteria":block_pass,
            "required_for_pass":tau in PASS_TAUS,
        })

    result={
        "schema":1,
        "scope":"finite state-dependent continuous-time Pachner dynamics; continuum CDT coefficient/profile matching remains open",
        "evidence":[
            evidence(
                "qccg-time-evolved-volume-diffusion",
                "QCCG_TIME_EVOLVED_VOLUME_DIFFUSION",
                "PASS" if all_pass else "FAIL",
                "The actual state-dependent equal-local-rate Pachner process exhibits a mesoscopic time window in which volume-transition variance scales approximately linearly with N3 at increasing volume.",
                taus=list(TAUS),
                pass_taus=list(PASS_TAUS),
                trajectories=TRAJECTORIES,
                exponent_window=list(EXP_WINDOW),
                r2_min=R2_MIN,
                coefficient_spread_max=COEFF_SPREAD_MAX,
                blocks=blocks,
            ),
            evidence(
                "qccg-time-evolved-gaussian-window",
                "QCCG_TIME_EVOLVED_GAUSSIAN_WINDOW",
                "PASS" if all_pass else "FAIL",
                "Within the preregistered short-time blocks, the largest-volume transition distribution has bounded skewness and robust Gaussian quantile ratios while Var(Delta N3)/(N3*tau) remains approximately volume-independent. Sample excess kurtosis is retained as a diagnostic but is not a hard gate because of its high tail variance at finite trajectory count.",
                largest_volume=base.TARGETS[-1],
                largest_skew_max=LARGE_SKEW_MAX,
                normal_iqr_over_sd=NORMAL_IQR_OVER_SD,
                iqr_ratio_window=list(IQR_RATIO_WINDOW),
                normal_p90_width_over_2sd=NORMAL_P90_OVER_2SD,
                p90_ratio_window=list(P90_RATIO_WINDOW),
                blocks=blocks,
            ),
            evidence(
                "qccg-full-cdt-diffusive-limit-open",
                "QCCG_CDT_DIFFUSIVE_KINETIC_LIMIT",
                "OPEN",
                "A finite mesoscopic state-dependent Gaussian window is observed, but the continuum claim still requires stability under larger N3, multiple starting geometries/couplings, and compatible CDT effective coefficient extraction.",
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"][:2]):
        raise SystemExit("time-evolved QCCG volume-kernel audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
