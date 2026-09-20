#!/usr/bin/env python3
"""Large-volume Pachner move-count scaling for the QCCG -> CDT bridge.

We generate closed S^3 triangulations at four increasing N3 values by 1->4
refinement of the boundary of a 4-simplex.  At fixed N3 we decorrelate the
triangulation by repeated 2->3 followed by 3->2 moves (net Delta N3=0).

For the equal-local-rate graph-Laplacian parent, the instantaneous second
volume-jump moment is

    D2 = 9 (C_14 + C_41) + (C_23 + C_32),

where C_ab is the number of physical local Pachner moves of that type and
Delta N3 = (+3,-3,+1,-1).

The CDT kinetic mechanism requires D2 to be extensive, D2 ~ N3^p with p=1,
so that a diffusive Gaussian coarse limit has exponent
(Delta N3)^2 / O(N3).

This is a geometry/move-count scaling test, not yet a measured long-time
Gaussian transfer kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import run_time_local_slice_transfer_toy as qslice


TARGETS = (20, 41, 80, 119)
SAMPLES_PER_TARGET = 6
MIX_CYCLES = 35
SEED = 20260921
P_MIN = 0.85
P_MAX = 1.15
R2_MIN = 0.985


def fresh_refine(S, rng):
    tet = rng.choice(sorted(S))
    used = {v for t in S for v in t}
    v = max(used) + 1
    return qslice.apply14(S, (tet, v))


def build_target(n3, seed):
    if (n3 - 5) % 3 != 0 or n3 < 5:
        raise ValueError("target must be 5 mod 3")
    rng = random.Random(seed)
    S = qslice.spatial_s3()
    while len(S) < n3:
        S = fresh_refine(S, rng)
        if not qslice.manifold(S):
            raise RuntimeError("1->4 refinement broke manifold")

    # Geometry-mixing pairs with zero net N3 change.
    accepted_pairs = 0
    for _ in range(MIX_CYCLES):
        c23 = qslice.candidates23(S)
        if not c23:
            continue
        S1 = qslice.apply23(S, rng.choice(c23))
        c32 = qslice.candidates32(S1)
        if not c32:
            continue
        S2 = qslice.apply32(S1, rng.choice(c32))
        if len(S2) == n3 and qslice.manifold(S2):
            S = S2
            accepted_pairs += 1
    return S, accepted_pairs


def counts(S):
    return {
        "14": len(S),  # one physical insertion site per tetrahedron
        "41": len(qslice.candidates41(S)),
        "23": len(qslice.candidates23(S)),
        "32": len(qslice.candidates32(S)),
    }


def moment(c):
    return 9 * (c["14"] + c["41"]) + c["23"] + c["32"]


def linear_fit(xs, ys):
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    sxx = sum((x-xm)**2 for x in xs)
    sxy = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
    slope = sxy/sxx
    intercept = ym-slope*xm
    pred = [intercept+slope*x for x in xs]
    sse = sum((y-p)**2 for y,p in zip(ys,pred))
    sst = sum((y-ym)**2 for y in ys)
    r2 = 1-sse/sst if sst>0 else 1.0
    return slope,intercept,r2


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-large-volume-pachner-scaling",
        "artifact":"qccg/run_qccg_large_volume_diffusion_scan.py",
        "note":note,"metadata":metadata,
    }


def main():
    rows=[]
    aggregate=[]
    all_manifold=True
    for ti,n3 in enumerate(TARGETS):
        vals=[]
        for s in range(SAMPLES_PER_TARGET):
            S,mixed=build_target(n3,SEED+1000*ti+s)
            all_manifold=all_manifold and qslice.manifold(S) and len(S)==n3
            c=counts(S)
            d2=moment(c)
            vals.append(d2)
            rows.append({
                "N3":n3,"sample":s,"counts":c,
                "D2":d2,"D2_over_N3":d2/n3,
                "mix_pairs_accepted":mixed,
            })
        mean=sum(vals)/len(vals)
        var=sum((x-mean)**2 for x in vals)/len(vals)
        aggregate.append({
            "N3":n3,
            "mean_D2":mean,
            "std_D2":math.sqrt(var),
            "mean_D2_over_N3":mean/n3,
        })

    x=[math.log(r["N3"]) for r in aggregate]
    y=[math.log(r["mean_D2"]) for r in aggregate]
    p,logA,r2=linear_fit(x,y)
    extensive=(P_MIN<=p<=P_MAX and r2>=R2_MIN)
    ratios=[r["mean_D2_over_N3"] for r in aggregate]
    ratio_spread=max(ratios)/min(ratios)

    result={
        "schema":1,
        "scope":"increasing closed-S3 Pachner move-count scaling; not long-time QCCG transfer dynamics",
        "evidence":[
            evidence(
                "qccg-large-volume-pachner-scan",
                "QCCG_LARGE_VOLUME_PACHNER_SCAN",
                "PASS" if all_manifold else "FAIL",
                "Closed S3 triangulations are generated and geometry-mixed at four increasing spatial volumes while preserving manifold incidence.",
                targets=list(TARGETS),
                samples_per_target=SAMPLES_PER_TARGET,
                mix_cycles=MIX_CYCLES,
                rows=rows,
            ),
            evidence(
                "qccg-large-volume-diffusion-exponent",
                "QCCG_LARGE_VOLUME_DIFFUSION_EXPONENT",
                "PASS" if extensive else "FAIL",
                "The equal-local-rate second Pachner volume-jump moment is fitted to D2=A*N3^p over the preregistered increasing-volume scan.",
                exponent_p=p,
                amplitude_A=math.exp(logA),
                loglog_r2=r2,
                exponent_window=[P_MIN,P_MAX],
                r2_min=R2_MIN,
                D2_over_N3_spread=ratio_spread,
                aggregate=aggregate,
            ),
            evidence(
                "qccg-long-time-gaussian-volume-kernel-open",
                "QCCG_CDT_DIFFUSIVE_KINETIC_LIMIT",
                "OPEN",
                "Extensive instantaneous move-count scaling does not by itself establish Gaussian long-time volume transfer. Time-blocked transition distributions must still be measured on the large causal ensemble.",
            ),
        ],
    }
    pth=Path(args.out);pth.parent.mkdir(parents=True,exist_ok=True)
    pth.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"][:2]):
        raise SystemExit("large-volume QCCG diffusion scaling audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
