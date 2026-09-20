#!/usr/bin/env python3
"""QCCG CDT kinetic-coefficient stability audit.

Uses the same state-dependent continuous-time Pachner process as
run_qccg_time_evolved_volume_kernel.py.

Near n=m=N, the CDT transfer law

    M_nm ~ exp[- (n-m)^2 / (Gamma (n+m))]

is a Gaussian in Delta n with

    Var(Delta n | N) ~ Gamma_block * N.

For a continuous-time local process observed over a block tau,

    Var(Delta n | N) ~ D_vol * N * tau,

so Gamma_block/tau = D_vol.  A necessary precursor to a stable CDT kinetic
coefficient is therefore volume-independence of Var/(N tau) and approximate
time-block stability of its large-volume extrapolation.

This audit measures that quantity directly from QCCG dynamics.  It does not
identify the microscopic QCCG time unit with the CDT lattice proper-time unit;
that scale map remains a separate universality obligation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_time_evolved_volume_kernel as tev


TAUS = (0.025, 0.05, 0.075)
TRAJECTORIES = 350
MAX_VOLUME_SPREAD = 1.25
MAX_BLOCK_MEAN_SPREAD = 1.20
EXP_WINDOW = (0.85, 1.15)
R2_MIN = 0.96
SEED = 20260921


SOURCES = [
    {
        "title": "The effective action in 4-dim CDT. The transfer matrix approach",
        "journal": "JHEP 06 (2014) 034",
        "arxiv": "1403.5940",
        "doi": "10.1007/JHEP06(2014)034",
    },
    {
        "title": "Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group",
        "journal": "Phys. Rev. D 110, 126006 (2024)",
        "doi": "10.1103/PhysRevD.110.126006",
    },
]


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
        "engine":"qccg-cdt-kinetic-coefficient-stability",
        "artifact":"qccg/run_qccg_cdt_kinetic_coefficient_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    start_states={}
    for ti,n3 in enumerate(base.TARGETS):
        start_states[n3]=base.build_target(n3,base.SEED+1000*ti)[0]

    blocks=[]
    block_means=[]
    all_volume_stable=True
    all_exponent_stable=True

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
            dvol=var/(n3*tau)
            rows.append({
                "N3":n3,
                "mean_delta":mean,
                "variance":var,
                "D_vol_var_over_N3_tau":dvol,
                "Gamma_block_var_over_N3":var/n3,
                "skewness":skew,
                "excess_kurtosis_diagnostic":excess,
            })

        p,A,r2=logfit(rows)
        dvals=[r["D_vol_var_over_N3_tau"] for r in rows]
        dmean=sum(dvals)/len(dvals)
        spread=max(dvals)/min(dvals)
        block_means.append(dmean)
        volume_stable=spread<=MAX_VOLUME_SPREAD
        exponent_stable=EXP_WINDOW[0]<=p<=EXP_WINDOW[1] and r2>=R2_MIN
        all_volume_stable=all_volume_stable and volume_stable
        all_exponent_stable=all_exponent_stable and exponent_stable
        blocks.append({
            "tau":tau,
            "variance_exponent":p,
            "variance_amplitude":A,
            "variance_loglog_r2":r2,
            "D_vol_mean":dmean,
            "D_vol_volume_spread":spread,
            "volume_stable":volume_stable,
            "exponent_stable":exponent_stable,
            "rows":rows,
        })

    mean_spread=max(block_means)/min(block_means)
    block_stable=mean_spread<=MAX_BLOCK_MEAN_SPREAD
    passed=all_volume_stable and all_exponent_stable and block_stable

    # Block Gamma must scale approximately linearly with tau if D_vol is stable.
    gamma_rows=[
        {
            "tau":b["tau"],
            "Gamma_block_mean":b["D_vol_mean"]*b["tau"],
            "Gamma_block_over_tau":b["D_vol_mean"],
        }
        for b in blocks
    ]

    result={
        "schema":1,
        "scope":"QCCG kinetic coefficient per microscopic time unit; CDT time-unit matching remains open",
        "sources":SOURCES,
        "evidence":[
            evidence(
                "qccg-cdt-kinetic-coefficient-stability",
                "QCCG_CDT_KINETIC_COEFFICIENT_STABILITY",
                "PASS" if passed else "FAIL",
                "The state-dependent QCCG Pachner process has Var(Delta N3)/(N3*tau) approximately independent of volume and stable across the audited time blocks, providing a measured kinetic coefficient per microscopic time unit.",
                taus=list(TAUS),
                trajectories=TRAJECTORIES,
                targets=list(base.TARGETS),
                max_volume_spread=MAX_VOLUME_SPREAD,
                max_block_mean_spread=MAX_BLOCK_MEAN_SPREAD,
                block_mean_spread=mean_spread,
                blocks=blocks,
                gamma_rows=gamma_rows,
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-time-unit-map-open",
                "QCCG_CDT_TIME_UNIT_MAP",
                "OPEN",
                "The kinetic coefficient is measured per QCCG microscopic time unit. A justified map to the CDT lattice proper-time unit or an equivalent dimensionless observable is still required before identifying the numerical Gamma values directly.",
                block_means=block_means,
                sources=SOURCES,
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("QCCG CDT kinetic coefficient stability audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
