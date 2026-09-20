#!/usr/bin/env python3
"""Preregistered confirmation of curvature-weighted QCCG critical dynamics.

Thresholds are inherited from pre-existing unweighted QCCG audits before this
confirmation sample is generated:
- variance exponent in [0.85,1.15];
- variance log-log R2 >= 0.96;
- Var/(N3*tau) volume spread <= 1.45;
- CDT-inspired potential-derivative fit R2 >= 0.80;
- positive mu and lambda;
- lambda block-to-block spread <= 1.35.

The curvature/volume candidate is selected only from the deterministic
pre-time critical-line calculation.  This script uses a fresh random seed and
does not tune thresholds to its output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_time_evolved_volume_kernel as tev
import run_qccg_curvature_critical_line as crit
import run_qccg_curvature_weighted_time_exploration as explore


TAUS=(0.025,0.05)
TRAJECTORIES=500
SEED=20270219
EXP_WINDOW=(0.85,1.15)
R2_MIN=0.96
D_SPREAD_MAX=1.45
POTENTIAL_R2_MIN=0.80
LAMBDA_SPREAD_MAX=1.35
MU_SPREAD_MAX=1.50


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-curvature-weighted-time-confirmation",
        "artifact":"qccg/run_qccg_curvature_weighted_time_confirmation.py",
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
        raise SystemExit("no preregistered curvature critical candidate")
    selected=max(candidates,key=lambda x:x["fit"]["potential_derivative_r2"])
    kR=selected["kappa_R"];kV=selected["kappa_V_critical"]

    starts={}
    for ti,n3 in enumerate(base.TARGETS):
        starts[n3]=base.build_target(n3,base.SEED+1000*ti)[0]

    blocks=[]
    all_blocks=True
    mus=[];lams=[]
    for tau in TAUS:
        rows=[]
        for ti,n3 in enumerate(base.TARGETS):
            vals=[
                explore.evolve_delta_weighted(
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
            })

        p,A,r2=explore.logfit(rows)
        dvals=[r["D_vol"] for r in rows]
        dspread=max(dvals)/min(dvals)
        pot=explore.fit_potential(rows)
        block_pass=(
            EXP_WINDOW[0]<=p<=EXP_WINDOW[1]
            and r2>=R2_MIN
            and dspread<=D_SPREAD_MAX
            and pot["potential_derivative_r2"]>=POTENTIAL_R2_MIN
            and pot["mu"]>0
            and pot["lambda"]>0
        )
        all_blocks=all_blocks and block_pass
        mus.append(pot["mu"]);lams.append(pot["lambda"])
        blocks.append({
            "tau":tau,
            "variance_exponent":p,
            "variance_amplitude":A,
            "variance_loglog_r2":r2,
            "D_vol_spread":dspread,
            "potential":pot,
            "block_pass":block_pass,
            "rows":rows,
        })

    mu_spread=max(mus)/min(mus) if min(mus)>0 else float("inf")
    lambda_spread=max(lams)/min(lams) if min(lams)>0 else float("inf")
    coefficient_stable=(mu_spread<=MU_SPREAD_MAX and lambda_spread<=LAMBDA_SPREAD_MAX)
    passed=all_blocks and coefficient_stable

    result={
        "schema":1,
        "scope":"fresh-seed preregistered time-evolved confirmation of the QCCG curvature critical point",
        "selected":selected,
        "thresholds":{
            "taus":list(TAUS),
            "trajectories":TRAJECTORIES,
            "seed":SEED,
            "variance_exponent_window":list(EXP_WINDOW),
            "variance_r2_min":R2_MIN,
            "D_vol_spread_max":D_SPREAD_MAX,
            "potential_r2_min":POTENTIAL_R2_MIN,
            "lambda_spread_max":LAMBDA_SPREAD_MAX,
            "mu_spread_max":MU_SPREAD_MAX,
        },
        "evidence":[
            evidence(
                "qccg-curvature-critical-time-confirmation",
                "QCCG_CURVATURE_CRITICAL_TIME_CONFIRMATION",
                "PASS" if passed else "FAIL",
                "A fresh-seed state-dependent curvature-weighted QCCG simulation tests the preselected zero-drift critical point against preregistered kinetic and CDT-potential stability thresholds.",
                selected_kappa_R=kR,
                selected_kappa_V=kV,
                rates=explore.weighted_rates(kR,kV),
                blocks=blocks,
                mu_values=mus,
                lambda_values=lams,
                mu_spread=mu_spread,
                lambda_spread=lambda_spread,
                thresholds={
                    "variance_exponent_window":list(EXP_WINDOW),
                    "variance_r2_min":R2_MIN,
                    "D_vol_spread_max":D_SPREAD_MAX,
                    "potential_r2_min":POTENTIAL_R2_MIN,
                    "lambda_spread_max":LAMBDA_SPREAD_MAX,
                    "mu_spread_max":MU_SPREAD_MAX,
                },
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("curvature-weighted critical dynamics confirmation failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
