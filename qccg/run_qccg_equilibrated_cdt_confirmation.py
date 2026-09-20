#!/usr/bin/env python3
"""Fresh-seed CDT coefficient confirmation from fixed-volume equilibrated QCCG slices.

This is the repair test for the failed curvature-weighted confirmation.

The previous fresh-seed run kept the CDT functional form at each time block but
failed cross-block mu/lambda stability.  The diagnosed confounder was that each
N3 started from one non-equilibrated internal geometry.  Here, each trajectory
starts from an independently selected snapshot of the curvature-weighted
conditional geometry ensemble at the same N3.

The confirmation thresholds are unchanged:
- variance exponent in [0.85,1.15];
- variance log-log R2 >= 0.96;
- D_vol spread <= 1.45;
- potential derivative R2 >= 0.80;
- mu > 0, lambda > 0;
- mu block spread <= 1.50;
- lambda block spread <= 1.35.

No threshold is relaxed relative to the failed confirmation.
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
import run_qccg_curvature_weighted_time_exploration as explore
import run_qccg_fixed_volume_curvature_thermalization as therm


TAUS=(0.025,0.05)
TRAJECTORIES=500
THERM_SEED=20270407
DYN_SEED=20270419

EXP_WINDOW=(0.85,1.15)
R2_MIN=0.96
D_SPREAD_MAX=1.45
POTENTIAL_R2_MIN=0.80
LAMBDA_SPREAD_MAX=1.35
MU_SPREAD_MAX=1.50


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-equilibrated-cdt-confirmation",
        "artifact":"qccg/run_qccg_equilibrated_cdt_confirmation.py",
        "note":note,"metadata":metadata,
    }


def main():
    pre_rows=crit.sample_rows()
    scans=[]
    for kR in crit.KAPPA_R_VALUES:
        kV,res=crit.solve_kappa_v(pre_rows,kR)
        fit=crit.fit_potential(pre_rows,kR,kV)
        scans.append({"kappa_R":kR,"kappa_V_critical":kV,"residual":res,"fit":fit})
    candidates=[
        x for x in scans
        if x["fit"]["potential_derivative_r2"]>=crit.R2_SELECTION_TARGET
        and x["fit"]["mu"]>0
    ]
    if not candidates:
        raise SystemExit("no preregistered critical candidate")
    selected=max(candidates,key=lambda x:x["fit"]["potential_derivative_r2"])
    kR=selected["kappa_R"];kV=selected["kappa_V_critical"]

    equilibrated={}
    thermal_rows=[]
    for ti,N in enumerate(base.TARGETS):
        out=therm.thermalize_target(N,kR,kV,THERM_SEED+10000*ti)
        if len(out["samples"]) != therm.SAMPLES_PER_TARGET:
            raise SystemExit(f"thermalization failed to collect target samples at N3={N}")
        equilibrated[N]=out["samples"]
        thermal_rows.append({
            "N3":N,
            "acceptance_fraction":out["acceptance_fraction"],
            "n_samples":len(out["samples"]),
            "sample_stats":out["sample_stats"],
        })

    blocks=[]
    mus=[];lams=[]
    all_blocks=True

    for tau in TAUS:
        rows=[]
        for ti,N in enumerate(base.TARGETS):
            pool=equilibrated[N]
            vals=[]
            for j in range(TRAJECTORIES):
                # Deterministic cycling through equilibrium snapshots with
                # trajectory-specific stochastic evolution.
                S0=pool[j % len(pool)]
                vals.append(
                    explore.evolve_delta_weighted(
                        S0,tau,
                        DYN_SEED+int(tau*100000)*100000+ti*10000+j,
                        kR,kV
                    )
                )
            mean,var,skew,excess=tev.moments(vals)
            rows.append({
                "N3":N,
                "mean_delta":mean,
                "variance":var,
                "drift_rate":mean/tau,
                "variance_rate":var/tau,
                "D_vol":var/(N*tau),
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
    coefficient_stable=mu_spread<=MU_SPREAD_MAX and lambda_spread<=LAMBDA_SPREAD_MAX
    passed=all_blocks and coefficient_stable

    result={
        "schema":1,
        "scope":"fresh-seed curvature-weighted QCCG confirmation initialized from fixed-volume equilibrium geometries",
        "selected":selected,
        "thermalization":thermal_rows,
        "thresholds":{
            "taus":list(TAUS),
            "trajectories":TRAJECTORIES,
            "thermalization_seed":THERM_SEED,
            "dynamics_seed":DYN_SEED,
            "variance_exponent_window":list(EXP_WINDOW),
            "variance_r2_min":R2_MIN,
            "D_vol_spread_max":D_SPREAD_MAX,
            "potential_r2_min":POTENTIAL_R2_MIN,
            "lambda_spread_max":LAMBDA_SPREAD_MAX,
            "mu_spread_max":MU_SPREAD_MAX,
        },
        "evidence":[
            evidence(
                "qccg-equilibrated-cdt-coefficient-confirmation",
                "QCCG_EQUILIBRATED_CDT_COEFFICIENT_CONFIRMATION",
                "PASS" if passed else "FAIL",
                "A fresh-seed weighted QCCG transfer test starts every trajectory from the fixed-volume curvature-equilibrated conditional ensemble and reuses the original preregistered kinetic/potential and cross-block coefficient-stability thresholds.",
                selected_kappa_R=kR,
                selected_kappa_V=kV,
                blocks=blocks,
                mu_values=mus,
                lambda_values=lams,
                mu_spread=mu_spread,
                lambda_spread=lambda_spread,
                thermalization=thermal_rows,
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("equilibrated CDT coefficient confirmation failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
