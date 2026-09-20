#!/usr/bin/env python3
"""Fresh confirmation at the equilibrated QCCG curvature/volume critical line.

Selection:
- kappa_R is the preregistered curvature candidate -1;
- kappa_V is selected from a fixed-volume curvature-equilibrated geometry
  ensemble using the zero large-volume drift condition.

Confirmation:
- independent fixed-volume thermalization seed;
- independent time-evolution seed;
- unchanged kinetic/potential thresholds from both previous failed
  confirmations.

Thus a PASS cannot be obtained by loosening the earlier gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_time_evolved_volume_kernel as tev
import run_qccg_curvature_weighted_time_exploration as explore
import run_qccg_fixed_volume_curvature_thermalization as therm
import run_qccg_equilibrated_critical_line as eqcrit


TAUS=(0.025,0.05)
TRAJECTORIES=500
CONFIRM_THERM_SEED=20270603
CONFIRM_DYN_SEED=20270617

EXP_WINDOW=(0.85,1.15)
R2_MIN=0.96
D_SPREAD_MAX=1.45
POTENTIAL_R2_MIN=0.80
LAMBDA_SPREAD_MAX=1.35
MU_SPREAD_MAX=1.50


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-equilibrated-critical-time-confirmation",
        "artifact":"qccg/run_qccg_equilibrated_critical_confirmation.py",
        "note":note,"metadata":metadata,
    }


def main():
    # Selection ensemble and corrected critical line.
    selection_samples,selection_meta,old_kv=eqcrit.thermal_samples()
    kV,res=eqcrit.solve_kv(selection_samples)
    selection_fit=eqcrit.fit_potential(selection_samples,kV)
    kR=eqcrit.KAPPA_R

    # Fresh conditional geometry ensemble for confirmation.
    pools={}
    confirm_thermal=[]
    for ti,N in enumerate(base.TARGETS):
        out=therm.thermalize_target(
            N,kR,kV,CONFIRM_THERM_SEED+10000*ti
        )
        if len(out["samples"])!=therm.SAMPLES_PER_TARGET:
            raise SystemExit(f"confirmation thermalization failed at N3={N}")
        pools[N]=out["samples"]
        confirm_thermal.append({
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
            pool=pools[N]
            vals=[]
            for j in range(TRAJECTORIES):
                S0=pool[j%len(pool)]
                vals.append(
                    explore.evolve_delta_weighted(
                        S0,tau,
                        CONFIRM_DYN_SEED+int(tau*100000)*100000+ti*10000+j,
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
    coeff_stable=(mu_spread<=MU_SPREAD_MAX and lambda_spread<=LAMBDA_SPREAD_MAX)
    passed=all_blocks and coeff_stable

    result={
        "schema":1,
        "scope":"fresh-seed confirmation at equilibrium-recomputed QCCG curvature/volume critical line",
        "selection":{
            "kappa_R":kR,
            "old_seed_kappa_V":old_kv,
            "equilibrated_kappa_V":kV,
            "zero_drift_residual":res,
            "selection_potential_fit":selection_fit,
            "selection_thermalization":selection_meta,
        },
        "confirmation":{
            "thermalization_seed":CONFIRM_THERM_SEED,
            "dynamics_seed":CONFIRM_DYN_SEED,
            "trajectories":TRAJECTORIES,
            "taus":list(TAUS),
            "thermalization":confirm_thermal,
        },
        "thresholds":{
            "variance_exponent_window":list(EXP_WINDOW),
            "variance_r2_min":R2_MIN,
            "D_vol_spread_max":D_SPREAD_MAX,
            "potential_r2_min":POTENTIAL_R2_MIN,
            "mu_spread_max":MU_SPREAD_MAX,
            "lambda_spread_max":LAMBDA_SPREAD_MAX,
        },
        "evidence":[
            evidence(
                "qccg-equilibrated-critical-time-confirmation",
                "QCCG_EQUILIBRATED_CRITICAL_TIME_CONFIRMATION",
                "PASS" if passed else "FAIL",
                "A fresh conditional-geometry ensemble and fresh dynamics seed re-test the unchanged CDT kinetic/potential criteria at the equilibrium-recomputed zero-drift volume coupling.",
                kappa_R=kR,
                kappa_V=kV,
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
                    "mu_spread_max":MU_SPREAD_MAX,
                    "lambda_spread_max":LAMBDA_SPREAD_MAX,
                },
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("equilibrated critical-line confirmation failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
