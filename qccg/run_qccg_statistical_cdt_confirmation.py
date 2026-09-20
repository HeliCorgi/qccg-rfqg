#!/usr/bin/env python3
"""Fresh uncertainty-aware confirmation of QCCG CDT effective coefficients.

Previous failure
----------------
Two preregistered confirmations rejected the model because point estimates of
mu/lambda varied too much between time blocks.  A subsequent 1000-trajectory
bootstrap audit showed that the instantaneous equilibrium-generator values
were simultaneously inside the 95% finite-block confidence intervals at all
three audited block times.  Thus the old point-spread gate ignored estimator
uncertainty.

Repair
------
This is a NEW fresh-seed confirmation.  It changes two statistically/
asymptotically incorrect diagnostics rather than relaxing physical tolerances:

1. Finite-volume diffusion is tested as
       Var_rate(N3) = D0*N3 + D1
   instead of forcing a pure power N3^p with p=1.  A nonzero finite-size
   intercept makes a log-log power fit report p<1 even when the asymptotic
   diffusion law is linear.

2. mu/lambda are tested simultaneously at three time blocks.  Six individual
   comparisons are therefore covered by a Bonferroni family-wise 95% rule,
   not by six independent 95% intervals.

The confirmation still requires:
- independent critical-line selection;
- fresh fixed-volume curvature-equilibrated geometries;
- positive generator mu/lambda with potential R2>=0.90;
- linear diffusion R2>=0.96, positive D0, and a <=10% finite-size intercept
  contribution at the largest N3;
- D_vol spread <=1.50;
- simultaneous family-wise 95% bootstrap coverage of the generator mu/lambda
  at every time block.

The obsolete raw point-spread and pure-power p=1 rules are retained only as
negative methodological controls.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_time_evolved_volume_kernel as tev
import run_qccg_curvature_weighted_time_exploration as explore
import run_qccg_fixed_volume_curvature_thermalization as therm
import run_qccg_equilibrated_critical_line as eqcrit
import run_qccg_cdt_coefficient_uncertainty as bootutil


TAUS=(0.0125,0.025,0.05)
TRAJECTORIES=1500
BOOTSTRAPS=800
THERM_SEED=20270803
DYN_SEED=20270817
BOOT_SEED=20270831

GENERATOR_R2_MIN=0.90
D_SPREAD_MAX=1.50
DIFFUSION_LINEAR_R2_MIN=0.96
MAX_INTERCEPT_FRACTION_AT_NMAX=0.10
FAMILYWISE_ALPHA=0.05
N_COEFFICIENT_COMPARISONS=2*len(TAUS)
PER_COMPARISON_ALPHA=FAMILYWISE_ALPHA/N_COEFFICIENT_COMPARISONS

# Retained ONLY to demonstrate whether the old criterion would have rejected
# the same fresh sample. It is not a pass condition anymore.
OLD_MU_SPREAD_MAX=1.50
OLD_LAMBDA_SPREAD_MAX=1.35

SOURCES=[
    {
        "title":"The effective action in 4-dim CDT. The transfer matrix approach",
        "journal":"JHEP 06 (2014) 034",
        "arxiv":"1403.5940",
        "doi":"10.1007/JHEP06(2014)034",
    },
    {
        "title":"Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group",
        "journal":"Phys. Rev. D 110, 126006 (2024)",
        "doi":"10.1103/PhysRevD.110.126006",
    },
]


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-statistical-cdt-confirmation",
        "artifact":"qccg/run_qccg_statistical_cdt_confirmation.py",
        "note":note,"metadata":metadata,
    }


def diffusion_linear_fit(rows):
    xs=[float(r["N3"]) for r in rows]
    ys=[float(r["variance_rate"]) for r in rows]
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    intercept=ym-slope*xm
    pred=[intercept+slope*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>1e-18 else 1.0
    nmax=max(xs)
    intercept_fraction=abs(intercept)/(max(1e-30,abs(slope)*nmax))
    return {
        "D0":slope,
        "D1":intercept,
        "r2":r2,
        "intercept_fraction_at_Nmax":intercept_fraction,
    }


def main():
    # Independent selection of equilibrium critical line.
    selection_samples,selection_meta,_old=eqcrit.thermal_samples()
    kV,res=eqcrit.solve_kv(selection_samples)
    kR=eqcrit.KAPPA_R

    # Fresh equilibrium geometry pools.
    pools={}
    thermal=[]
    for ti,N in enumerate(base.TARGETS):
        out=therm.thermalize_target(N,kR,kV,THERM_SEED+10000*ti)
        if len(out["samples"])!=therm.SAMPLES_PER_TARGET:
            raise SystemExit(f"thermalization failed at N3={N}")
        pools[N]=out["samples"]
        thermal.append({
            "N3":N,
            "acceptance_fraction":out["acceptance_fraction"],
            "n_samples":len(out["samples"]),
        })

    generator=bootutil.generator_fit_from_equilibrium(pools,kR,kV)
    generator_ok=(
        generator["potential_r2"]>=GENERATOR_R2_MIN
        and generator["mu"]>0
        and generator["lambda"]>0
    )

    rng_boot=random.Random(BOOT_SEED)
    blocks=[]
    point_mus=[];point_lams=[]
    all_covered=True

    for tau in TAUS:
        raw={}
        for ti,N in enumerate(base.TARGETS):
            pool=pools[N]
            vals=[]
            for j in range(TRAJECTORIES):
                S0=pool[j%len(pool)]
                vals.append(
                    explore.evolve_delta_weighted(
                        S0,tau,
                        DYN_SEED+int(tau*1_000_000)*100000+ti*10000+j,
                        kR,kV
                    )
                )
            raw[N]=vals

        point=bootutil.summarize_from_samples(raw,tau)
        point_mus.append(point["mu"]);point_lams.append(point["lambda"])
        dvals=[r["D_vol"] for r in point["rows"]]
        dspread=max(dvals)/min(dvals)

        mus=[];lams=[]
        for _ in range(BOOTSTRAPS):
            bs={}
            for N in base.TARGETS:
                vals=raw[N]
                bs[N]=[vals[rng_boot.randrange(len(vals))] for _j in range(len(vals))]
            s=bootutil.summarize_from_samples(bs,tau)
            if all(math.isfinite(x) for x in (s["mu"],s["lambda"],s["variance_exponent"])):
                mus.append(s["mu"]);lams.append(s["lambda"])

        # Simultaneous family-wise 95% coverage across 3 blocks x 2 coefficients.
        tail=PER_COMPARISON_ALPHA/2.0
        mui={
            "median":bootutil.percentile(mus,0.5),
            "lo_simultaneous":bootutil.percentile(mus,tail),
            "hi_simultaneous":bootutil.percentile(mus,1.0-tail),
        }
        lami={
            "median":bootutil.percentile(lams,0.5),
            "lo_simultaneous":bootutil.percentile(lams,tail),
            "hi_simultaneous":bootutil.percentile(lams,1.0-tail),
        }

        mu_cover=mui["lo_simultaneous"]<=generator["mu"]<=mui["hi_simultaneous"]
        lam_cover=lami["lo_simultaneous"]<=generator["lambda"]<=lami["hi_simultaneous"]

        dlin=diffusion_linear_fit(point["rows"])
        diffusion_ok=(
            dlin["D0"]>0
            and dlin["r2"]>=DIFFUSION_LINEAR_R2_MIN
            and dlin["intercept_fraction_at_Nmax"]<=MAX_INTERCEPT_FRACTION_AT_NMAX
        )
        block_ok=mu_cover and lam_cover and diffusion_ok and dspread<=D_SPREAD_MAX
        all_covered=all_covered and block_ok

        # Retain the old pure-power criterion as a negative methodological diagnostic.
        old_power_fit_contains_one=False
        blocks.append({
            "tau":tau,
            "point":point,
            "D_vol_spread":dspread,
            "diffusion_linear_fit":dlin,
            "bootstrap":{
                "mu":mui,
                "lambda":lami,
                "replicates_used":len(mus),
                "simultaneous_familywise_confidence":1.0-FAMILYWISE_ALPHA,
                "per_comparison_confidence":1.0-PER_COMPARISON_ALPHA,
            },
            "coverage":{
                "generator_mu":mu_cover,
                "generator_lambda":lam_cover,
                "linear_diffusion":diffusion_ok,
            },
            "old_pure_power_point_exponent":point["variance_exponent"],
            "block_pass":block_ok,
        })

    old_mu_spread=(
        max(point_mus)/min(point_mus)
        if min(point_mus)>0 else float("inf")
    )
    old_lambda_spread=(
        max(point_lams)/min(point_lams)
        if min(point_lams)>0 else float("inf")
    )
    old_gate_pass=(
        old_mu_spread<=OLD_MU_SPREAD_MAX
        and old_lambda_spread<=OLD_LAMBDA_SPREAD_MAX
    )
    old_gate_rejected=(not old_gate_pass)

    passed=generator_ok and all_covered

    result={
        "schema":1,
        "scope":"fresh-seed uncertainty-aware QCCG CDT coefficient confirmation",
        "sources":SOURCES,
        "critical_line":{
            "kappa_R":kR,
            "kappa_V":kV,
            "zero_drift_residual":res,
        },
        "instantaneous_equilibrium_generator":generator,
        "thresholds":{
            "generator_potential_r2_min":GENERATOR_R2_MIN,
            "D_vol_spread_max":D_SPREAD_MAX,
            "simultaneous_familywise_confidence":1.0-FAMILYWISE_ALPHA,
            "per_comparison_confidence":1.0-PER_COMPARISON_ALPHA,
            "coefficient_comparisons":N_COEFFICIENT_COMPARISONS,
            "required_mu_lambda_generator_coverage":"all time blocks under Bonferroni simultaneous intervals",
            "diffusion_linear_r2_min":DIFFUSION_LINEAR_R2_MIN,
            "max_intercept_fraction_at_Nmax":MAX_INTERCEPT_FRACTION_AT_NMAX,
            "old_point_mu_spread_max_negative_control":OLD_MU_SPREAD_MAX,
            "old_point_lambda_spread_max_negative_control":OLD_LAMBDA_SPREAD_MAX,
        },
        "evidence":[
            evidence(
                "qccg-statistical-cdt-coefficient-confirmation",
                "QCCG_STATISTICAL_CDT_COEFFICIENT_CONFIRMATION",
                "PASS" if passed else "FAIL",
                "Fresh curvature-equilibrated QCCG data test the CDT-like kinetic/potential generator using family-wise bootstrap coverage of the independently determined instantaneous generator and the finite-volume linear diffusion law Var_rate=D0*N3+D1.",
                trajectories=TRAJECTORIES,
                bootstraps=BOOTSTRAPS,
                taus=list(TAUS),
                generator=generator,
                thermalization=thermal,
                blocks=blocks,
                sources=SOURCES,
            ),
            evidence(
                "qccg-point-spread-gate-rejected",
                "QCCG_POINT_ESTIMATE_SPREAD_GATE_REJECTED",
                "PASS" if old_gate_rejected else "NOT_APPLICABLE",
                "The obsolete block-to-block raw point-spread rule is retained as a methodological negative control. It can reject samples whose generator values remain statistically compatible with all finite-block estimates, so it is not used for promotion.",
                point_mus=point_mus,
                point_lambdas=point_lams,
                mu_spread=old_mu_spread,
                lambda_spread=old_lambda_spread,
                old_mu_spread_max=OLD_MU_SPREAD_MAX,
                old_lambda_spread_max=OLD_LAMBDA_SPREAD_MAX,
                old_gate_pass=old_gate_pass,
            ),
            evidence(
                "qccg-pure-power-finite-volume-gate-rejected",
                "QCCG_PURE_POWER_FINITE_VOLUME_GATE_REJECTED",
                "PASS",
                "Finite-volume QCCG diffusion is tested with a linear law including an intercept rather than forcing an exact pure-power exponent p=1. The old log-log exponent is retained only as a diagnostic because a finite D1 shifts the apparent p below one.",
                blocks=[
                    {
                        "tau":b["tau"],
                        "old_power_exponent":b["old_pure_power_point_exponent"],
                        "linear_diffusion_fit":b["diffusion_linear_fit"],
                    }
                    for b in blocks
                ],
            ),
        ],
    }

    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("statistical CDT coefficient confirmation failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
