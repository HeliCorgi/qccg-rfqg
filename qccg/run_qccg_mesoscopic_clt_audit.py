#!/usr/bin/env python3
"""Mesoscopic Gaussianization mechanism for the QCCG -> CDT volume bridge.

Uses the increasing-volume triangulations from
run_qccg_large_volume_diffusion_scan.py.

For a frozen continuous-time local Pachner generator with unit rate per move,
the volume-jump cumulants per unit time are

  K_r = sum_moves (Delta N3)^r.

Choose a mesoscopic block time tau_N=N3^(-1/2).  If all K_r are extensive:
- expected number of local events ~ N3*tau_N ~ N3^(1/2) -> infinity;
- relative drift ~ tau_N*K1/N3 ~ N3^(-1/2) -> 0;
- standardized skewness ~ N3^(-1/4);
- excess kurtosis ~ N3^(-1/2).

This is a frozen-generator CLT mechanism, not yet a proof for the fully
state-dependent time-evolved QCCG transfer kernel.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import run_qccg_large_volume_diffusion_scan as base


SKEW_EXP_WINDOW = (-0.36, -0.14)
KURT_EXP_WINDOW = (-0.62, -0.38)
DRIFT_EXP_WINDOW = (-0.62, -0.38)
R2_MIN = 0.98


def cumulants(c):
    return {
        1: 3*c["14"] - 3*c["41"] + c["23"] - c["32"],
        2: 9*(c["14"] + c["41"]) + c["23"] + c["32"],
        3: 27*c["14"] - 27*c["41"] + c["23"] - c["32"],
        4: 81*(c["14"] + c["41"]) + c["23"] + c["32"],
    }


def logfit(rows, key):
    xs=[math.log(r["N3"]) for r in rows]
    ys=[math.log(r[key]) for r in rows]
    xm=sum(xs)/len(xs); ym=sum(ys)/len(ys)
    sxx=sum((x-xm)**2 for x in xs)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
    intercept=ym-slope*xm
    pred=[intercept+slope*x for x in xs]
    sse=sum((y-p)**2 for y,p in zip(ys,pred))
    sst=sum((y-ym)**2 for y in ys)
    r2=1-sse/sst if sst>0 else 1.0
    return {"exponent":slope,"amplitude":math.exp(intercept),"r2":r2}


def evidence(eid,obligation,status,note,**metadata):
    return {
        "id":eid,"obligation":obligation,"status":status,
        "engine":"qccg-mesoscopic-volume-clt",
        "artifact":"qccg/run_qccg_mesoscopic_clt_audit.py",
        "note":note,"metadata":metadata,
    }


def main():
    raw=[]
    aggregate=[]
    for ti,n3 in enumerate(base.TARGETS):
        vals=[]
        for s in range(base.SAMPLES_PER_TARGET):
            S,mixed=base.build_target(n3,base.SEED+1000*ti+s)
            c=base.counts(S)
            K=cumulants(c)
            tau=n3**-0.5
            var=tau*K[2]
            skew=(tau*K[3])/(var**1.5)
            excess=(tau*K[4])/(var**2)
            rel_drift=(tau*K[1])/n3
            vals.append({
                "abs_skewness":abs(skew),
                "excess_kurtosis":excess,
                "abs_relative_drift":abs(rel_drift),
            })
            raw.append({
                "N3":n3,"sample":s,"tau":tau,"counts":c,
                "K1":K[1],"K2":K[2],"K3":K[3],"K4":K[4],
                "abs_skewness":abs(skew),
                "excess_kurtosis":excess,
                "abs_relative_drift":abs(rel_drift),
                "mix_pairs_accepted":mixed,
            })
        aggregate.append({
            "N3":n3,
            "mean_abs_skewness":sum(v["abs_skewness"] for v in vals)/len(vals),
            "mean_excess_kurtosis":sum(v["excess_kurtosis"] for v in vals)/len(vals),
            "mean_abs_relative_drift":sum(v["abs_relative_drift"] for v in vals)/len(vals),
        })

    fs=logfit(aggregate,"mean_abs_skewness")
    fk=logfit(aggregate,"mean_excess_kurtosis")
    fd=logfit(aggregate,"mean_abs_relative_drift")

    skew_pass=SKEW_EXP_WINDOW[0]<=fs["exponent"]<=SKEW_EXP_WINDOW[1] and fs["r2"]>=R2_MIN
    kurt_pass=KURT_EXP_WINDOW[0]<=fk["exponent"]<=KURT_EXP_WINDOW[1] and fk["r2"]>=R2_MIN
    drift_pass=DRIFT_EXP_WINDOW[0]<=fd["exponent"]<=DRIFT_EXP_WINDOW[1] and fd["r2"]>=R2_MIN
    passed=skew_pass and kurt_pass and drift_pass

    result={
        "schema":1,
        "scope":"frozen-generator mesoscopic CLT mechanism; full time-evolved transfer kernel remains open",
        "evidence":[
            evidence(
                "qccg-mesoscopic-clt-scaling",
                "QCCG_MESOSCOPIC_VOLUME_CLT_SCALING",
                "PASS" if passed else "FAIL",
                "At tau_N=N3^(-1/2), standardized volume-jump skewness and excess kurtosis decrease with the expected mesoscopic central-limit scaling while relative drift vanishes.",
                aggregate=aggregate,
                skew_fit=fs,kurtosis_fit=fk,relative_drift_fit=fd,
                skew_expected_exponent="-1/4",
                kurtosis_expected_exponent="-1/2",
                relative_drift_expected_exponent="-1/2",
                exponent_windows={
                    "skew":list(SKEW_EXP_WINDOW),
                    "kurtosis":list(KURT_EXP_WINDOW),
                    "relative_drift":list(DRIFT_EXP_WINDOW),
                },
                r2_min=R2_MIN,
            ),
            evidence(
                "qccg-frozen-generator-gaussianization",
                "QCCG_FROZEN_GENERATOR_GAUSSIANIZATION",
                "PASS" if skew_pass and kurt_pass else "FAIL",
                "The frozen local Pachner generator has a mesoscopic regime in which higher standardized volume cumulants vanish with increasing N3.",
                raw=raw,
                aggregate=aggregate,
            ),
            evidence(
                "qccg-full-volume-kernel-open",
                "QCCG_CDT_DIFFUSIVE_KINETIC_LIMIT",
                "OPEN",
                "The local generator Gaussianizes, but the evolving triangulation changes its move counts. Full time-blocked QCCG transition histograms must still demonstrate a stable Gaussian kernel with Var(Delta N3|N3) proportional to N3.",
            ),
        ],
    }
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if any(e["status"]=="FAIL" for e in result["evidence"][:2]):
        raise SystemExit("QCCG mesoscopic CLT audit failed")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);args=ap.parse_args();main()
