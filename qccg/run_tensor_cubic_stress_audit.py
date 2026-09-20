#!/usr/bin/env python3
"""Multi-kinematics stress test for the finite-Weyl tensor cubic.

Extends the selected three-point audit to several independent complex
massless momentum configurations with common tilde spinor.  For each (++-)
configuration it checks:
- nonzero continuum EH cubic coefficient;
- lattice cubic physical amplitude approaches the continuum target;
- pure-gauge replacement amplitudes decrease toward zero under refinement.

This is still a finite set of on-shell tests, not a full nonlinear Ward proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_tensor_cubic_lattice_audit as base


TLAM = (1 + 0j, 0j)
A_VALUES = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625)
FINAL_REL_TOL = 2.0e-5
FINAL_GAUGE_REL_TOL = 2.0e-5
GAUGE_NUMERICAL_FLOOR = 1.0e-12

LAMBDA_CASES = (
    ((1 + 0j, 0j), (0j, 1 + 0j), (-1 + 0j, -1 + 0j)),
    ((1 + 0j, 1 + 0j), (1 + 0j, -2 + 0j), (-2 + 0j, 1 + 0j)),
    ((2 + 0j, 1 + 0j), (-1 + 0j, 2 + 0j), (-1 + 0j, -3 + 0j)),
    ((1 + 0j, 2 + 0j), (2 + 0j, -3 + 0j), (-3 + 0j, 1 + 0j)),
)


def choose_minus_ref(lam):
    for q in ((1 + 0j, 0j), (0j, 1 + 0j), (1 + 0j, 1 + 0j)):
        if abs(base.bracket(q, lam)) > 1.0e-12:
            return q
    raise RuntimeError("no nonparallel negative-helicity reference spinor")


def eps_for(lams):
    out = []
    for i, lam in enumerate(lams):
        if i < 2:
            e = base.pol_plus(lam, TLAM, (0j, 1 + 0j))
        else:
            e = base.pol_minus(lam, TLAM, choose_minus_ref(lam))
        out.append(base.graviton_pol(e))
    return out


def kinematics(lams):
    ks = tuple(base.momentum(lam, TLAM) for lam in lams)
    kcov = tuple(base.lower(k) for k in ks)
    return ks, kcov


def null_norm(k):
    return sum(base.ETA[i] * k[i] * k[i] for i in range(4))


def add_vecs(vs):
    return [sum(v[i] for v in vs) for i in range(4)]


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "tensor-cubic-multikinematics-stress",
        "artifact": "qccg/run_tensor_cubic_stress_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    all_rows = []
    all_pass = True
    xis = (
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, -1.0),
    )

    for case_id, lams in enumerate(LAMBDA_CASES, start=1):
        ks, kcov = kinematics(lams)
        nulls = [abs(null_norm(k)) for k in ks]
        momsum = add_vecs(ks)
        kin_ok = max(nulls) < 1.0e-12 and max(abs(x) for x in momsum) < 1.0e-12

        eps = eps_for(lams)
        cont = base.cubic_coefficient(eps, kcov)
        cont_nonzero = abs(cont) > 1.0e-10

        rows = []
        rels = []
        grels = []
        for a in A_VALUES:
            deriv = [base.qhat(k, a) for k in kcov]
            amp = base.cubic_coefficient(eps, deriv)
            rel = abs(amp - cont) / max(1.0e-30, abs(cont))
            rels.append(rel)

            gauges = []
            for leg in range(3):
                ep = list(eps)
                ep[leg] = base.gauge_eps(deriv[leg], xis[leg])
                ga = base.cubic_coefficient(ep, deriv)
                gauges.append(abs(ga) / max(1.0e-30, abs(cont)))
            grel = max(gauges)
            grels.append(grel)
            rows.append({
                "a": a,
                "physical_relative_error": rel,
                "gauge_relative_errors": gauges,
                "max_gauge_relative_error": grel,
            })

        physical_improves = rels[-1] < rels[0] and rels[-1] <= FINAL_REL_TOL
        gauge_improves = (
            grels[-1] <= FINAL_GAUGE_REL_TOL
            and (
                (grels[0] > GAUGE_NUMERICAL_FLOOR and grels[-1] < grels[0])
                or (grels[0] <= GAUGE_NUMERICAL_FLOOR and grels[-1] <= GAUGE_NUMERICAL_FLOOR)
            )
        )
        case_pass = kin_ok and cont_nonzero and physical_improves and gauge_improves
        all_pass = all_pass and case_pass
        all_rows.append({
            "case": case_id,
            "lambdas": [[[z.real, z.imag] for z in lam] for lam in lams],
            "null_norm_abs": nulls,
            "momentum_sum_abs": [abs(x) for x in momsum],
            "continuum_amplitude": [cont.real, cont.imag],
            "case_pass": case_pass,
            "rows": rows,
        })

    result = {
        "schema": 1,
        "scope": "finite multi-kinematics on-shell tensor-cubic stress; not all-momenta Ward proof",
        "evidence": [
            evidence(
                "qccg-cubic-multikinematics-stress",
                "QCCG_CUBIC_MULTI_KINEMATICS_STRESS",
                "PASS" if all_pass else "FAIL",
                "The same finite-Weyl tensor cubic is stress-tested on four independent complex massless (++-) configurations for physical-amplitude convergence and gauge-replacement suppression.",
                a_values=list(A_VALUES),
                final_physical_relative_tolerance=FINAL_REL_TOL,
                final_gauge_relative_tolerance=FINAL_GAUGE_REL_TOL,
                gauge_numerical_floor=GAUGE_NUMERICAL_FLOOR,
                gauge_monotonicity_rule="decrease above numerical floor; remain within floor if already machine-zero",
                cases=all_rows,
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all_pass:
        raise SystemExit("tensor cubic multi-kinematics stress failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
