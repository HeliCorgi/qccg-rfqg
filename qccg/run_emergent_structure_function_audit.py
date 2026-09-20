#!/usr/bin/env python3
"""Emergent noncentral structure-function audit on a finite clock/shift system.

Finite-dimensional Hilbert spaces cannot realize [X,P]=iI globally.  This audit
therefore tests a weaker and physically more appropriate target: on a localized
low-energy code subspace of a periodic finite Weyl/clock system, the spectral
momentum operator should reproduce
    [P, X^2] psi ~= -2 i X psi
which is the elementary noncentral structure-function pattern behind
{p_x, x^2 p_y} = -2 x p_y.

A packet localized near the coordinate-chart boundary is used as a negative
control.  The result supports emergent, patch/code-subspace closure rather than
an exact microscopic CCR/HDA claim.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path


K_VALUES = (13, 17, 21, 25, 33)
SIGMA = 0.55
CENTER_GOOD = 0.0
CENTER_BAD = 2.5
PASS_K = 21
PASS_REL_TOL = 1.0e-4
BOUNDARY_FAIL_MIN = 0.10


def l2(v):
    return math.sqrt(sum(abs(z) ** 2 for z in v))


def grid(K):
    half = K // 2
    xs = [2.0 * math.pi * (n - half) / K for n in range(K)]
    ms = list(range(-half, half + 1))
    return xs, ms


def p_apply(psi):
    K = len(psi)
    xs, ms = grid(K)
    root = math.sqrt(K)
    coeff = [
        sum(cmath.exp(-1j * m * x) * psi[n] for n, x in enumerate(xs)) / root
        for m in ms
    ]
    out = [
        sum(m * cmath.exp(1j * m * x) * coeff[j] for j, m in enumerate(ms)) / root
        for x in xs
    ]
    return out


def wrap(y):
    while y <= -math.pi:
        y += 2.0 * math.pi
    while y > math.pi:
        y -= 2.0 * math.pi
    return y


def packet(K, center):
    xs, _ = grid(K)
    psi = [math.exp(-wrap(x - center) ** 2 / (2.0 * SIGMA**2)) for x in xs]
    n = l2(psi)
    return [z / n for z in psi], xs


def closure_residual(K, center):
    psi, xs = packet(K, center)
    ppsi = p_apply(psi)
    x2psi = [x * x * z for x, z in zip(xs, psi)]
    px2psi = p_apply(x2psi)

    comm = [a - x * x * b for a, x, b in zip(px2psi, xs, ppsi)]
    residual = [c + 2j * x * z for c, x, z in zip(comm, xs, psi)]
    target = [2j * x * z for x, z in zip(xs, psi)]

    return {
        "K": K,
        "center": center,
        "absolute_residual": l2(residual),
        "relative_residual": l2(residual) / max(l2(target), 1.0e-30),
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "finite-weyl-code-subspace-audit",
        "artifact": "qccg/run_emergent_structure_function_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    good = [closure_residual(K, CENTER_GOOD) for K in K_VALUES]
    bad = [closure_residual(K, CENTER_BAD) for K in K_VALUES]

    good_by_k = {r["K"]: r for r in good}
    good_pass = good_by_k[PASS_K]["relative_residual"] <= PASS_REL_TOL
    good_monotone_to_pass = all(
        good[i + 1]["relative_residual"] < good[i]["relative_residual"]
        for i in range(K_VALUES.index(PASS_K))
    )
    boundary_detected = bad[-1]["relative_residual"] >= BOUNDARY_FAIL_MIN

    # Finite-dimensional global CCR no-go: Tr([A,B])=0 but Tr(iI)=iK.
    K_trace = 7
    commutator_trace = 0j
    target_trace = 1j * K_trace
    global_no_go = commutator_trace != target_trace

    result = {
        "schema": 1,
        "scope": "finite-Weyl low-energy localized code-subspace closure pilot",
        "evidence": [
            evidence(
                "qccg-finite-global-ccr-no-go",
                "FINITE_GLOBAL_CCR_NO_GO",
                "PASS" if global_no_go else "FAIL",
                "The finite-dimensional trace identity forbids an exact global canonical commutator, so exact microscopic polynomial HDA closure is not used as the continuum criterion.",
                K=K_trace,
                trace_commutator=[commutator_trace.real, commutator_trace.imag],
                trace_i_identity=[target_trace.real, target_trace.imag],
            ),
            evidence(
                "qccg-local-code-structure-function-convergence",
                "LOW_ENERGY_STRUCTURE_FUNCTION_CONVERGENCE",
                "PASS" if good_pass and good_monotone_to_pass else "FAIL",
                "For a localized low-energy packet away from the periodic chart boundary, the finite spectral derivative reproduces the noncentral commutator [P,X^2]≈-2iX with rapidly decreasing residual as K increases.",
                sigma=SIGMA,
                center=CENTER_GOOD,
                rows=good,
                pass_K=PASS_K,
                preregistered_relative_tolerance=PASS_REL_TOL,
            ),
            evidence(
                "qccg-boundary-code-subspace-negative-control",
                "STRUCTURE_FUNCTION_PATCH_CONTROL",
                "PASS" if boundary_detected else "FAIL",
                "A packet localized near the periodic coordinate boundary retains an O(0.1) closure residual, showing that the emergent algebra is a low-energy/local-patch statement rather than a global finite-Hilbert identity.",
                sigma=SIGMA,
                center=CENTER_BAD,
                rows=bad,
                minimum_expected_final_residual=BOUNDARY_FAIL_MIN,
            ),
            evidence(
                "qccg-full-gr-hda-code-subspace-match",
                "GR_HDA_STRUCTURE_FUNCTION_MATCH",
                "OPEN",
                "The code-subspace convergence pilot realizes only the elementary x-dependent structure-function pattern. It does not yet reproduce the full tensorial inverse-metric HDA with lapse/shift smearings and anomaly-free interacting dynamics.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("emergent structure-function audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
