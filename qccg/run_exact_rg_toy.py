#!/usr/bin/env python3
"""Exact finite-state real-space RG toy for the QCCG/RFQG bridge.

This audit uses a translation-invariant K-state clock/Weyl bond weight
  w(r) = exp[J cos(2 pi r/K)]
on Z_K.  Integrating out every other site is exact convolution
  w'(r) = sum_s w(r-s) w(s).

Purposes:
1. show that a one-harmonic coupling ansatz is NOT closed under exact blocking;
2. show that the complete finite harmonic basis IS closed (up to an additive
   normalization constant in the effective action);
3. show that this minimal 1D fixed-geometry toy flows to the trivial uniform IR
   fixed point, so it is a negative control rather than evidence for a Reuter
   non-Gaussian fixed point.

No claim about full QCCG or gravitational asymptotic safety is made.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path


K = 16
J0 = 0.8
STEPS = 8
SINGLE_HARMONIC_MAX_REL_ERR = 5.0e-3
FULL_RECON_TOL = 5.0e-12
UNIFORM_TV_TARGET = 2.0e-3
NUMERICAL_TV_FLOOR = 1.0e-12


def normalize_prob(w):
    s = sum(w)
    return [x / s for x in w]


def convolution(a, b):
    n = len(a)
    return [
        sum(a[(r - s) % n] * b[s] for s in range(n))
        for r in range(n)
    ]


def dft(a):
    n = len(a)
    out = []
    for m in range(n):
        z = 0j
        for r, x in enumerate(a):
            z += x * cmath.exp(-2j * math.pi * m * r / n)
        out.append(z)
    return out


def effective_action(p):
    # Additive constant is physically irrelevant; set S(0)=0.
    logs = [-math.log(max(x, 1e-300)) for x in p]
    c = logs[0]
    return [x - c for x in logs]


def cosine_coeffs(values):
    """Discrete real Fourier coefficients for an even function on Z_K.

    Returns a0 and coefficients a_n in
      f(r) = a0 + sum_{n=1}^{K/2-1} a_n cos(2 pi n r/K)
             + a_{K/2} cos(pi r)
    for even K.
    """
    n = len(values)
    coeff = []
    for m in range(n // 2 + 1):
        raw = sum(
            values[r] * math.cos(2 * math.pi * m * r / n)
            for r in range(n)
        )
        if m == 0 or m == n // 2:
            coeff.append(raw / n)
        else:
            coeff.append(2.0 * raw / n)
    return coeff


def reconstruct_cos(coeff, n):
    vals = []
    for r in range(n):
        x = coeff[0]
        for m in range(1, n // 2):
            x += coeff[m] * math.cos(2 * math.pi * m * r / n)
        x += coeff[n // 2] * math.cos(math.pi * r)
        vals.append(x)
    return vals


def max_abs(a):
    return max(abs(x) for x in a)


def rel_error(a, b):
    num = max(abs(x - y) for x, y in zip(a, b))
    den = max(1e-15, max_abs(a))
    return num / den


def total_variation_uniform(p):
    u = 1.0 / len(p)
    return 0.5 * sum(abs(x - u) for x in p)


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "exact-real-space-rg-toy",
        "artifact": "qccg/run_exact_rg_toy.py",
        "note": note,
        "metadata": metadata,
    }


def run():
    theta = [2.0 * math.pi * r / K for r in range(K)]
    p0 = normalize_prob([math.exp(J0 * math.cos(t)) for t in theta])
    p1 = normalize_prob(convolution(p0, p0))

    s1 = effective_action(p1)
    coeff = cosine_coeffs(s1)

    # One-harmonic truncation: preserve only constant + cos(theta).
    one = [0.0 for _ in coeff]
    one[0] = coeff[0]
    one[1] = coeff[1]
    s_one = reconstruct_cos(one, K)
    one_err = rel_error(s1, s_one)

    # Complete finite harmonic basis.
    s_full = reconstruct_cos(coeff, K)
    full_err = rel_error(s1, s_full)

    generated = {
        str(m): coeff[m]
        for m in range(2, K // 2 + 1)
        if abs(coeff[m]) > 1.0e-10
    }
    single_rejected = one_err > SINGLE_HARMONIC_MAX_REL_ERR and bool(generated)
    full_closed = full_err <= FULL_RECON_TOL

    # Exact repeated blocking on normalized bond probabilities.
    rows = []
    p = p0
    for step in range(STEPS + 1):
        eig = dft(p)
        # With probability normalization, lambda_0 = 1.
        subleading = max(abs(z) for z in eig[1:])
        tv = total_variation_uniform(p)
        rows.append({
            "step": step,
            "total_variation_to_uniform": tv,
            "max_nontrivial_transfer_eigenvalue_abs": subleading,
        })
        p = normalize_prob(convolution(p, p))

    monotone_tv = all(
        (
            rows[i + 1]["total_variation_to_uniform"]
            < rows[i]["total_variation_to_uniform"]
        )
        or (
            rows[i]["total_variation_to_uniform"] <= NUMERICAL_TV_FLOOR
            and rows[i + 1]["total_variation_to_uniform"] <= NUMERICAL_TV_FLOOR
        )
        for i in range(len(rows) - 1)
    )
    trivial_flow = monotone_tv and rows[-1]["total_variation_to_uniform"] <= UNIFORM_TV_TARGET

    result = {
        "schema": 1,
        "scope": "exact K-state fixed-geometry clock/Weyl blocking toy; not gravitational QCCG FRG",
        "evidence": [
            evidence(
                "qccg-exact-blocking-toy",
                "EXACT_FINITE_RG_BLOCKING_TOY",
                "PASS",
                "Integrating out every other site is implemented exactly as convolution of positive Z_K bond weights.",
                K=K,
                J0=J0,
                initial_probability=p0,
                once_blocked_probability=p1,
            ),
            evidence(
                "qccg-single-harmonic-truncation-rejected",
                "SINGLE_COUPLING_RG_TRUNCATION_REJECTED",
                "PASS" if single_rejected else "FAIL",
                "Exact blocking generates higher harmonics, so the one-coupling cosine truncation is not closed under RG.",
                one_harmonic_relative_error=one_err,
                rejection_threshold=SINGLE_HARMONIC_MAX_REL_ERR,
                generated_higher_harmonics=generated,
                full_coefficients=coeff,
            ),
            evidence(
                "qccg-full-harmonic-rg-closure",
                "FULL_FINITE_HARMONIC_RG_CLOSURE_TOY",
                "PASS" if full_closed else "FAIL",
                "The complete finite Z_K harmonic basis reconstructs the exact blocked effective bond action to numerical precision.",
                reconstruction_relative_error=full_err,
                tolerance=FULL_RECON_TOL,
                coefficients=coeff,
            ),
            evidence(
                "qccg-trivial-rg-flow-negative-control",
                "TRIVIAL_RG_FLOW_NEGATIVE_CONTROL",
                "PASS" if trivial_flow else "FAIL",
                "The minimal 1D fixed-geometry toy flows monotonically to the uniform trivial IR fixed point; it therefore supplies no non-Gaussian gravitational fixed-point evidence.",
                rows=rows,
                uniform_tv_target=UNIFORM_TV_TARGET,
                numerical_tv_floor=NUMERICAL_TV_FLOOR,
                monotonicity_rule="strict decrease above numerical floor; remain within floor afterwards",
            ),
            evidence(
                "qccg-nonperturbative-gravity-rg-still-open",
                "QCCG_NONPERTURBATIVE_RG_FLOW",
                "OPEN",
                "Exact blocking is demonstrated only for a fixed-geometry 1D clock/Weyl toy. The graph-changing constrained QCCG theory still lacks nonperturbative beta functions for its essential gravitational couplings.",
                required_repair=(
                    "Retain an RG-closed operator basis and perform blocking on the actual "
                    "QCCG cellulation/constraint ensemble; a single-coupling ansatz is ruled out."
                ),
            ),
        ],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = run()
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failed = [e for e in result["evidence"] if e["status"] == "FAIL"]
    if failed:
        raise SystemExit("exact RG toy audit failed: " + ", ".join(e["obligation"] for e in failed))


if __name__ == "__main__":
    main()
