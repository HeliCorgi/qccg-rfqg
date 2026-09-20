#!/usr/bin/env python3
"""QCCG generator -> symmetric transfer normalization audit.

Purpose
-------
The current QCCG curvature-equilibrated analysis infers a reversible
Fokker-Planck potential

    U(N) = mu * N^(1/3) - lambda * N

together with an affine diffusion law

    a(N) = D * (N - n0).

Those coefficients are NOT automatically the coefficients appearing in the
symmetric CDT transfer-matrix action.

For a reversible one-dimensional diffusion

    L f = b f' + (a/2) f''

with stationary density pi ~ exp(-U), detailed balance implies

    b = (a' - a U') / 2.

The similarity transform from the Markov generator to a symmetric Euclidean
Hamiltonian gives

    H_sym = -1/2 d_N (a d_N) + W(N),

    W(N) = a U'^2 / 8 - (a U')' / 4.

Thus the short-time symmetric kernel is controlled by W, not U itself.

The same derivation fixes the near-diagonal kinetic normalization.  If

    a(N) = D (N-n0),

then over a QCCG time block tau

    Var(Delta N | N) = D tau (N-n0),

while the finite-volume CDT transfer form is

    exp[-(Delta N)^2 / (Gamma (N-n0))]

near n=m=N.  Therefore

    Gamma_block = D * tau.

The remaining time-unit identification between QCCG tau and one CDT transfer
step is intentionally left open.

Primary literature anchors
--------------------------
- Ambjorn et al., "The transfer matrix in four-dimensional CDT",
  JHEP 09 (2012) 017, arXiv:1205.3791,
  DOI:10.1007/JHEP09(2012)017.
- Ambjorn et al., "Is lattice quantum gravity asymptotically safe? Making
  contact between causal dynamical triangulations and the functional
  renormalization group", Phys. Rev. D 110, 126006 (2024),
  DOI:10.1103/PhysRevD.110.126006.

This audit is a normalization / negative-control audit.  It does not yet
construct the full coarse symmetric QCCG transfer matrix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sympy as sp

import run_qccg_large_volume_diffusion_scan as base
import run_qccg_equilibrated_critical_line as eqcrit
import run_qccg_cdt_coefficient_uncertainty as bootutil


SOURCES = [
    {
        "title": "The transfer matrix in four-dimensional CDT",
        "journal": "JHEP 09 (2012) 017",
        "arxiv": "1205.3791",
        "doi": "10.1007/JHEP09(2012)017",
    },
    {
        "title": "Is lattice quantum gravity asymptotically safe? Making contact between causal dynamical triangulations and the functional renormalization group",
        "journal": "Phys. Rev. D 110, 126006 (2024)",
        "doi": "10.1103/PhysRevD.110.126006",
        "equation_42": "(omega^2 Gamma)/(omega0^2 sqrt(N4)) ~= 1.63 lambda_k g_k",
    },
]


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "qccg-generator-transfer-normalization",
        "artifact": "qccg/run_qccg_generator_transfer_normalization.py",
        "note": note,
        "metadata": metadata,
    }


def measured_generator():
    """Recompute the equilibrium generator on the audited fixed-volume ensemble."""
    samples, thermal_meta, _old_kv = eqcrit.thermal_samples()
    kV, residual = eqcrit.solve_kv(samples)
    kR = eqcrit.KAPPA_R
    gen = bootutil.generator_fit_from_equilibrium(samples, kR, kV)

    rows = []
    for N, Ss in sorted(samples.items()):
        vals = [eqcrit.local_moments(base.counts(S), kV) for S in Ss]
        rows.append({
            "N3": N,
            "mean_drift_rate": sum(v[0] for v in vals) / len(vals),
            "mean_variance_rate": sum(v[1] for v in vals) / len(vals),
        })

    Ns = [float(r["N3"]) for r in rows]
    aa = [r["mean_variance_rate"] for r in rows]
    D, intercept, r2, pred = eqcrit.linear_fit(Ns, aa)
    n0 = -intercept / D

    return {
        "kappa_R": kR,
        "kappa_V": kV,
        "zero_drift_residual": residual,
        "generator": gen,
        "D": D,
        "intercept": intercept,
        "n0": n0,
        "diffusion_r2": r2,
        "rows": [
            {**row, "affine_variance_rate_fit": fit}
            for row, fit in zip(rows, pred)
        ],
        "thermalization": thermal_meta,
    }


def symbolic_map():
    N, D, n0, mu, lam, tau = sp.symbols(
        "N D n0 mu lam tau", positive=True, real=True
    )

    U = mu * N ** sp.Rational(1, 3) - lam * N
    a = D * (N - n0)
    Up = sp.diff(U, N)
    W = sp.simplify(
        a * Up**2 / 8
        - sp.diff(a * Up, N) / 4
    )
    W_expanded = sp.expand_power_base(sp.expand(W), force=True)

    gamma_block = sp.simplify(D * tau)

    # Leading CDT-like powers in W.
    coeff_N = sp.simplify(W_expanded.coeff(N, 1))
    coeff_N13 = sp.simplify(W_expanded.coeff(N ** sp.Rational(1, 3), 1))

    # A naive identification W == c*U would require the two leading
    # coefficient ratios to agree. They do not for positive mu,lambda.
    ratio_linear = sp.simplify(coeff_N / (-lam))
    ratio_curvature = sp.simplify(coeff_N13 / mu)
    naive_same_scale = sp.simplify(ratio_linear - ratio_curvature) == 0

    return {
        "U": str(U),
        "a": str(a),
        "W": str(W_expanded),
        "Gamma_block": str(gamma_block),
        "coeff_N": str(coeff_N),
        "coeff_N13": str(coeff_N13),
        "ratio_linear_to_U": str(ratio_linear),
        "ratio_curvature_to_U": str(ratio_curvature),
        "naive_same_scale": bool(naive_same_scale),
    }


def main():
    measured = measured_generator()
    symbolic = symbolic_map()

    mu = measured["generator"]["mu"]
    lam = measured["generator"]["lambda"]
    D = measured["D"]
    n0 = measured["n0"]

    # Evaluate the leading symmetric-transfer potential coefficients.
    linear_W = D * lam**2 / 8.0
    curvature_W = -D * lam * mu / 12.0

    kinetic_pass = (
        D > 0
        and measured["diffusion_r2"] >= 0.99
        and math.isfinite(n0)
    )

    naive_rejected = (
        mu > 0
        and lam > 0
        and linear_W > 0
        and curvature_W < 0
        and symbolic["naive_same_scale"] is False
    )

    result = {
        "schema": 1,
        "scope": "generator-to-symmetric-transfer normalization; full coarse transfer extraction remains open",
        "sources": SOURCES,
        "measured_equilibrium_generator": measured,
        "symbolic_map": symbolic,
        "evidence": [
            evidence(
                "qccg-generator-transfer-kinetic-map",
                "QCCG_CDT_GENERATOR_TO_TRANSFER_KINETIC_MAP",
                "PASS" if kinetic_pass else "FAIL",
                "For an affine reversible QCCG volume diffusion a(N)=D*(N-n0), the short-time symmetric transfer kernel has the CDT finite-volume kinetic denominator with Gamma_block=D*tau. The numerical QCCG-to-CDT time-unit map remains separate.",
                D=D,
                n0=n0,
                diffusion_r2=measured["diffusion_r2"],
                symbolic_Gamma_block=symbolic["Gamma_block"],
                sources=SOURCES,
            ),
            evidence(
                "qccg-naive-fp-potential-identification-rejected",
                "QCCG_CDT_NAIVE_POTENTIAL_IDENTIFICATION_REJECTED",
                "PASS" if naive_rejected else "FAIL",
                "The equilibrium Fokker-Planck potential U(N)=mu*N^(1/3)-lambda*N is not the symmetric transfer-matrix potential. Similarity transformation produces W=a*U'^2/8-(a*U')'/4, with different leading coefficients and additional inverse-power terms.",
                generator_mu=mu,
                generator_lambda=lam,
                D=D,
                n0=n0,
                leading_W_linear_coefficient=linear_W,
                leading_W_N13_coefficient=curvature_W,
                symbolic_map=symbolic,
                sources=SOURCES,
            ),
            evidence(
                "qccg-symmetric-transfer-extraction-open",
                "QCCG_CDT_SYMMETRIC_TRANSFER_EXTRACTION",
                "OPEN",
                "A coarse symmetric QCCG transfer kernel must still be constructed or measured directly across volume sectors before the transfer-action potential coefficients delta/lambda and the de Sitter profile omega can be inserted into the CDT-FRG mapping.",
                required_next_step=(
                    "Build a dense fixed-volume QCCG macro generator from curvature-equilibrated conditional ensembles; "
                    "infer macro detailed-balance weights; symmetrize the generator; exponentiate it over audited time blocks; "
                    "and fit the resulting symmetric kernel directly to the CDT affine kinetic plus N^(1/3)/N potential."
                ),
                sources=SOURCES,
            ),
            evidence(
                "qccg-cdt-time-unit-map-still-open",
                "QCCG_CDT_TIME_UNIT_MAP",
                "OPEN",
                "Gamma_block=D*tau is known in microscopic QCCG time units, but the normalization of one QCCG time block relative to the CDT transfer-step/proper-time convention has not yet been fixed.",
                D=D,
                sources=SOURCES,
            ),
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not kinetic_pass or not naive_rejected:
        raise SystemExit("generator-to-transfer normalization audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
