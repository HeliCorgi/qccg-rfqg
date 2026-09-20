#!/usr/bin/env python3
"""Finite-Weyl code-subspace 1D HDA analogue.

On a continuum circle define the symmetrized first-order operators
  H[N] = 1/2 { A_N(X), P },  A_N=N f,
  D[V] = 1/2 { V(X), P }.
They obey
  [H[N], H[M]] = -i D[f^2 (N M' - M N')]
for the half-density representation.

This audit realizes X as the diagonal clock coordinate and P as the finite
spectral/Weyl momentum operator.  Smooth low-Fourier code states are tested for
convergence to the noncentral, smearing-dependent structure-function algebra.
A near-Nyquist state is a negative control for aliasing.

This is a 1D HDA analogue, not the full 3+1 GR hypersurface-deformation algebra.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path


K_VALUES = (17, 25, 33, 49)
LOW_MODES = (-2, -1, 0, 1, 2)
F_ALPHA = 0.2
PASS_REL_TOL = 5.0e-10
ALIAS_FAIL_MIN = 1.0e-2


def grid(K):
    xs = [2.0 * math.pi * n / K for n in range(K)]
    half = K // 2
    ms = list(range(-half, half + 1))
    return xs, ms


def spectral_p_matrix(K):
    xs, ms = grid(K)
    root = math.sqrt(K)
    # F[n,m] = <x_n|m>
    F = [[cmath.exp(1j * m * x) / root for m in ms] for x in xs]
    # P = F diag(m) F^dagger.
    P = [[0j for _ in range(K)] for _ in range(K)]
    for a in range(K):
        for b in range(K):
            P[a][b] = sum(F[a][j] * ms[j] * F[b][j].conjugate() for j in range(K))
    return P


def diag(vals):
    K = len(vals)
    return [[vals[i] if i == j else 0j for j in range(K)] for i in range(K)]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mscale(A, s):
    return [[s * z for z in row] for row in A]


def comm(A, B):
    return madd(matmul(A, B), mscale(matmul(B, A), -1))


def sym_first(Avals, P):
    A = diag(Avals)
    return mscale(madd(matmul(A, P), matmul(P, A)), 0.5)


def apply(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def norm(v):
    return math.sqrt(sum(abs(z) ** 2 for z in v))


def fourier_state(K, modes):
    xs, _ = grid(K)
    coeff = {m: complex(1.0 + 0.13 * m, -0.2 + 0.07 * m) for m in modes}
    vals = [
        sum(c * cmath.exp(1j * m * x) for m, c in coeff.items())
        for x in xs
    ]
    n = norm(vals)
    return [z / n for z in vals]


def residual(K, modes):
    xs, _ = grid(K)
    P = spectral_p_matrix(K)

    N = [math.sin(x) for x in xs]
    M = [math.cos(x) for x in xs]
    f = [1.0 + F_ALPHA * math.cos(x) for x in xs]

    # N M' - M N' = -1 for N=sin x, M=cos x.
    A = [n * ff for n, ff in zip(N, f)]
    B = [m * ff for m, ff in zip(M, f)]
    V = [-(ff ** 2) for ff in f]

    HN = sym_first(A, P)
    HM = sym_first(B, P)
    DV = sym_first(V, P)

    lhs = comm(HN, HM)
    # target: [HN,HM] + i DV = 0
    R = madd(lhs, mscale(DV, 1j))

    psi = fourier_state(K, modes)
    rpsi = apply(R, psi)
    tpsi = apply(DV, psi)
    return {
        "K": K,
        "modes": list(modes),
        "absolute_residual": norm(rpsi),
        "relative_residual": norm(rpsi) / max(norm(tpsi), 1.0e-30),
    }


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "finite-weyl-1d-hda-code-audit",
        "artifact": "qccg/run_1d_hda_code_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    low = [residual(K, LOW_MODES) for K in K_VALUES]
    low_pass = max(r["relative_residual"] for r in low) <= PASS_REL_TOL

    high = []
    for K in K_VALUES:
        h = K // 2
        modes = (h - 2, h - 1)
        high.append(residual(K, modes))
    alias_detected = high[-1]["relative_residual"] >= ALIAS_FAIL_MIN

    # Confirm the structure function is not a constant: f^2 varies over the circle.
    xs, _ = grid(K_VALUES[-1])
    qinv = [(1.0 + F_ALPHA * math.cos(x)) ** 2 for x in xs]
    noncentral = max(qinv) - min(qinv) > 0.1

    result = {
        "schema": 1,
        "scope": "finite-Weyl low-Fourier 1D HDA analogue; not full 3+1 GR HDA",
        "evidence": [
            evidence(
                "qccg-1d-hda-code-match",
                "ONE_D_HDA_STRUCTURE_FUNCTION_CODE_MATCH",
                "PASS" if low_pass and noncentral else "FAIL",
                "On smooth low-Fourier code states the finite spectral/Weyl operators reproduce the 1D HDA-like commutator with smearing dependence and a nonconstant metric-like structure function q^xx=f(X)^2.",
                alpha=F_ALPHA,
                smearing_N="sin(x)",
                smearing_M="cos(x)",
                structure_function="(1+alpha*cos(x))^2",
                rows=low,
                preregistered_relative_tolerance=PASS_REL_TOL,
                structure_function_range=[min(qinv), max(qinv)],
            ),
            evidence(
                "qccg-1d-hda-alias-negative-control",
                "ONE_D_HDA_ALIASING_NEGATIVE_CONTROL",
                "PASS" if alias_detected else "FAIL",
                "Near-Nyquist code states violate the continuum HDA-like commutator because finite spectral products alias, so the result is explicitly a low-energy code-subspace statement.",
                rows=high,
                minimum_expected_final_relative_residual=ALIAS_FAIL_MIN,
            ),
            evidence(
                "qccg-full-3plus1-hda-still-open",
                "GR_HDA_STRUCTURE_FUNCTION_MATCH",
                "OPEN",
                "The 1D analogue now includes smearings and a noncentral metric-like structure function, but no 3+1 tensorial inverse-metric HDA with Hamiltonian/momentum constraints has been derived from QCCG.",
                next_step=(
                    "Promote f(X)^2 to a matrix-valued q^{ij}(X) code operator and construct "
                    "multiple Hamiltonian/momentum generators whose low-energy commutators reproduce "
                    "the tensorial HDA smearings while preserving the interacting physical kernel."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("1D HDA code-subspace audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
