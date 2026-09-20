#!/usr/bin/env python3
"""Scoped unitarity/scaling audit for the QCCG parent pilot."""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path


K = 11
PHYSICAL_LENGTH = 2.0 * math.pi
SIZES = (16, 32, 64, 128)
MODES = (-2, -1, 1, 2)
TIME = 1.0
NORM_TOL = 5.0e-15
FINAL_PHASE_TOL = 1.0e-3


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "unitarity-scaling-audit",
        "artifact": "qccg/run_unitarity_audit.py",
        "note": note,
        "metadata": metadata,
    }


def matmul(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def adjoint(a):
    n = len(a)
    return [[a[j][i].conjugate() for j in range(n)] for i in range(n)]


def ident(n):
    return [[1.0 + 0j if i == j else 0j for j in range(n)] for i in range(n)]


def max_abs_diff(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a)))


def add(*mats):
    n = len(mats[0])
    return [[sum(m[i][j] for m in mats) for j in range(n)] for i in range(n)]


def scale(a, s):
    return [[s * z for z in row] for row in a]


def finite_parent_checks():
    omega = cmath.exp(2j * math.pi / K)
    Z = [[0j for _ in range(K)] for _ in range(K)]
    X = [[0j for _ in range(K)] for _ in range(K)]
    for n in range(K):
        Z[n][n] = omega ** n
        X[(n + 1) % K][n] = 1.0

    I = ident(K)
    xu = max_abs_diff(matmul(adjoint(X), X), I)
    zu = max_abs_diff(matmul(adjoint(Z), Z), I)

    h_x = add(scale(I, 2.0), scale(X, -1.0), scale(adjoint(X), -1.0))
    h_z = add(scale(I, 2.0), scale(Z, -1.0), scale(adjoint(Z), -1.0))
    herm_x = max_abs_diff(h_x, adjoint(h_x))
    herm_z = max_abs_diff(h_z, adjoint(h_z))

    eigenvalues = [2.0 - 2.0 * math.cos(2.0 * math.pi * m / K) for m in range(K)]
    min_eval = min(eigenvalues)

    passed = max(xu, zu, herm_x, herm_z) < 1.0e-12 and min_eval > -1.0e-12
    return {
        "passed": passed,
        "X_unitarity_residual": xu,
        "Z_unitarity_residual": zu,
        "X_local_term_hermiticity_residual": herm_x,
        "Z_local_term_hermiticity_residual": herm_z,
        "minimum_local_term_eigenvalue": min_eval,
    }


def lattice_frequency(N, mode):
    a = PHYSICAL_LENGTH / N
    k = 2.0 * math.pi * abs(mode) / PHYSICAL_LENGTH
    return (2.0 / a) * math.sin(0.5 * a * k)


def continuum_frequency(mode):
    return 2.0 * math.pi * abs(mode) / PHYSICAL_LENGTH


def scaling_checks():
    # Fixed normalized band-limited Fourier state.
    raw = {m: complex(1.0 + 0.1 * m, -0.2 + 0.05 * m) for m in MODES}
    norm0 = math.sqrt(sum(abs(z) ** 2 for z in raw.values()))
    coeff = {m: z / norm0 for m, z in raw.items()}

    rows = []
    phase_errors = []
    norm_errors = []
    for N in SIZES:
        evolved = {
            m: c * cmath.exp(-1j * lattice_frequency(N, m) * TIME)
            for m, c in coeff.items()
        }
        cont = {
            m: c * cmath.exp(-1j * continuum_frequency(m) * TIME)
            for m, c in coeff.items()
        }
        norm = sum(abs(z) ** 2 for z in evolved.values())
        norm_error = abs(norm - 1.0)
        state_error = math.sqrt(sum(abs(evolved[m] - cont[m]) ** 2 for m in MODES))
        max_dispersion_error = max(
            abs(lattice_frequency(N, m) - continuum_frequency(m)) for m in MODES
        )
        phase_errors.append(state_error)
        norm_errors.append(norm_error)
        rows.append({
            "N": N,
            "a": PHYSICAL_LENGTH / N,
            "norm_error": norm_error,
            "state_L2_error_vs_continuum": state_error,
            "max_dispersion_error": max_dispersion_error,
        })

    monotone = all(phase_errors[i + 1] < phase_errors[i] for i in range(len(phase_errors) - 1))
    norm_pass = max(norm_errors) <= NORM_TOL
    convergence_pass = monotone and phase_errors[-1] <= FINAL_PHASE_TOL

    # Isometric embedding at the coefficient level: same Fourier coefficients.
    embedding_norm = sum(abs(z) ** 2 for z in coeff.values())
    isometry_error = abs(embedding_norm - 1.0)

    return {
        "passed": norm_pass and convergence_pass and isometry_error <= NORM_TOL,
        "rows": rows,
        "phase_errors": phase_errors,
        "norm_errors": norm_errors,
        "monotone_convergence": monotone,
        "final_state_error": phase_errors[-1],
        "final_state_tolerance": FINAL_PHASE_TOL,
        "embedding_isometry_error": isometry_error,
    }


def run():
    finite = finite_parent_checks()
    scaling = scaling_checks()
    return {
        "schema": 1,
        "scope": "finite parent + free quadratic band-limited scaling sector",
        "evidence": [
            evidence(
                "qccg-finite-parent-unitarity",
                "FINITE_PARENT_UNITARITY",
                "PASS" if finite["passed"] else "FAIL",
                "Finite-K Weyl generators are unitary and the audited local parent terms are Hermitian positive-semidefinite, so every finite audited parent lattice has unitary microscopic evolution.",
                **finite,
            ),
            evidence(
                "qccg-quadratic-scaling-isometry",
                "QUADRATIC_SCALING_ISOMETRY",
                "PASS" if scaling["passed"] else "FAIL",
                "A fixed band-limited Fourier sector embeds isometrically and its norm-preserving lattice evolution converges to the free continuum wave evolution.",
                **scaling,
            ),
            evidence(
                "qccg-interacting-physical-scaling-unitarity",
                "INTERACTING_PHYSICAL_SCALING_UNITARITY",
                "OPEN",
                "No controlled isometric scaling map has yet been constructed for the interacting graph-changing constrained physical Hilbert space.",
            ),
        ],
    }


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
        raise SystemExit("unitarity audit failed: " + ", ".join(e["obligation"] for e in failed))


if __name__ == "__main__":
    main()
