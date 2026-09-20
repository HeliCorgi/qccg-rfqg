#!/usr/bin/env python3
"""Selected lattice tensor-cubic Ward audit for the QCCG parent.

Construction:
- each collective tensor component is represented near the flat vacuum by the
  Hermitian finite-Weyl sine coordinate S=(Z-Z^dag)/(2 i delta);
- derivatives in the cubic Einstein-Hilbert Gamma-Gamma vertex are replaced by
  local symmetric lattice symbols sin(a k)/a;
- operator products are understood in Hermitian symmetrized form.

At cubic order around S=0, the finite-Weyl sine coordinate has unit linear
derivative, so the three-field vertex is the lattice EH cubic vertex.

The audit checks selected complex on-shell three-point kinematics:
- physical cubic amplitude converges to the continuum EH target under a->0;
- replacing each leg by the corresponding lattice pure-gauge polarization
  makes the cubic coefficient vanish at numerical precision.

This is a construction/selected-kinematics audit, not a derivation of the
vertex from QCCG RG and not a full all-momenta Ward proof.
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path


ETA = (1.0, -1.0, -1.0, -1.0)
A_VALUES = (0.5, 0.25, 0.125, 0.0625)
K_WEYL = 64
GAUGE_TOL = 5.0e-13
FINAL_REL_TARGET = 5.0e-8


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vscale(a, s):
    return [s * x for x in a]


def lower(v):
    return [ETA[i] * v[i] for i in range(4)]


def outer(a, b):
    return [[a[i] * b[j] for j in range(4)] for i in range(4)]


def madd(A, B):
    return [[A[i][j] + B[i][j] for j in range(4)] for i in range(4)]


def mscale(A, s):
    return [[s * A[i][j] for j in range(4)] for i in range(4)]


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def trace(A):
    return sum(A[i][i] for i in range(4))


def bisp_to_vec(M):
    return [
        (M[0][0] + M[1][1]) / 2,
        (M[0][1] + M[1][0]) / 2,
        (M[1][0] - M[0][1]) / (2j),
        (M[0][0] - M[1][1]) / 2,
    ]


def spinor_outer(a, b):
    return [[a[i] * b[j] for j in range(2)] for i in range(2)]


def bracket(a, b):
    return a[0] * b[1] - a[1] * b[0]


def momentum(lam, tlam):
    return bisp_to_vec(spinor_outer(lam, tlam))


def pol_plus(lam, tlam, tq=(0j, 1 + 0j)):
    den = bracket(tlam, tq)
    M = [[lam[i] * tq[j] / den for j in range(2)] for i in range(2)]
    return bisp_to_vec(M)


def pol_minus(lam, tlam, q):
    den = bracket(q, lam)
    M = [[q[i] * tlam[j] / den for j in range(2)] for i in range(2)]
    return bisp_to_vec(M)


def graviton_pol(e):
    ec = lower(e)
    return outer(ec, ec)


TLAM = (1 + 0j, 0j)
LAMS = (
    (1 + 0j, 0j),
    (0j, 1 + 0j),
    (-1 + 0j, -1 + 0j),
)
KS = tuple(momentum(l, TLAM) for l in LAMS)
KCOV = tuple(lower(k) for k in KS)


def physical_eps(i, helicity):
    lam = LAMS[i]
    if helicity == "+":
        e = pol_plus(lam, TLAM)
    else:
        q = (0j, 1 + 0j) if i == 0 else (1 + 0j, 0j)
        e = pol_minus(lam, TLAM, q)
    return graviton_pol(e)


def cubic_l3_value(eps_list, deriv_vecs, mask):
    h = [[0j] * 4 for _ in range(4)]
    dh = [[[0j] * 4 for _ in range(4)] for _ in range(4)]

    for leg in range(3):
        if not ((mask >> leg) & 1):
            continue
        ep = eps_list[leg]
        for i in range(4):
            for j in range(4):
                h[i][j] += ep[i][j]
        for a in range(4):
            fac = 1j * deriv_vecs[leg][a]
            for i in range(4):
                for j in range(4):
                    dh[a][i][j] += fac * ep[i][j]

    # h^{mu nu}=eta_mu eta_nu h_{mu nu}
    hup = [[ETA[i] * ETA[j] * h[i][j] for j in range(4)] for i in range(4)]
    htr = sum(ETA[i] * h[i][i] for i in range(4))
    P1 = [[
        (0.5 * htr * ETA[i] if i == j else 0j) - hup[i][j]
        for j in range(4)
    ] for i in range(4)]

    G1 = [[[0j for _ in range(4)] for _ in range(4)] for _ in range(4)]
    G2 = [[[0j for _ in range(4)] for _ in range(4)] for _ in range(4)]

    for r in range(4):
        for m in range(4):
            for n in range(4):
                g1 = 0j
                g2 = 0j
                for s in range(4):
                    A = dh[m][s][n] + dh[n][s][m] - dh[s][m][n]
                    eta_rs = ETA[r] if r == s else 0.0
                    g1 += 0.5 * eta_rs * A
                    g2 += -0.5 * hup[r][s] * A
                G1[r][m][n] = g1
                G2[r][m][n] = g2

    L = 0j
    for m in range(4):
        for n in range(4):
            if m != n:
                eta_mn = 0.0
            else:
                eta_mn = ETA[m]
            b11 = 0j
            b12 = 0j
            for r in range(4):
                for s in range(4):
                    b11 += (
                        G1[r][m][s] * G1[s][n][r]
                        - G1[r][m][n] * G1[s][r][s]
                    )
                    b12 += (
                        G1[r][m][s] * G2[s][n][r]
                        + G2[r][m][s] * G1[s][n][r]
                        - G1[r][m][n] * G2[s][r][s]
                        - G2[r][m][n] * G1[s][r][s]
                    )
            L += eta_mn * b12 + P1[m][n] * b11
    return L


def cubic_coefficient(eps_list, deriv_vecs):
    out = 0j
    for mask in range(8):
        bits = mask.bit_count()
        out += ((-1) ** (3 - bits)) * cubic_l3_value(eps_list, deriv_vecs, mask)
    return out


def qhat(kcov, a):
    return [cmath.sin(a * z) / a for z in kcov]


def gauge_eps(q, xi_con):
    xi_cov = lower(list(xi_con))
    return madd(outer(q, xi_cov), outer(xi_cov, q))


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "finite-weyl-tensor-cubic-audit",
        "artifact": "qccg/run_tensor_cubic_lattice_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    helicities = ("+", "+", "-")
    eps_phys = [physical_eps(i, helicities[i]) for i in range(3)]
    continuum_amp = cubic_coefficient(eps_phys, KCOV)
    continuum_nonzero = abs(continuum_amp) > 1e-12

    xis = (
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, -1.0),
    )

    rows = []
    rel_errors = []
    max_gauge = []
    for a in A_VALUES:
        deriv = [qhat(k, a) for k in KCOV]
        amp = cubic_coefficient(eps_phys, deriv)
        rel = abs(amp - continuum_amp) / abs(continuum_amp)
        rel_errors.append(rel)

        gauges = []
        for leg in range(3):
            eps = list(eps_phys)
            eps[leg] = gauge_eps(deriv[leg], xis[leg])
            ga = cubic_coefficient(eps, deriv)
            gauges.append(abs(ga))
        maxg = max(gauges)
        max_gauge.append(maxg)
        rows.append({
            "a": a,
            "physical_amplitude_real": amp.real,
            "physical_amplitude_imag": amp.imag,
            "relative_error_vs_continuum": rel,
            "gauge_replacement_abs_amplitudes": gauges,
            "max_gauge_abs_amplitude": maxg,
        })

    monotone = all(rel_errors[i + 1] < rel_errors[i] for i in range(len(rel_errors) - 1))
    convergence = monotone and rel_errors[-1] <= FINAL_REL_TARGET
    gauge_pass = max(max_gauge) <= GAUGE_TOL

    delta = 2.0 * math.pi / K_WEYL
    result = {
        "schema": 1,
        "scope": "engineered finite-Weyl sine-coordinate / lattice-EH cubic construction at selected on-shell kinematics",
        "evidence": [
            evidence(
                "qccg-tensor-cubic-lattice-construction",
                "QCCG_TENSOR_CUBIC_VERTEX_CONSTRUCTED",
                "PASS" if continuum_nonzero else "FAIL",
                "An explicit finite-Weyl sine-coordinate tensor cubic is defined by a local lattice discretization of the EH Gamma-Gamma cubic structure; at cubic order the sine-coordinate map has unit linear derivative.",
                K_weyl=K_WEYL,
                delta=delta,
                construction=(
                    "S=(Z-Z^dag)/(2 i delta); replace derivatives in the cubic "
                    "Gamma-Gamma term by local symmetric lattice differences; "
                    "Hermitian-symmetrize operator products."
                ),
                helicities=list(helicities),
                continuum_target_amplitude={"real": continuum_amp.real, "imag": continuum_amp.imag},
            ),
            evidence(
                "qccg-selected-cubic-ward-convergence",
                "QCCG_CUBIC_SELECTED_WARD_MATCH",
                "PASS" if convergence and gauge_pass else "FAIL",
                "For the audited complex on-shell configuration, the lattice tensor cubic amplitude converges to the EH target and lattice pure-gauge replacements vanish at numerical precision.",
                rows=rows,
                final_relative_target=FINAL_REL_TARGET,
                gauge_tolerance=GAUGE_TOL,
                monotone_convergence=monotone,
            ),
            evidence(
                "qccg-full-cubic-ward-open",
                "QCCG_FULL_CUBIC_WARD_MATCH",
                "OPEN",
                "No all-momenta/all-polarization nonlinear Ward proof or RG derivation of the engineered tensor cubic has yet been established.",
                remaining=(
                    "Derive the tensor cubic from the microscopic RG flow rather than insert the EH tensor structure, "
                    "and test the complete nonlinear Ward identity beyond selected three-point kinematics."
                ),
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("tensor cubic lattice audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
