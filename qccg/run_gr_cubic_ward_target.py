#!/usr/bin/env python3
"""Calibrate a cubic GR Ward-identity target at selected complex on-shell kinematics.

We expand the Einstein-Hilbert Gamma-Gamma Lagrangian around flat space to
cubic order.  Using complex massless three-point momenta with k1+k2+k3=0, the
audit checks:
- the chosen physical helicity configuration has a nonzero cubic coefficient;
- replacing any selected external graviton polarization by
  k_(mu xi_(nu)+k_(nu xi_(mu) gives zero cubic coefficient.

This is a kinematic calibration of the target Ward behavior, not a proof of
the full off-shell nonlinear Ward identities and not a QCCG vertex match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp


I = sp.I
eta = sp.diag(1, -1, -1, -1)
t1, t2, t3 = sp.symbols("t1 t2 t3")
ts = (t1, t2, t3)


def angle(a, b):
    return a[0] * b[1] - a[1] * b[0]


def square(a, b):
    return a[0] * b[1] - a[1] * b[0]


def vec_from_bisp(M):
    return sp.Matrix([
        sp.simplify((M[0, 0] + M[1, 1]) / 2),
        sp.simplify((M[0, 1] + M[1, 0]) / 2),
        sp.simplify((M[1, 0] - M[0, 1]) / (2 * I)),
        sp.simplify((M[0, 0] - M[1, 1]) / 2),
    ])


def momentum(lam, tlam):
    return vec_from_bisp(sp.Matrix(lam) * sp.Matrix(tlam).T)


def pol_minus(lam, tlam, q):
    return vec_from_bisp(sp.Matrix(q) * sp.Matrix(tlam).T / angle(q, lam))


def pol_plus(lam, tlam, tq):
    return vec_from_bisp(sp.Matrix(lam) * sp.Matrix(tq).T / square(tlam, tq))


def lower(v):
    return eta * v


def graviton_pol(e):
    ec = lower(e)
    return ec * ec.T


tlam = (sp.Integer(1), sp.Integer(0))
lams = (
    (sp.Integer(1), sp.Integer(0)),
    (sp.Integer(0), sp.Integer(1)),
    (sp.Integer(-1), sp.Integer(-1)),
)
ks = tuple(momentum(l, tlam) for l in lams)
kcov = tuple(lower(k) for k in ks)


def physical_pol(lam, helicity):
    if helicity == "+":
        # Common reference spinor works because [tilde lambda, tilde q] != 0.
        e = pol_plus(lam, tlam, (sp.Integer(0), sp.Integer(1)))
    else:
        q = (sp.Integer(0), sp.Integer(1)) if lam == lams[0] else (sp.Integer(1), sp.Integer(0))
        e = pol_minus(lam, tlam, q)
    return graviton_pol(e)


def cubic_coefficient(eps_list):
    h = sp.zeros(4)
    for tv, eps in zip(ts, eps_list):
        h += tv * eps

    dh = []
    for a in range(4):
        M = sp.zeros(4)
        for tv, kv, eps in zip(ts, kcov, eps_list):
            M += I * tv * kv[a] * eps
        dh.append(M)

    hup = eta * h * eta
    htr = sp.trace(eta * h)
    P1 = sp.Rational(1, 2) * htr * eta - hup

    G1 = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    G2 = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for m in range(4):
            for n in range(4):
                g1 = 0
                g2 = 0
                for s in range(4):
                    A = dh[m][s, n] + dh[n][s, m] - dh[s][m, n]
                    g1 += sp.Rational(1, 2) * eta[r, s] * A
                    g2 += -sp.Rational(1, 2) * hup[r, s] * A
                G1[r][m][n] = sp.expand(g1)
                G2[r][m][n] = sp.expand(g2)

    L3 = 0
    for m in range(4):
        for n in range(4):
            b11 = 0
            b12 = 0
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
            L3 += eta[m, n] * b12 + P1[m, n] * b11

    poly = sp.expand(L3)
    return sp.simplify(poly.coeff(t1, 1).coeff(t2, 1).coeff(t3, 1))


def gauge_pol(leg, xi_con):
    xi_cov = lower(sp.Matrix(xi_con))
    k = kcov[leg]
    return k * xi_cov.T + xi_cov * k.T


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "sympy-gr-cubic-ward-target",
        "artifact": "qccg/run_gr_cubic_ward_target.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    nulls = [sp.simplify((k.T * eta * k)[0]) for k in ks]
    mom_sum = sp.simplify(ks[0] + ks[1] + ks[2])
    kinematics_pass = all(x == 0 for x in nulls) and mom_sum == sp.zeros(4, 1)

    helicities = ("+", "+", "-")
    physical_eps = [physical_pol(lams[i], helicities[i]) for i in range(3)]
    physical_amp = cubic_coefficient(physical_eps)
    physical_nonzero = physical_amp != 0

    xis = (
        (0, 1, 1, 0),
        (1, 0, 1, 1),
        (1, 1, 0, -1),
    )
    gauge_rows = []
    gauge_pass = True
    for leg in range(3):
        eps = list(physical_eps)
        eps[leg] = gauge_pol(leg, xis[leg])
        amp = sp.simplify(cubic_coefficient(eps))
        ok = amp == 0
        gauge_pass = gauge_pass and ok
        gauge_rows.append({
            "leg": leg + 1,
            "xi": list(xis[leg]),
            "gauge_replaced_amplitude": str(amp),
            "zero": ok,
        })

    result = {
        "schema": 1,
        "scope": "selected complex on-shell GR cubic Ward calibration",
        "evidence": [
            evidence(
                "qccg-gr-cubic-kinematics",
                "GR_CUBIC_WARD_KINEMATIC_CONTROL",
                "PASS" if kinematics_pass and physical_nonzero else "FAIL",
                "The selected complex three-point momenta are null, conserve momentum, and give a nonzero EH cubic coefficient for the audited physical helicity configuration.",
                momenta=[[str(x) for x in k] for k in ks],
                null_norms=[str(x) for x in nulls],
                momentum_sum=[str(x) for x in mom_sum],
                helicities=list(helicities),
                physical_amplitude=str(physical_amp),
            ),
            evidence(
                "qccg-gr-cubic-gauge-replacement",
                "GR_CUBIC_GAUGE_REPLACEMENT_CONTROL",
                "PASS" if gauge_pass else "FAIL",
                "At the audited on-shell kinematics, replacing each selected external polarization in turn by a pure-gauge tensor makes the EH cubic coefficient vanish.",
                rows=gauge_rows,
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("GR cubic Ward target audit failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
