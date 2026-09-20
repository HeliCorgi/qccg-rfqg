#!/usr/bin/env python3
"""Calibrate manifold-likeness diagnostics for the QCCG continuum gate.

This does NOT measure an actual QCCG cellulation ensemble yet.

It calibrates two dimension diagnostics on known periodic hypercubic reference
complexes:
- spectral dimension from the heat-kernel return probability;
- large-radius volume-growth dimension of Z^D.

3D and 5D references are negative controls for a 4D classifier.  The actual
QCCG cellulation-liquid measurement remains a separate OPEN obligation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


L = 32
TIMES = (8.0, 10.0, 15.0)
D4_MIN = 3.8
D4_MAX = 4.25
VOLUME_D4_MIN = 3.75
VOLUME_D4_MAX = 4.10
R1 = 16
R2 = 64


def p1(t):
    return sum(
        math.exp(-4.0 * t * math.sin(math.pi * n / L) ** 2)
        for n in range(L)
    ) / L


def spectral_dimension(D, t, eps=0.05):
    t1 = t * math.exp(-eps)
    t2 = t * math.exp(eps)
    p_lo = p1(t1) ** D
    p_hi = p1(t2) ** D
    return -2.0 * (math.log(p_hi) - math.log(p_lo)) / (2.0 * eps)


def lattice_ball_volume(D, r):
    # |{x in Z^D : ||x||_1 <= r}| = sum_j 2^j C(D,j) C(r,j)
    return sum(
        (2 ** j) * math.comb(D, j) * math.comb(r, j)
        for j in range(0, min(D, r) + 1)
    )


def volume_growth_dimension(D):
    v1 = lattice_ball_volume(D, R1)
    v2 = lattice_ball_volume(D, R2)
    eff = (math.log(v2) - math.log(v1)) / (math.log(R2) - math.log(R1))
    return {"D": D, "r1": R1, "r2": R2, "V1": v1, "V2": v2, "effective_dimension": eff}


def classify4(ds_rows, vol):
    ds_mean = sum(x["spectral_dimension"] for x in ds_rows) / len(ds_rows)
    return (
        D4_MIN <= ds_mean <= D4_MAX
        and VOLUME_D4_MIN <= vol["effective_dimension"] <= VOLUME_D4_MAX
    ), ds_mean


def evidence(eid, obligation, status, note, **metadata):
    return {
        "id": eid,
        "obligation": obligation,
        "status": status,
        "engine": "manifold-diagnostic-calibration",
        "artifact": "qccg/run_manifold_diagnostic_audit.py",
        "note": note,
        "metadata": metadata,
    }


def main():
    refs = {}
    classifications = {}
    for D in (3, 4, 5):
        ds_rows = [{"t": t, "spectral_dimension": spectral_dimension(D, t)} for t in TIMES]
        vol = volume_growth_dimension(D)
        is4, ds_mean = classify4(ds_rows, vol)
        refs[str(D)] = {
            "spectral": ds_rows,
            "spectral_mean": ds_mean,
            "volume_growth": vol,
        }
        classifications[str(D)] = is4

    d4_pass = classifications["4"]
    controls_pass = (not classifications["3"]) and (not classifications["5"])

    result = {
        "schema": 1,
        "scope": "reference-complex calibration only; not a QCCG continuum result",
        "evidence": [
            evidence(
                "qccg-manifold-diagnostic-calibrated",
                "MANIFOLD_DIAGNOSTIC_CALIBRATED",
                "PASS" if d4_pass else "FAIL",
                "The preregistered spectral-dimension plus volume-growth classifier identifies the known 4D periodic hypercubic reference.",
                L=L,
                times=list(TIMES),
                spectral_window=[D4_MIN, D4_MAX],
                volume_window=[VOLUME_D4_MIN, VOLUME_D4_MAX],
                references=refs,
            ),
            evidence(
                "qccg-manifold-dimension-controls",
                "MANIFOLD_DIMENSION_CONTROLS",
                "PASS" if controls_pass else "FAIL",
                "The same preregistered classifier rejects known 3D and 5D periodic hypercubic references as 4D.",
                classifications=classifications,
                references=refs,
            ),
            evidence(
                "qccg-cellulation-manifold-measurement",
                "QCCG_CELLULATION_GEOMETRY_MEASURED",
                "OPEN",
                "No actual QCCG cellulation-liquid ensemble has yet been generated at increasing size and measured with the calibrated dimension diagnostics.",
            ),
        ],
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(e["status"] == "FAIL" for e in result["evidence"]):
        raise SystemExit("manifold diagnostic calibration failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main()
