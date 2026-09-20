# QCCG × RFQG bridge pilot

This directory is a deliberately scoped TheoryGate field test for the proposed
QCCG microscopic / RFQG-5 continuum connection.

It does **not** claim that QCCG already derives general relativity.

The first pilot checks only:

1. a finite local Weyl register;
2. the small-phase quadratic limit of the exact finite-Weyl local term;
3. a declared periodic quadratic tensor Hessian with two massless branches;
4. preregistered (1/L) finite-size scaling of the first tensor excitation;
5. persistence of declared gapped defect sectors;
6. the conditional algebra (2=D-2\Rightarrow D=4) under the separately
   declared geometric-reciprocity model choice.

TheoryGate is intentionally configured so those checks can promote only
`QCCG_QUADRATIC_TT_TOY`.

The physically stronger claims remain blocked until independent evidence exists
for:

- derivation from one explicit interacting finite-qudit parent Hamiltonian;
- a smooth 3+1 Lorentzian continuum phase;
- common matter/gravity Lorentz universality;
- continuum diffeomorphism/BRST identities;
- physical unitarity of the scaling limit;
- matching to the RFQG-5/asymptotic-safety universality class;
- the GR infrared effective action;
- real-time strong-gravity dynamics.

## Run

```bash
python qccg/run_quadratic_audit.py \
  --out artifacts/qccg/evidence.json \
  --summary artifacts/qccg/summary.json

theorygate validate qccg/theorygate.yaml

theorygate check qccg/theorygate.yaml \
  --evidence artifacts/qccg/evidence.json \
  --require QCCG_QUADRATIC_TT_TOY

theorygate check qccg/theorygate.yaml \
  --evidence artifacts/qccg/evidence.json \
  --json
```

The last report should support the scoped quadratic claim while keeping
`QCCG_EMERGENT_MASSLESS_SPIN2`, `QCCG_RFQG_UNIVERSALITY`,
`QCCG_GR_IR`, and `QCCG_STRONG_GRAVITY` blocked.


## Current extended audit ladder

The pilot now also records:

- one explicit local finite-Weyl parent Hamiltonian and a symbolic derivation of
  its quadratic Hessian;
- negative controls for a tensor mass, an ungapped defect sector and spatial
  anisotropy;
- a Lorentz audit that distinguishes irrelevant lattice (k^4) corrections
  from marginal two-derivative velocity splittings;
- finite-lattice microscopic unitarity and an isometric free band-limited
  continuum scaling test;
- the rank-four linearized first-class constraint target and rank-two TT
  projector;
- a nonlinear negative control rejecting a mutually commuting Hamiltonian-
  projector completion of the GR hypersurface-deformation algebra.

The strongest supported claim is intentionally still model-internal:
`QCCG_QUADRATIC_UNITARY_SCALING_TOY`.

### Important failed attempt

An intermediate workflow run failed after the nonlinear HDA prerequisites were
added.  The reason was that `NONLINEAR_CONSTRAINT_CLOSURE` correctly became
`BLOCKED` while its new prerequisites had no evidence, but the old CI
assertion still expected it to be `OPEN`.

The repair did not weaken the assertion.  Instead, the missing negative-control
evidence was implemented:

- generic GR HDA target: nonzero;
- naive mutually commuting projector bracket: zero;
- mismatch: detected and the naive completion rejected.

With those prerequisites discharged, the nonlinear closure obligation is now
`OPEN` for the intended scientific reason: the required noncommuting
operator-valued structure-function algebra has not yet been constructed.

## Remaining hard blockers

The current pilot does **not** discharge:

- interacting RG stability of the common causal cone;
- manifold-likeness / smooth 3+1 Lorentzian continuum reconstruction;
- nonlinear constraint / BRST closure;
- interacting graph-changing physical-Hilbert scaling unitarity;
- matching to the RFQG-5 / asymptotic-safety fixed point;
- the GR infrared physical effective action;
- real-time strong-gravity dynamics.
