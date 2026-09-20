# QCCG literature anchors

This file records external results used as **comparison anchors**.  They do not
promote a QCCG claim by themselves.  QCCG-specific matching obligations still
require QCCG calculations.

## CDT effective transfer matrix

1. J. Ambjørn, J. Gizbert-Studnicki, A. Görlich, J. Jurkiewicz,
   **The transfer matrix in four-dimensional CDT**,
   JHEP 09 (2012) 017.
   - arXiv:1205.3791
   - DOI: 10.1007/JHEP09(2012)017
   - Use here: existence of an effective transfer matrix labelled only by the
     spatial three-volume / scale factor.

2. J. Ambjørn, J. Gizbert-Studnicki, A. Görlich, J. Jurkiewicz,
   **The effective action in 4-dim CDT. The transfer matrix approach**,
   JHEP 06 (2014) 034.
   - arXiv:1403.5940
   - DOI: 10.1007/JHEP06(2014)034
   - Use here: de Sitter-phase reduced transfer matrix / effective action.
     The reference kinetic structure is
     ```text
     (1/Gamma) * (n_{t+1}-n_t)^2 / (n_{t+1}+n_t)
     ```
     with the spherical-topology minisuperspace potential conventionally
     parametrized by an `n^(1/3)` curvature term plus a linear volume term.

3. J. Ambjørn, J. Gizbert-Studnicki, A. Görlich, D. Németh,
   **Is lattice quantum gravity asymptotically safe? Making contact between
   causal dynamical triangulations and the functional renormalization group**,
   Phys. Rev. D 110, 126006 (2024).
   - DOI: 10.1103/PhysRevD.110.126006
   - Use here: comparison of the CDT scale-factor effective action with FRG,
     identification of the generic infinite-volume de Sitter limit with an
     FRG Gaussian/IR fixed-point limit, and a prescription for discussing a
     putative UV lattice fixed point.
   - Important limitation: the paper explicitly states that the CDT data allow
     a UV fixed point but do not yet prove one.

## QCCG matching policy

External CDT/FRG results are anchors, not imported QCCG evidence.

The intended chain is

```text
QCCG microscopic dynamics
    -> QCCG reduced N3 transfer matrix
    -> CDT-like effective transfer dynamics
    -> CDT/FRG matching coordinates
    -> essential critical-exponent / universality comparison
    -> Reuter fixed-point match
```

A literature result can establish that a bridge or target exists.  It cannot
establish that QCCG lands on that target without a QCCG calculation.


## Essential Reuter fixed-point target

4. A. Baldazzi, K. Falls, Y. Kluth, B. Knorr,
   **Robustness of the derivative expansion in asymptotic safety**,
   Phys. Rev. D 113, 026005 (2026).
   - arXiv:2312.03831
   - DOI: 10.1103/hlrm-d4g2
   - Use here: external essential-coupling comparison target at sixth derivative
     order.  The published fixed point has
     ```text
     g_N* = 0.364
     g_C3* = 4.490e-7
     theta_1 = 2.225
     theta_2 = -3.850
     ```
     and one relevant direction.
   - Important limitation: these numbers are an FRG target, not QCCG evidence.

## CDT -> FRG matching coordinates

For the 2024 CDT/FRG comparison, the round-four-sphere scale-factor variables
obey the published relation

```text
g_eff^2 = 24*pi*G_k/sqrt(V4)
        = (4/sqrt(6))*lambda_k*g_k
        ~= 1.633*lambda_k*g_k .
```

The QCCG program therefore needs large-volume measurements of at least
`N4`, `Gamma`, `omega`, and a controlled critical trajectory before this
map can be applied numerically.

The 2024 paper explicitly warns that a raw CDT/lattice critical exponent should
not automatically be identified with an FRG stability exponent because the
relation between the lattice correlation-length scale and the FRG
coarse-graining scale is not yet established.  The same restriction is imposed
on QCCG.


## Regge curvature for spatial-slice weighting

4. T. Regge,
   **General relativity without coordinates**,
   Nuovo Cimento 19 (1961) 558-571.
   - DOI: 10.1007/BF02733251
   - Use here: curvature on a simplicial manifold is concentrated on
     codimension-two hinges.  In a 3D tetrahedral slice the hinges are edges,
     motivating the local spatial curvature weight
     ```text
     R_Regge ~ sum_e l_e delta_e.
     ```
   - For the equilateral QCCG slice toy with unit edge length and tetrahedral
     dihedral angle `acos(1/3)`, this becomes
     ```text
     R_Regge ~ 2*pi*N1 - 6*acos(1/3)*N3
     ```
     up to the overall Regge normalization.

The curvature-weighted QCCG rate
```text
w(S -> S') = exp[-(S(S')-S(S))/2]
```
is a QCCG modeling choice used because it gives exact pairwise detailed
balance with weight `exp(-S)`.  Regge's paper supplies the geometric action
anchor, not this stochastic rate prescription.
