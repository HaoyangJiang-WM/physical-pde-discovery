"""Explicit fitted equations from the latest demo; synthetic data only."""

def d_opt_rhs(x):
    return -0.0 * x - 0.3152890658124521 * x / (0.4 + x)

def struct_risk_rhs(x):
    return -0.0 * x - 0.3152890658124521 * x / (0.4 + x)
