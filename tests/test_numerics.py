import numpy as np

from research.incompressible import advect, poiseuille_profile, pressure_project


def mean_abs_divergence(u, v, dx=1.0):
    div = np.gradient(u, dx, axis=1) + np.gradient(v, dx, axis=0)
    return float(np.abs(div).mean())


def test_zero_velocity_advection_is_identity():
    field = np.arange(36, dtype=float).reshape(6, 6)
    zeros = np.zeros_like(field)
    actual = advect(field, zeros, zeros, dt=0.25)
    np.testing.assert_allclose(actual, field, rtol=0.0, atol=1e-12)


def test_pressure_projection_reduces_divergence():
    n = 24
    y, x = np.mgrid[0:n, 0:n]
    u = np.sin(2 * np.pi * x / (n - 1)) * np.cos(2 * np.pi * y / (n - 1))
    v = np.cos(2 * np.pi * x / (n - 1)) * np.sin(2 * np.pi * y / (n - 1))
    before = mean_abs_divergence(u, v)
    u2, v2, reported = pressure_project(u, v, iterations=200)
    after = mean_abs_divergence(u2, v2)
    assert after < before * 0.35
    assert np.isclose(after, reported, rtol=1e-12, atol=1e-12)


def test_poiseuille_profile_boundary_symmetry_and_viscosity_scaling():
    base = poiseuille_profile(height=33, pressure_gradient=1.0, viscosity=1.0)
    viscous = poiseuille_profile(height=33, pressure_gradient=1.0, viscosity=2.0)
    assert base[0] == 0.0
    assert base[-1] == 0.0
    np.testing.assert_allclose(base, base[::-1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(viscous, base / 2.0, rtol=1e-12, atol=1e-12)
