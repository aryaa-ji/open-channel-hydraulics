"""Tests against hand calculations and textbook relations."""

import math
import pytest

from hydraulics import Channel, weir_discharge, weir_head, hydraulic_jump, conjugate_depth
from hydraulics.open_channel import G


def test_rectangular_geometry():
    ch = Channel(b=2.0, m=0.0)
    assert ch.area(1.0) == pytest.approx(2.0)
    assert ch.wetted_perimeter(1.0) == pytest.approx(4.0)
    assert ch.hydraulic_radius(1.0) == pytest.approx(0.5)


def test_trapezoidal_geometry():
    ch = Channel(b=3.0, m=2.0)
    y = 1.5
    assert ch.area(y) == pytest.approx((3.0 + 2.0 * 1.5) * 1.5)
    assert ch.wetted_perimeter(y) == pytest.approx(3.0 + 2 * 1.5 * math.sqrt(5))
    assert ch.top_width(y) == pytest.approx(9.0)


def test_normal_depth_reproduces_discharge():
    ch = Channel(b=3.0, m=1.5, n=0.025, S=0.0005)
    Q = 10.0
    yn = ch.normal_depth(Q)
    assert ch.discharge(yn) == pytest.approx(Q, rel=1e-6)


def test_critical_depth_rectangular_closed_form():
    # Rectangular channel: yc = (q^2 / g)^(1/3)
    b, Q = 4.0, 12.0
    ch = Channel(b=b)
    expected = ((Q / b) ** 2 / G) ** (1 / 3)
    assert ch.critical_depth(Q) == pytest.approx(expected, rel=1e-6)


def test_froude_is_one_at_critical_depth():
    ch = Channel(b=3.0, m=1.0)
    Q = 8.0
    yc = ch.critical_depth(Q)
    assert ch.froude(yc, Q) == pytest.approx(1.0, rel=1e-5)


def test_mild_and_steep_slopes():
    Q = 5.0
    assert Channel(b=3.0, n=0.015, S=0.0005).summary(Q)["slope_type"] == "mild"
    assert Channel(b=3.0, n=0.015, S=0.05).summary(Q)["slope_type"] == "steep"


def test_weir_head_inverts_discharge():
    b, P = 2.0, 0.6
    Q = weir_discharge(0.3, b, P)
    assert weir_head(Q, b, P) == pytest.approx(0.3, rel=1e-6)


def test_belanger_equation():
    y1, q = 0.2, 2.0
    fr1 = q / (y1 * math.sqrt(G * y1))
    expected = 0.5 * y1 * (math.sqrt(1 + 8 * fr1 ** 2) - 1)
    assert conjugate_depth(y1, q) == pytest.approx(expected)


def test_jump_conserves_momentum():
    # Specific force M = q^2 / (g y) + y^2 / 2 must be equal on both sides
    y1, q = 0.15, 1.5
    y2 = conjugate_depth(y1, q)
    M1 = q ** 2 / (G * y1) + y1 ** 2 / 2
    M2 = q ** 2 / (G * y2) + y2 ** 2 / 2
    assert M1 == pytest.approx(M2, rel=1e-9)


def test_jump_needs_supercritical_inflow():
    with pytest.raises(ValueError):
        conjugate_depth(y1=2.0, q=1.0)


def test_triangular_channel_critical_depth():
    # Triangular channel (b = 0): yc = (2 Q^2 / (g m^2))^(1/5)
    m, Q = 1.0, 2.0
    expected = (2 * Q ** 2 / (G * m ** 2)) ** 0.2
    assert Channel(b=0.0, m=m).critical_depth(Q) == pytest.approx(expected, rel=1e-6)


def test_solver_accurate_for_very_small_flow():
    ch = Channel(b=3.0)
    Q = 1e-4
    assert ch.discharge(ch.normal_depth(Q)) == pytest.approx(Q, rel=1e-6)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        Channel(b=-1.0)
    with pytest.raises(ValueError):
        Channel(b=2.0).normal_depth(0.0)
    with pytest.raises(ValueError):
        weir_head(-1.0, b=2.0, P=0.5)


def test_jump_summary_fields():
    res = hydraulic_jump(0.15, 1.0)  # Fr1 ~ 5.5 -> steady jump
    assert res["y2_m"] > res["y1_m"]
    assert res["froude_2"] < 1.0
    assert res["energy_loss_m"] > 0
    assert res["type"] == "steady jump"
