"""
Hydraulic structures: sharp-crested weirs and hydraulic jumps.

All units are SI. Hydraulic-jump relations are for a horizontal,
rectangular channel (classical Belanger equation).
"""

import math

from .open_channel import G


# --- Sharp-crested rectangular weir ----------------------------------------
def rehbock_cd(H: float, P: float) -> float:
    """Discharge coefficient after Rehbock: Cd = 0.611 + 0.08 H / P.

    H : head above the weir crest [m]
    P : weir height above the channel bed [m]
    Valid for a fully ventilated weir, roughly H/P < 1 and H > 0.03 m.
    """
    if H <= 0 or P <= 0:
        raise ValueError("Head H and weir height P must be positive.")
    return 0.611 + 0.08 * H / P


def weir_discharge(H: float, b: float, P: float, Cd: float | None = None) -> float:
    """Discharge over a full-width sharp-crested weir (Poleni equation).

    Q = (2/3) * Cd * b * sqrt(2g) * H^(3/2)

    If Cd is not given, the Rehbock coefficient is used.
    """
    if b <= 0:
        raise ValueError("Weir width b must be positive.")
    cd = rehbock_cd(H, P) if Cd is None else Cd
    return (2.0 / 3.0) * cd * b * math.sqrt(2.0 * G) * H ** 1.5


def weir_head(Q: float, b: float, P: float, rel_tol: float = 1e-10) -> float:
    """Head H [m] over a sharp-crested weir needed to pass discharge Q."""
    if Q <= 0 or b <= 0 or P <= 0:
        raise ValueError("Q, b and P must all be positive.")
    lo, hi = 1e-9, 1.0
    while weir_discharge(hi, b, P) < Q:
        lo = hi
        hi *= 2.0
        if hi > 1e4:
            raise RuntimeError("Could not bracket the solution.")
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if weir_discharge(mid, b, P) < Q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- Hydraulic jump (rectangular, horizontal bed) ---------------------------
def froude_rect(y: float, q: float) -> float:
    """Froude number in a rectangular channel, q = unit discharge [m^2/s]."""
    return q / (y * math.sqrt(G * y))


def conjugate_depth(y1: float, q: float) -> float:
    """Sequent depth y2 downstream of a jump (Belanger equation).

    y2 / y1 = 0.5 * (sqrt(1 + 8 Fr1^2) - 1)
    """
    fr1 = froude_rect(y1, q)
    if fr1 <= 1.0:
        raise ValueError(f"Inflow must be supercritical (Fr1 = {fr1:.2f}).")
    return 0.5 * y1 * (math.sqrt(1.0 + 8.0 * fr1 ** 2) - 1.0)


def jump_energy_loss(y1: float, y2: float) -> float:
    """Head loss in a hydraulic jump: dE = (y2 - y1)^3 / (4 y1 y2) [m]."""
    return (y2 - y1) ** 3 / (4.0 * y1 * y2)


def jump_type(fr1: float) -> str:
    """Classification of the jump by inflow Froude number (USBR)."""
    if fr1 <= 1.0:
        return "no jump"
    if fr1 < 1.7:
        return "undular jump"
    if fr1 < 2.5:
        return "weak jump"
    if fr1 < 4.5:
        return "oscillating jump"
    if fr1 < 9.0:
        return "steady jump"
    return "strong jump"


def hydraulic_jump(y1: float, q: float) -> dict:
    """Full hydraulic-jump analysis for inflow depth y1 and unit discharge q."""
    fr1 = froude_rect(y1, q)
    y2 = conjugate_depth(y1, q)
    E1 = y1 + (q / y1) ** 2 / (2.0 * G)
    dE = jump_energy_loss(y1, y2)
    return {
        "y1_m": y1,
        "y2_m": y2,
        "froude_1": fr1,
        "froude_2": froude_rect(y2, q),
        "energy_loss_m": dE,
        "efficiency_pct": 100.0 * (E1 - dE) / E1,
        # Approximate roller length for a classical jump (L ~ 6 y2)
        "length_approx_m": 6.0 * y2,
        "type": jump_type(fr1),
    }
