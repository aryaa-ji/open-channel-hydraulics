"""
Worked example: a trapezoidal canal, a measuring weir and a stilling basin.

Run from the project root:
    python examples/demo.py
Figures are saved to the figures/ folder.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydraulics import Channel, weir_discharge, weir_head, hydraulic_jump  # noqa: E402
from hydraulics.open_channel import G  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def canal_example():
    print("=== 1. Trapezoidal canal (Manning) ===")
    canal = Channel(b=3.0, m=1.5, n=0.025, S=0.0005)
    Q = 10.0
    for key, value in canal.summary(Q).items():
        print(f"  {key:<20} {value:.3f}" if isinstance(value, float) else f"  {key:<20} {value}")

    # Specific-energy diagram
    yc = canal.critical_depth(Q)
    yn = canal.normal_depth(Q)
    y = np.linspace(0.3, 3.0, 300)
    E = [canal.specific_energy(d, Q) for d in y]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(E, y, label=f"E-y curve, Q = {Q} m³/s")
    ax.plot([0, 3.2], [0, 3.2], "k--", lw=0.8, label="E = y")
    ax.axhline(yc, color="tab:red", ls=":", label=f"critical depth yc = {yc:.2f} m")
    ax.axhline(yn, color="tab:green", ls="-.", label=f"normal depth yn = {yn:.2f} m")
    ax.set(xlabel="Specific energy E [m]", ylabel="Depth y [m]",
           title="Specific-energy diagram", xlim=(0, 3.2), ylim=(0, 3.0))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "specific_energy.png")

    # Rating curve
    depths = np.linspace(0.1, 3.0, 100)
    flows = [canal.discharge(d) for d in depths]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(flows, depths)
    ax.set(xlabel="Discharge Q [m³/s]", ylabel="Normal depth y [m]",
           title="Stage–discharge (rating) curve")
    fig.tight_layout()
    fig.savefig(FIG / "rating_curve.png")


def weir_example():
    print("\n=== 2. Sharp-crested measuring weir (Poleni + Rehbock) ===")
    b, P = 2.0, 0.8
    for Q in (0.2, 0.5, 1.0):
        print(f"  Q = {Q:.1f} m³/s  ->  head H = {weir_head(Q, b, P):.3f} m")

    H = np.linspace(0.02, 0.6, 100)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(H, [weir_discharge(h, b, P) for h in H])
    ax.set(xlabel="Head over crest H [m]", ylabel="Discharge Q [m³/s]",
           title=f"Weir rating curve (b = {b} m, P = {P} m)")
    fig.tight_layout()
    fig.savefig(FIG / "weir_rating.png")


def jump_example():
    print("\n=== 3. Hydraulic jump in a stilling basin ===")
    q, y1 = 2.0, 0.25  # unit discharge [m²/s], depth at the spillway toe [m]
    res = hydraulic_jump(y1, q)
    for key, value in res.items():
        print(f"  {key:<16} {value:.3f}" if isinstance(value, float) else f"  {key:<16} {value}")

    # y2/y1 and relative energy loss versus Froude number
    # Practical design range Fr1 = 1.1 ... 12; invert Fr = q / sqrt(g y^3) for y1
    froudes = np.linspace(1.1, 12.0, 200)
    y1s = (q ** 2 / (G * froudes ** 2)) ** (1 / 3)
    results = [hydraulic_jump(d, q) for d in y1s]
    fr = [r["froude_1"] for r in results]
    ratio = [r["y2_m"] / r["y1_m"] for r in results]
    loss = [100 - r["efficiency_pct"] for r in results]

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(fr, ratio, color="tab:blue", label="y2 / y1")
    ax1.set(xlabel="Inflow Froude number Fr₁", ylabel="Sequent depth ratio y2 / y1",
            title="Hydraulic jump characteristics")
    ax2 = ax1.twinx()
    ax2.plot(fr, loss, color="tab:orange", ls="--", label="energy loss")
    ax2.set_ylabel("Relative energy loss [%]")
    ax2.grid(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "hydraulic_jump.png")


if __name__ == "__main__":
    canal_example()
    weir_example()
    jump_example()
    print(f"\nFigures saved to {FIG}")
