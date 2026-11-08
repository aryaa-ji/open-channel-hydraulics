"""Simple open-channel hydraulics toolkit."""

from .open_channel import Channel
from .structures import (
    weir_discharge,
    weir_head,
    rehbock_cd,
    hydraulic_jump,
    conjugate_depth,
    jump_energy_loss,
)

__all__ = [
    "Channel",
    "weir_discharge",
    "weir_head",
    "rehbock_cd",
    "hydraulic_jump",
    "conjugate_depth",
    "jump_energy_loss",
]
__version__ = "0.1.0"
