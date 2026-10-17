"""
NeuroSeg - Theme Re-export
Allows 'from neuroseg_theme import ...' from project root.
"""

from src.neuroseg_theme import (
    apply_theme,
    hero_header,
    PLOT,
    MASK_COLORS,
    COLORS,
)

__all__ = ["apply_theme", "hero_header", "PLOT", "MASK_COLORS", "COLORS"]
