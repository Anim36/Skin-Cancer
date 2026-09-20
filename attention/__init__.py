"""Attention modules for feature refinement."""

from attention.cbam import CBAM
from attention.se_block import SEBlock

__all__ = ["CBAM", "SEBlock"]
