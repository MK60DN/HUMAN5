"""
BrainForge: 左右脑协作的高效LLM微调系统
"""

__version__ = "0.1.0"

from .core.brain import BrainForge
from .core.config import BrainForgeConfig

__all__ = ["BrainForge", "BrainForgeConfig"]