"""
External Tool Integrations for CGCNN

Interfaces for integrating CGCNN with popular materials science tools
including ASE, PyMatGen, VASP, Quantum ESPRESSO, and Materials Project.

Author: lunazhang
Date: 2023
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .ase_interface import ASEInterface, CGCNNCalculator
        from .materials_project import MaterialsProjectAPI
    except ImportError:
        pass

__all__ = ['ASEInterface', 'CGCNNCalculator', 'MaterialsProjectAPI'] 