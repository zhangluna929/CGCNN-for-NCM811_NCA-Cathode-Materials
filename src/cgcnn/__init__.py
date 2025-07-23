"""
CGCNN - Crystal Graph Convolutional Neural Networks for Material Property Prediction

Implementation of the CGCNN algorithm for predicting materials properties 
from crystal structures using graph convolutional neural networks.

Author: lunazhang
Reference: Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018)
"""

from .model import CrystalGraphConvNet, CrystalGraphConvNetMulti, ConvLayer
from .data import CIFData, collate_pool

__version__ = "1.0.0"
__all__ = [
    "CrystalGraphConvNet",
    "CrystalGraphConvNetMulti", 
    "ConvLayer",
    "CIFData",
    "collate_pool"
] 