import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class AttachGraphFeat(BaseTransform):
    r""" 
    Drop vertex and edge features from graph
    """
    def __init__(self, path_graph_feat: str):

    def __call__(self, data: Data):
        data.x = None
        data.edge_attr = None
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'