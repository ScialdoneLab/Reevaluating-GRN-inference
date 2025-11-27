import torch_geometric.nn as pyg_nn
import torch.nn as nn


def get_mp_layer(name):
    """ 
        Gets a Layer from src.nn.layers or pyg_nn, with src.nn.layers having precedence. Useful for message passing and normalization layers
        Case sensitive
    """
    return getattr(pyg_nn, name)


def get_layer(name):
    """ 
        Gets a Layer from pyg_nn or torch.nn, with pyg_nn having precedence. Useful for activation and linear layers 
        Case sensitive.
    """
    return getattr(nn, name)
