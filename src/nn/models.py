import src.nn.encoders as encoders
import src.nn.decoders as decoders
from torch_geometric.nn.models import GAE
import torch


class AutoEncoder(GAE):
    def __init__(self, input_dim, opts, num_nodes):
        encoder = getattr(encoders, opts.encoder)(input_dim, opts, num_nodes)
        decoder = getattr(decoders, opts.decoder)(opts)
        super(AutoEncoder, self).__init__(encoder=encoder, decoder=decoder)

class NaiveModel(torch.nn.Module):
    def __init__(self, input_dim, opts, num_nodes):
        super(NaiveModel, self).__init__()
        self.dummy_parameter = (torch.nn.Parameter(torch.rand((1,))))
        opts.latent_dim = input_dim
        self.decoder = getattr(decoders, opts.decoder)(opts)
    
    def encode(self, x, *args, **kwargs):
        return x
    
    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)