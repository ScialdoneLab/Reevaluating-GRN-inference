import torch
from src.nn.nn_utils import get_layer, get_mp_layer
from torch_geometric.nn import Sequential
from torch_geometric.utils import add_self_loops, to_undirected


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, opts, num_nodes):
        assert opts.n_conv_layers in [1, 2, 3]
        assert opts.layer_ratio >= 1

        super(GNNEncoder, self).__init__()
        self.n_conv_layers = opts.n_conv_layers
        hidden1 = int(opts.latent_dim*pow(opts.layer_ratio, opts.n_conv_layers-1))

        layers = torch.nn.ModuleList()
        flow = []

        layers.append(get_layer("Dropout")(opts.dropout_ratio))
        flow.append("x -> x")
        layers.append(get_mp_layer(opts.mplayer)(input_dim, hidden1, *opts.mplayer_args, **opts.mplayer_kwargs))
        flow.append("x, edge_index -> x")

        if opts.n_conv_layers >= 2:
            layers.append(get_layer(opts.activation_layer)())
            flow.append("x -> x")

            layers.append(get_layer("Dropout")(opts.dropout_ratio))
            flow.append("x -> x")

            hidden2 = int(opts.latent_dim*pow(opts.layer_ratio, opts.n_conv_layers-2))

            layers.append(get_mp_layer(opts.mplayer)(hidden1, hidden2, *opts.mplayer_args, **opts.mplayer_kwargs))
            flow.append("x, edge_index -> x")

        if opts.n_conv_layers == 3:
            layers.append(get_layer("Dropout")(opts.dropout_ratio))
            flow.append("x -> x")
            
            layers.append(get_mp_layer(opts.mplayer)(hidden2, opts.latent_dim, *opts.mplayer_args, **opts.mplayer_kwargs))
            flow.append("x, edge_index -> x")

        self.flow = flow
        self.nn = Sequential("x, edge_index", [(layer, flow) for layer, flow in zip(layers, flow)])

    def forward(self, x, edge_index):
        return self.nn(x, add_self_loops(to_undirected(edge_index), num_nodes=x.shape[0])[0])

class GNNGraphOnlyEncoder(GNNEncoder):
    def __init__(self, input_dim, opts, num_nodes):
        super(GNNGraphOnlyEncoder, self).__init__(num_nodes, opts, num_nodes)

    def forward(self, x, edge_index):
        print(x.shape, torch.eye(x.shape[0], device=x.device).shape)
        return self.nn(torch.eye(x.shape[0], device=x.device), add_self_loops(to_undirected(edge_index), num_nodes=x.shape[0])[0])

class MLPEncoder(torch.nn.Module):
    def __init__(self, input_dim, opts, num_nodes):
        assert opts.n_conv_layers in [1, 2, 3]
        assert opts.layer_ratio >= 1

        super(MLPEncoder, self).__init__()
        self.n_conv_layers = opts.n_conv_layers
        hidden1 = int(opts.latent_dim*pow(opts.layer_ratio, opts.n_conv_layers-1))

        layers = torch.nn.ModuleList()
        layers.append(get_layer("Linear")(input_dim, hidden1))
        layers.append(get_layer(opts.activation_layer)())
        layers.append(get_layer("Dropout")(opts.dropout_ratio))

        if self.n_conv_layers >= 2:
            hidden2 = int(opts.latent_dim*pow(opts.layer_ratio, opts.n_conv_layers-2))
            layers.append(get_layer("Linear")(hidden1, hidden2))
            layers.append(get_layer(opts.activation_layer)())
            layers.append(get_layer("Dropout")(opts.dropout_ratio))

        if self.n_conv_layers == 3:
            layers.append(get_layer("Linear")(hidden2, opts.latent_dim))
            layers.append(get_layer(opts.activation_layer)())
            layers.append(get_layer("Dropout")(opts.dropout_ratio))
            
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index):
        return self.nn(x)

class IdentityEncoder(torch.nn.Module):
    def __init__(self, input_dim, opts, num_nodes):
        """ Dummy encoder that does nothing """
        super(IdentityEncoder, self).__init__()
        opts.latent_dim = input_dim
        self.dummy_parameter = (torch.nn.Parameter(torch.rand((1,))))

    def forward(self, x, *args, **kwargs):
        return x