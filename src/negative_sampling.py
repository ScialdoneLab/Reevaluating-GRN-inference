from torch_geometric.utils import negative_sampling, coalesce
from torch_geometric.utils import structured_negative_sampling as str_neg_sampling
from torch_geometric.utils import structured_negative_sampling_feasible as str_negative_sampling_feasible
import torch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def neg_sampling(data, space="full", type="degree_aware"):
    import random
    
    assert(space in ["full", "pot_net"])
    assert(type in ["tail", "head_or_tail", "random", "degree_aware"])
    
    #randomly sample edges from full space or pot_net
    if type == "random":
        if(space == "pot_net"):
            try:
                sample_indices = random.sample(range(data.pot_net.shape[1]), data.pos_edges.shape[1])
            except ValueError:
                # in case our negative set is smaller than the positive set, which can happen for test and val
                sample_indices = range(data.pot_net.shape[1])
            return data.pot_net[:, sample_indices]
        else:
            return negative_sampling(data.known_edges, num_nodes=data.num_nodes, num_neg_samples=data.pos_edges.shape[1]) 

    elif type == "degree_aware":
        # match negative tail nodes in-degree distribution to positive targets while perturbing only the tail and avoiding existing positives and self-loops
        pos_mask = data.known_edges_label == 1
        pos_edges = data.known_edges[:, pos_mask]
        pos_sources = pos_edges[0]
        pos_targets = pos_edges[1]

        # build global in-degree distribution over positive targets
        base_weights = torch.bincount(pos_targets, minlength=data.num_nodes).float()
        # avoid division by zero in case there are no positives
        base_weights = base_weights / base_weights.sum().clamp_min(1e-12)
        base_weights = base_weights.to(pos_sources.device)

        sampled_tails = []
        # for each positive source, sample a new tail according to masked base weights
        for i in range(pos_sources.shape[0]):
            s = int(pos_sources[i].item())
            weights = base_weights.clone()
            # disallow self-loop
            weights[s] = 0.0
            # disallow existing positive neighbors of s
            neighbor_mask = (data.known_edges[0, :] == s) & pos_mask
            if torch.any(neighbor_mask):
                neighbors = data.known_edges[1, neighbor_mask]
                weights[neighbors] = 0.0
            total = float(weights.sum().item())
            if total <= 0.0:
                # fallback: uniform over allowed nodes
                weights = torch.ones_like(base_weights)
                weights[s] = 0.0
                if torch.any(neighbor_mask):
                    weights[neighbors] = 0.0
                total = float(weights.sum().item())
                if total <= 0.0:
                    # as a last resort (degenerate fully-connected positive neighborhood)
                    # allow all except self
                    weights = torch.ones_like(base_weights)
                    weights[s] = 0.0
                    total = float(weights.sum().item())
            weights = weights / max(total, 1e-12)
            tail = torch.multinomial(weights, 1)[0]
            sampled_tails.append(tail)

        sampled_tails = torch.stack(sampled_tails)
        return torch.vstack((pos_sources, sampled_tails))

    elif type in ["tail", "head_or_tail"]:
        assert(str_negative_sampling_feasible(data.known_edges, num_nodes=data.num_nodes,
                                  contains_neg_self_loops=False))
        
        # perturb tail node first
        result = str_neg_sampling(data.known_edges, num_nodes=data.num_nodes,
                                  contains_neg_self_loops=False)
        
        tail_perturbed = torch.vstack((result[0][data.known_edges_label == 1], result[2][data.known_edges_label == 1]))
        
        if(type == "tail"): 
            return tail_perturbed
        
        # perturb head node now by switching, perturbing tail node and switching again
        inv_edges = torch.vstack((data.known_edges[1, :], data.known_edges[0, :]))
        
        assert(str_negative_sampling_feasible(inv_edges, num_nodes=data.num_nodes, contains_neg_self_loops=False))
        result = str_neg_sampling(inv_edges, num_nodes=data.num_nodes, contains_neg_self_loops=False)
        
        head_perturbed = torch.vstack((result[2][data.known_edges_label == 1], result[0][data.known_edges_label == 1]))

        result = coalesce(torch.hstack((tail_perturbed, head_perturbed)))

        # downsample so we have same amount as pos edges
        sample_indices = random.sample(range(result.shape[1]), data.pos_edges.shape[1])
        sample_indices = torch.LongTensor(sample_indices)
        
        return result[:, sample_indices]


def inspect_degree_bias(pos_edges: torch.LongTensor, neg_edges: torch.LongTensor, plot: bool = True, title: str = ""):
    edges = pos_edges.numpy().transpose()

    # add dummy edges so nx doesnt relabel
    edges = np.vstack((edges, np.asarray(
        (np.arange(0, max(pos_edges.max(), neg_edges.max())), 
        np.arange(0, max(pos_edges.max(), neg_edges.max())))
        ).transpose()
        ))

    edge_df = pd.DataFrame(edges, columns=["source", "target"])

    train_graph = nx.from_pandas_edgelist(edge_df, create_using=nx.DiGraph)
    train_in_degrees = {key: val -1 for key, val in train_graph.in_degree()}  # remove one degree bc we added self-loop dummy edges
    train_out_degrees = {key: val - 1 for key, val in train_graph.out_degree()}   # remove one degree bc we added self-loop dummy edges

    y_pos_out = [train_out_degrees[node.item()] for node in pos_edges[0, :]]

    y_neg_out = [train_out_degrees[node.item()] for node in neg_edges[0, :]]    
    print("Mean out_degree positive edges: {}\nMean out_degree negative edges: {}".format(np.mean(y_pos_out), np.mean(y_neg_out)))

    y_pos_in = [train_in_degrees[node.item()] for node in pos_edges[1, :]]

    y_neg_in = [train_in_degrees[node.item()] for node in neg_edges[1, :]]

    print("Mean in_degree positive edges: {}\nMean in_degree negative edges: {}".format(np.mean(y_pos_in), np.mean(y_neg_in)))

    if plot:
        return plot_degree_bias(y_pos_in, y_neg_in, y_pos_out, y_neg_out, title)
     

def plot_degree_bias(in_pos, in_neg, out_pos, out_neg, title =""):
    fig, axs = plt.subplots(1,2)
    fig.suptitle(title)

    axs[0].set_title('In Degree')
    axs[0].set_xlabel('Degree')
    axs[0].set_ylabel('Frequency')
    axs[1].set_title('Out Degree')
    axs[1].set_xlabel('Degree')
    axs[1].set_ylabel('Frequency')

    scatter_alpha = 0.4
    marker_size = 20

    for pos, neg, ax in zip((in_pos, out_pos), (in_neg, out_neg), axs):

        deg_vals_pos, deg_freq_pos = np.unique(pos, return_counts=True)
        deg_vals_neg, deg_freq_neg = np.unique(neg, return_counts=True)

        # --- Include 0 on the x-axis for in-degree ---
        min_deg = min(
            np.min(deg_vals_pos) if len(deg_vals_pos) > 0 else np.inf,
            np.min(deg_vals_neg) if len(deg_vals_neg) > 0 else np.inf,
            0
        )
        if min_deg > 0:
            # Insert a zero value with frequency 0
            deg_vals_pos = np.insert(deg_vals_pos, 0, 0)
            deg_freq_pos = np.insert(deg_freq_pos, 0, 0)
            deg_vals_neg = np.insert(deg_vals_neg, 0, 0)
            deg_freq_neg = np.insert(deg_freq_neg, 0, 0)
        elif min_deg < 0:  # Unlikely, but safe
            pass
        # If degree=0 does actually exist, it's already present

        ax.scatter(deg_vals_pos, deg_freq_pos, alpha=scatter_alpha, s=marker_size, color='g', label='positive')
        ax.scatter(deg_vals_neg, deg_freq_neg, alpha=scatter_alpha, s=marker_size, color='b', label='negative')
        # Vertical lines for mean degree
        mean_in_pos = np.mean(pos)
        mean_in_neg = np.mean(neg)
        ax.axvline(mean_in_pos, color='g', linestyle='--', label='pos mean')
        ax.axvline(mean_in_neg, color='b', linestyle='--', label='neg mean')
        

        ax.set_xscale('symlog', linthresh=1)  # symlog so that 0 can appear
        ax.set_yscale('log')
        # Set x-axis to start just below zero, e.g., -1
        ax.set_xlim(left=-0.1, right=1500)

        # Make sure that 0 is labelled explicitly on the axis
        xticks = ax.get_xticks()
        if not np.any(np.isclose(xticks, 0)):
            ax.set_xticks(np.insert(xticks, 0, 0))
        else: 
            ax.set_xticks(xticks)  # we do that so mpl doesnt whine wen we set the xticklabels later
        # Optionally: set ticklabel "0"
        xlabels = [str(int(x)) if x != 0 else "0" for x in ax.get_xticks()]
        ax.set_xticklabels(xlabels)
        ax.legend(loc="lower left")

    plt.tight_layout()

    return fig, axs

