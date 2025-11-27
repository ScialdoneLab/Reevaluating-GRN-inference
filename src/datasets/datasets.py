from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as pyg_utils
from torch_geometric import seed_everything
from torch_geometric.transforms import RandomLinkSplit
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, minmax_scale
import torch
import numpy as np
import random

class GranPyDataset(InMemoryDataset):
    """ Abstract class that governs the joint behaviours of all datasets in granPy, such as datset naming an storage behaviour.
        Also governs the joint preprocessing details such as addition of inverse edges etc."""
    def __init__(self, root, opts, hash):
        self.root = root
        self.hash = hash
        self.val_seed = opts.val_seed
        self.canonical_test_seed = opts.canonical_test_seed
        self.test_fraction = opts.test_fraction
        self.val_fraction = opts.val_fraction
        self.undirected = opts.undirected
        self.eval_split = opts.eval_split
        self.sampling_power = opts.sampling_power
        super().__init__(root)

        self.train_data, self.val_data, self.test_data = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        pass
    
    @property
    def processed_file_names(self):
        return [self.hash + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        edgelist = self._data.edge_index
        features = self._data.x

        features = torch.FloatTensor(minmax_scale(features, axis=0))

        if pyg_utils.contains_self_loops(edgelist):
            edgelist = pyg_utils.remove_self_loops(edgelist)[0]

        data = Data(x=features,
                    edge_index=edgelist,
                    num_nodes=features.shape[0])

        train_data, val_data, test_data = self.split_data(data, self.test_fraction, self.val_fraction, self.canonical_test_seed, self.val_seed, self.undirected, self.eval_split)

        train_data.pot_net = self.construct_pot_net(train_data)
        val_data.pot_net = self.construct_pot_net(val_data)
        test_data.pot_net = self.construct_pot_net(test_data)

        torch.save([train_data, val_data, test_data], self.processed_paths[0])

    def split_data(self, data, test_fraction=0.2, val_fraction=0.2, test_seed=None, val_seed=None, undirected=False, eval_split="edges"):
        if eval_split == "nodes":
            return self.split_data_by_node(data, test_fraction=test_fraction, val_fraction=val_fraction, test_seed=test_seed, val_seed=val_seed, undirected=undirected, power=self.sampling_power)
        elif eval_split == "edges":
            return self.split_data_independent(data, test_fraction=test_fraction, val_fraction=val_fraction, test_seed=test_seed, val_seed=val_seed, undirected=undirected)
        elif eval_split == "genelink":
            return self.split_data_genelink(data, test_fraction=test_fraction, val_fraction=val_fraction, test_seed=test_seed, val_seed=val_seed, undirected=undirected)
        else:
            raise ValueError(f"Unknown eval split type: {eval_split}")

    @classmethod
    def split_data_by_node(self, data, test_fraction=0.2, val_fraction=0.2, test_seed=None, val_seed=None, undirected=False, power=-0.75):

        data = data.clone()
        
        if val_seed is None:
            val_seed = random.randint(0, 100)

        if test_seed is None:
            test_seed = random.randint(0, 100)

        # TODO check if this logic checks out with our assumptions
        if undirected:
            probabilities = torch.bincount(data.edge_index.flatten()).float()
        else:
            probabilities = torch.bincount(data.edge_index[0, :]).float()        

        
        probabilities /= probabilities.sum()
        probabilities = probabilities.pow(power)
        probabilities[probabilities.isinf()] = 0  # power operation can introduce inf values
        probabilities /= probabilities.sum()

        assert np.allclose(probabilities.sum(), 1)

        tolerance = 0.05

        trainval_edges, test_edges, sampled_nodes = self.sample_nodes(data.edge_index, probabilities, test_fraction, tolerance, seed=test_seed, sampled_nodes=[], undirected=undirected)

        train_edges, val_edges, _ = self.sample_nodes(data.edge_index, probabilities, val_fraction, tolerance, seed=val_seed, sampled_nodes=sampled_nodes, undirected=undirected)

        train_data = data.clone()
        train_data.edge_index = train_edges
        train_data.pos_edges = train_data.edge_index
        train_data.known_edges = train_data.edge_index
        train_data.known_edges_label = torch.ones(train_data.known_edges.shape[1])
        
        val_data = data.clone()
        val_data.edge_index = train_edges
        val_data.pos_edges = val_edges
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        val_data.known_edges_label = torch.hstack((torch.zeros_like(train_data.known_edges_label), torch.ones(val_data.pos_edges.shape[1])))
        
        test_data = data.clone()
        test_data.edge_index = torch.hstack((train_edges, val_edges))
        test_data.pos_edges = test_edges
        test_data.known_edges = torch.hstack((val_data.known_edges, test_data.pos_edges))
        test_data.known_edges_label = torch.hstack((torch.zeros_like(val_data.known_edges_label), torch.ones(test_data.pos_edges.shape[1])))

        return train_data, val_data, test_data

    def sample_nodes(edge_index, probabilities, sample_fraction, tolerance=0.05, seed=None, sampled_nodes=[], undirected=False):

        eligible_nodes = np.arange(probabilities.shape[0])
        nonzero_nodes = probabilities.nonzero().shape[0]
        max_num_test_edges = sample_fraction * edge_index.shape[1]
        upper_tolerance_threshold = (1 + tolerance) * max_num_test_edges
        lower_tolerance_threshold = (1 - tolerance) * max_num_test_edges
        initial_sampled_nodes = sampled_nodes[:]

        if seed is not None:
            np.random.seed(seed)

        test_subgraph = []
        num_sampled_test_edges = 0
        chunksize = max(min(100, int(nonzero_nodes / 10)), 1)
        patience = 3

        while not lower_tolerance_threshold < num_sampled_test_edges < upper_tolerance_threshold:
            if patience == 0: break
            chunk = np.random.choice(eligible_nodes, size=chunksize, replace=False, p=probabilities.numpy())
            try:
                chunk = [node for node in chunk.tolist() if node not in sampled_nodes]
            except TypeError:
                if chunk in sampled_nodes:
                    sampled_nodes.append(chunk)
                else:
                    chunk = [chunk]
            
            if len(chunk) == 0:
                continue
            test_subgraph.append(pyg_utils.k_hop_subgraph(chunk, 1, edge_index, relabel_nodes=False, flow="target_to_source", directed=not undirected)[1])

            num_sampled_test_edges = 0
            exceeded_flag = False
            for subgraph in test_subgraph: 
                num_sampled_test_edges += subgraph.shape[1]
                if num_sampled_test_edges > upper_tolerance_threshold:
                    test_subgraph = test_subgraph[:-1]
                    exceeded_flag = True
                    chunksize = int(chunksize/ 10)

            if not exceeded_flag:
                sampled_nodes.extend(chunk)  # if the sampled chunk did not exceed the number of edges, then add it to the sampled nodes
            
            if chunksize < 1:
                chunksize = 1
                patience -= 1 # if it fails to get a good sample even though we are already down to one sampled node

        newly_sampled_nodes = [node for node in sampled_nodes if node not in initial_sampled_nodes]

        _, test_edges, _, _ = pyg_utils.k_hop_subgraph(newly_sampled_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source", directed=not undirected)

        _, _, _, edge_mask = pyg_utils.k_hop_subgraph(sampled_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source", directed=not undirected)

        trainval_edges = edge_index[:, ~edge_mask]

        return trainval_edges, test_edges, sampled_nodes

    @classmethod
    def split_data_independent(self, data, test_fraction=0.2, val_fraction=0.2, test_seed=None, val_seed=None, undirected=False):

        data = data.clone()
        
        if val_seed is None:
            val_seed = random.randint(0, 100)

        if test_seed is None:
            test_seed = random.randint(0, 100)

        seed_everything(test_seed)


        traintest_split = RandomLinkSplit(num_val=0, num_test=test_fraction, is_undirected=undirected, add_negative_train_samples=False)

        trainval_data, _, test_data = traintest_split(data)

        real_val_frac = val_fraction / (1 - test_fraction)

        seed_everything(val_seed)

        trainval_split = RandomLinkSplit(num_val=real_val_frac, num_test=0, is_undirected=undirected, add_negative_train_samples=False)

        data.edge_index = trainval_data.edge_index

        train_data, val_data, _ = trainval_split(data)
        
        train_data.pos_edges = train_data.edge_index
        train_data.known_edges = train_data.edge_index
        train_data.known_edges_label = torch.ones(train_data.known_edges.shape[1])
        
        val_data.pos_edges = val_data.edge_label_index[:, val_data.edge_label == 1]
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        val_data.known_edges_label = torch.hstack((torch.zeros_like(train_data.known_edges_label), torch.ones(val_data.pos_edges.shape[1])))
        
        test_data.pos_edges = test_data.edge_label_index[:, test_data.edge_label == 1]
        test_data.known_edges = torch.hstack((val_data.known_edges, test_data.pos_edges))
        test_data.known_edges_label = torch.hstack((torch.zeros_like(val_data.known_edges_label), torch.ones(test_data.pos_edges.shape[1])))

        return train_data, val_data, test_data
    
    @classmethod
    def split_data_genelink(self, data, test_fraction=0.33, val_fraction=0.33, test_seed=None, val_seed=None, undirected=False):

        data = data.clone()
        
        if val_seed is None:
            val_seed = random.randint(0, 100)

        if test_seed is None:
            test_seed = random.randint(0, 100)

        seed_everything(test_seed)

        all_pos_edges = pyg_utils.coalesce(data.edge_index)
        tfs = all_pos_edges[0, :].unique()

        train_edges = []
        val_edges = []
        test_edges = []

        for tf in tfs:
            targets = all_pos_edges[1, all_pos_edges[0, :] == tf]
            num_targets = len(targets)
            if num_targets == 1:
                if random.random() < test_fraction:
                    test_edges.append((tf, targets[0]))
                else:
                    train_edges.append((tf, targets[0]))
            elif num_targets == 2:
                train_edges.append((tf, targets[0]))
                test_edges.append((tf, targets[1]))
            else:
                targets = targets[torch.randperm(num_targets)]
                num_val = int(num_targets * val_fraction)
                num_test = int(num_targets * test_fraction)
                num_train = num_targets - num_val - num_test
                train_edges.extend([(tf, target) for target in targets[:num_train]])
                val_edges.extend([(tf, target) for target in targets[num_train:num_train + num_val]])
                test_edges.extend([(tf, target) for target in targets[num_train + num_val:]])

        train_edges = torch.tensor(train_edges).t().contiguous()
        val_edges = torch.tensor(val_edges).t().contiguous()
        test_edges = torch.tensor(test_edges).t().contiguous()

        train_data = data.clone()
        val_data = data.clone()
        test_data = data.clone()

        train_data.edge_index = train_edges
        val_data.edge_index = train_edges
        test_data.edge_index = torch.hstack((train_edges, val_edges))

        train_data.pos_edges = train_edges
        train_data.known_edges = train_data.edge_index
        train_data.known_edges_label = torch.ones(train_data.known_edges.shape[1])
        
        val_data.pos_edges = val_edges
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        val_data.known_edges_label = torch.hstack((torch.zeros_like(train_data.known_edges_label), torch.ones(val_data.pos_edges.shape[1])))
        
        test_data.pos_edges = test_edges
        test_data.known_edges = torch.hstack((val_data.known_edges, test_data.pos_edges))
        test_data.known_edges_label = torch.hstack((torch.zeros_like(val_data.known_edges_label), torch.ones(test_data.pos_edges.shape[1])))

        return train_data, val_data, test_data
    

    @classmethod
    def construct_pot_net(self, data) -> torch.LongTensor:
        """ Constructs the potential network between transcription factors and targets by connecting each TF to each target
            careful, has memory complexity of n*m where n is the number of tfs and m is the number of targets

            Returns only negative edges and a tf mask by which the edges can be batched to their tfs.
            """

        all_pos_edges = pyg_utils.coalesce(data.known_edges)

        tfs, num_targets_per_tf = all_pos_edges[0, :].unique(return_counts=True)

        targets = torch.arange(data.num_nodes)

        tfs_repeated = tfs.repeat_interleave(repeats=data.num_nodes).unsqueeze(0)

        tf_batches = tfs.repeat_interleave(repeats=data.num_nodes)

        targets_tiled = targets.tile((tfs.shape[0],)).unsqueeze(0)

        all_pot_net_edges, tf_batches = pyg_utils.remove_self_loops(torch.vstack((tfs_repeated, targets_tiled)), edge_attr=tf_batches)

        mask = torch.cat((tf_batches + 1, -1 * torch.ones((all_pos_edges.shape[1],))))
        reduced_edges, neg_batch_mask = pyg_utils.coalesce(torch.hstack((all_pot_net_edges, all_pos_edges)), mask, reduce="mul")

        neg_mask = neg_batch_mask > 0
        neg_edges = reduced_edges[:, neg_mask]

        return neg_edges
    
    def to(self, device):
        # for data_name in ["train_data", "val_data", "test_data"]:
        #    self.__setattr__(data_name, self.__getattr__(data_name).to(device))
        self.train_data = self.train_data.to(device)
        self.test_data = self.test_data.to(device)
        self.val_data = self.val_data.to(device)


class McCallaDataset(GranPyDataset):
    """ Governs the download and preprocessing of the four datasets from McCalla et al. """
    def __init__(self, root, opts, hash, features=True):
        self.features = features
        self.name = opts.dataset

        self.name2edgelist = {"zhao": f"gold_standards/mESC/mESC_{opts.groundtruth}.txt",
                              "jackson": f"gold_standards/yeast/yeast_{opts.groundtruth}.txt",
                              "shalek": f"gold_standards/mDC/mDC_{opts.groundtruth}.txt",
                              "han": f"gold_standards/hESC/hESC_{opts.groundtruth}.txt",
                              "mccallatest": "edgelist.csv"}
        
        self.name2featuretable = {"zhao": "expression_data/normalized/zhao_GSE114952.csv.gz",
                                  "jackson": "expression_data/normalized/jackson_GSE125162.csv.gz",
                                  "shalek": "expression_data/normalized/shalek_GSE48968.csv.gz",
                                  "han": "expression_data/normalized/han_GSE107552.csv.gz",
                                  "mccallatest": "node_features.csv.gz"}
        
        if self.name not in self.name2edgelist.keys():
            raise ValueError("Only datasets for zhao, jackson, shalek and han et al implemented.")
        
        super().__init__(root, opts, hash)

    def download(self) -> None:
        from urllib.request import urlretrieve
        import zipfile

        filename = os.path.join(self.raw_dir, "gold_standard_datasets.zip")
        print("Downloading edgelist for {} from {} to {}".format(self.name, self.edgelisturl, filename))
        urlretrieve(self.edgelisturl, filename)

        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        if self.features:
            filename = os.path.join(self.raw_dir, "expression_data.zip")
            print("Downloading features for {} from {} to {}".format(self.name, self.featuretableurl, filename))
            urlretrieve(self.featuretableurl, filename)
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

    @property
    def raw_file_names(self):
        files = []
        files.append(self.name2edgelist[self.name.lower()])
        if self.features:
            files.append(self.name2featuretable[self.name.lower()])
        return files

    def process(self) -> None:

        print("NO CACHE FOUND - Processing data...")

        if self.features:
            features = self.read_features()

        edges = self.read_edgelist()

        if not self.features:
            features = torch.eye(edges.max() + 1)

        self._data = Data(x=features,
                          edge_index=edges)

        super().process()

    def read_features(self) -> torch.FloatTensor:
        feature_df = pd.read_csv(self.raw_paths[1], compression="gzip")
        self.geneencoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        feature_df["Cell"] = self.geneencoder.fit_transform(feature_df["Cell"].values.reshape(-1, 1)).flatten()

        feature_df.set_index("Cell", inplace=True, drop=True)
        feature_df.sort_index(inplace=True)

        return torch.FloatTensor(feature_df.values)

    def read_edgelist(self) -> torch.LongTensor:
        edge_df = pd.read_csv(self.raw_paths[0], header=None, index_col=None, sep="\t")

        old_shape = edge_df.values.shape

        if hasattr(self, "geneencoder"):
            encoded_edges = self.geneencoder.transform(edge_df.values.reshape(-1, 1)).reshape(old_shape)
            encoded_edges = encoded_edges[~np.isnan(encoded_edges).any(axis=1)]
        else:
            self.geneencoder = OrdinalEncoder()
            encoded_edges = self.geneencoder.fit_transform(edge_df.values.reshape(-1, 1)).reshape(old_shape)

        return torch.LongTensor(encoded_edges).T
    
    @classmethod
    @property
    def edgelisturl(self):
        return "https://zenodo.org/records/5909090/files/gold_standard_datasets.zip"
    
    @classmethod
    @property
    def featuretableurl(self):
        return "https://zenodo.org/records/5909090/files/expression_data.zip"
    
    @classmethod
    @property
    def names(self):
        return ["zhao", "jackson", "shalek", "han", "mccallatest"]


class DatasetBootstrapper:
    def __init__(self, opts, hash):
        import src.datasets as datasets

        self.hash = hash
        self.opts = opts
        name = opts.dataset
        names = []
        self.datasetclass = None

        for possible_class in map(datasets.__dict__.get, datasets.__all__):
            try:
                names.extend(possible_class.names)
                if name in possible_class.names:
                    if self.datasetclass is None:
                        self.datasetclass = possible_class
                    else:
                        raise ValueError("Found at least two implementations of dataset with name {}: {} and {}".format(name, possible_class, self.datasetclass))
            except AttributeError:
                continue

        if self.datasetclass is None:
            raise ValueError("Found no class that implements a dataset with the name {}. Did you mean one of the following: {}".format(name, names))

    def get_dataset(self):
        return self.datasetclass(root=self.opts.root, hash=self.hash, opts=self.opts)
