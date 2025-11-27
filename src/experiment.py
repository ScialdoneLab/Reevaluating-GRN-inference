from src.datasets import DatasetBootstrapper
import src.nn.models as models
from src.utils import get_dataset_hash, get_model_hash, print_memory
from src.negative_sampling import neg_sampling
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import os
import sklearn.metrics as skmetrics
import copy
from tqdm import tqdm
import numpy as np
import wandb
import uuid

class Experiment:
    def __init__(self, opts):
        self.opts = opts

        self.model_hash = get_model_hash(opts)
        self.dataset_hash = get_dataset_hash(opts)
        
        self.device = self.devices[0]

        self.dataset = DatasetBootstrapper(opts, hash=self.dataset_hash).get_dataset()
        self.dataset.to(self.device)

        self.model = getattr(models, opts.model)(input_dim=self.dataset.train_data.x.shape[1], opts=opts, num_nodes=self.dataset.train_data.x.shape[0]).to(self.device)
        
        self.loss_function = BCEWithLogitsLoss()

        self.optimizer = Adam(self.model.parameters(), lr=opts.lr)
        self.lrscheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer)

        if self.opts.val_mode == "max":
            self.best_val_performance = -1e10
        elif self.opts.val_mode == "min":
            self.best_val_performance = 1e10
        else:
            raise ValueError("Validation Mode can either be max or min, but not {}".format(self.opts.val_mode))

        self.initial_patience = opts.es_patience
        self.patience = opts.es_patience
        
        self.binarize = opts.binarize_prediction

        self.test_performance = None

    def run(self, eval_only=False):
        
        self.cached_model = os.path.isfile(os.path.join(self.opts.model_path, self.model_hash + ".pt"))
        if self.opts.cache_model and self.cached_model:
            try: self.load_model()
            except FileNotFoundError:
                self.opts.cache_model = False
        if not (self.opts.cache_model and self.cached_model):
            if not eval_only:
                self.save_model()
                for epoch in (pbar := tqdm(range(self.opts.epochs))):
                    self.epoch = epoch
                    pbar.set_description("Best {}: {}".format(self.opts.val_metric, self.best_val_performance))
                    self.train_step()
                    
                    if self.opts.verbose:
                        print(f"epoch {epoch}: {print_memory(self.device)}")

                    if epoch % self.opts.eval_every == 0 and epoch > 0:
                        did_improve = self.eval_step(target="val")

                        if self.out_of_patience(did_improve):
                            break

            self.load_model()

        self.eval_step(target="test")

        if self.opts.wandb_tracking and (wandb.run is not None):
            wandb.log(self.test_performance)
            wandb.run.summary[f"val_{self.opts.val_metric}"] = self.best_val_performance
            if self.opts.wandb_save_model:
                self.log_model()
            
        if not self.opts.cache_model and not self.cached_model:
            self.delete_model()

        return self.test_performance

    def train_step(self):
        self.model.train()
        self.model.zero_grad()

        data = self.dataset.train_data

        loss, _, _ = self.get_loss(data, "train")

        if loss.requires_grad:

            loss.backward()

            self.optimizer.step()
        
        if self.opts.wandb_tracking: wandb.log({"train_loss": loss}, step=self.epoch)
        
        return loss

    @torch.no_grad
    def eval_step(self, target):
        self.model.eval()
        did_improve = False

        data = getattr(self.dataset, "{}_data".format(target))

        loss, pos_out, neg_out = self.get_loss(data, target)

        if target == "val":
            value = self.get_metric(pos_out, neg_out, [self.opts.val_metric])[self.opts.val_metric]
            if self.opts.wandb_tracking: wandb.log({("val_" + self.opts.val_metric): value, "val_loss": loss}, step=self.epoch)
            if self.is_best_val_performance(value):
                self.best_val_performance = value
                self.save_model()
                did_improve = True
            self.lrscheduler.step(loss)
            return did_improve
        elif target == "test":
            if self.opts.score_batched:
                self.test_performance = self.score_batched(self.dataset.test_data, self.opts.test_metrics)
            else:
                self.test_performance = self.get_metric(pos_out, neg_out, self.opts.test_metrics)
        return
    
    def is_best_val_performance(self, value):
        if self.opts.val_mode == "max":
            return self.best_val_performance < value
        else:
            return self.best_val_performance > value
    
    def get_loss(self, data, target="train"):
        neg_edges = self.get_negative_edges(data)
        

        z = self.model.encode(data.x, data.edge_index)
        pos_out = self.model.decode(z, data.pos_edges, pos_edge_index=data.edge_index)
        neg_out = self.model.decode(z, neg_edges, pos_edge_index=data.edge_index)
            
        if self.binarize and target != "train":
            pos_out = torch.bernoulli(pos_out)
            neg_out = torch.bernoulli(neg_out)

        pos_loss = self.loss_function(pos_out, torch.ones_like(pos_out))
        neg_loss = self.loss_function(neg_out, torch.zeros_like(neg_out))

        return pos_loss + neg_loss, pos_out, neg_out

    def save_model(self):
        state_dict = {
            'model_state_dict': self.model.state_dict(),
        }
        os.makedirs(os.path.dirname(self.opts.model_path), exist_ok=True)
        torch.save(state_dict, os.path.join(self.opts.model_path, self.model_hash + ".pt"))
       
    def delete_model(self):
        try:
            path = os.path.join(self.opts.model_path, self.model_hash + ".pt")
            os.remove(path)
        except: 
            print("Model could not be deleted - maybe it does not exist")
    
    def log_model(self):
        if wandb.run.sweep_id is not None:
            api = wandb.Api()
            sweep = api.sweep(f"scialdonelab/{self.opts.wandb_project}/sweeps/{wandb.run.sweep_id}")
            try: 
                best_score = sweep.best_run().summary[sweep.config["metric"]["name"]]
            except:
                best_score = 0
            if self.best_val_performance >= best_score:
                wandb.run.log_model(path=os.path.join(self.opts.model_path, self.model_hash + ".pt"))
        else:
            wandb.run.log_model(path=os.path.join(self.opts.model_path, self.model_hash + ".pt"))

    def load_model(self):
        '''loads a state dict either from the current run or from a given path'''
        path = os.path.join(self.opts.model_path, self.model_hash + ".pt")
        if torch.cuda.is_available():
            try:
                state_dict = torch.load(path)
            except RuntimeError:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict["model_state_dict"])

    def out_of_patience(self, did_improve: bool):
        if did_improve:
            self.patience = self.initial_patience
        else:
            self.patience -= 1

        return self.patience == 0

    def get_metric(self, pos_out: torch.Tensor, neg_out: torch.Tensor, metrics: list):
        all_out = torch.hstack((pos_out.squeeze(), neg_out.squeeze()))
        all_labels = torch.hstack((torch.ones_like(pos_out.squeeze()), torch.zeros_like(neg_out.squeeze())))

        values = {}
        for metric in metrics:
            function = getattr(skmetrics, metric)
            values.update({metric: function(y_true=all_labels.detach().cpu().numpy(), y_score=all_out.detach().cpu().numpy())})

        return values

    def get_negative_edges(self, data):
        if self.opts.negative_sampling.lower() == "structured_tail":
            return neg_sampling(data, space="pot_net", type="tail")
        elif self.opts.negative_sampling.lower() == "structured_head_or_tail":
            return neg_sampling(data, space="full", type="head_or_tail")
        elif self.opts.negative_sampling.lower() == "pot_net":
            return neg_sampling(data, space="pot_net", type="random")
        elif self.opts.negative_sampling.lower() == "random":
            return neg_sampling(data, space="full", type="random")
        elif self.opts.negative_sampling.lower() == "degree_aware":
            return neg_sampling(data, space="full", type="degree_aware")
        else:
            raise ValueError("Available Negative Sampling Types are structured_tail, structured_head_or_tail, pot_net, random and degree_aware. you have chosen {}".format(self.opts.negative_sampling))

    def score_batched(self, data, metrics):
        self.model.eval()
        self.model.zero_grad()

        z = self.model.encode(data.x, data.edge_index)

        pos_out = self.model.decode(z, data.pos_edges)
        neg_out = self.model.decode(z, data.pot_net)

        total_indices = torch.cat((data.pos_edges[0, :], data.pot_net[0, :])).unique().tolist()

        values = {metric: [] for metric in metrics}
        for tf in total_indices:
            pos = pos_out[data.pos_edges[0, :] == tf]
            neg = neg_out[data.pot_net[0, :] == tf]

            if len(pos) == 0 or len(neg) == 0:
                continue

            scores = torch.cat((pos, neg)).detach().cpu().numpy()
            labels = torch.cat((torch.ones_like(pos), torch.zeros_like(neg))).detach().cpu().numpy()

            for metric in metrics:
                function = getattr(skmetrics, metric)
                values[metric].append(function(y_true=labels, y_score=scores))

        return {metric: np.mean(list_of_values) for metric, list_of_values in values.items()}

    @property
    def devices(self):
        cuda = self.opts.cuda
        if cuda:
            try:
                assert torch.cuda.is_available()
            except AssertionError:
                if cuda == "auto":
                    print(
                        "CUDA set to auto, no CUDA device detected, setting to CPU")
                    devices = ["cpu"]
                    return devices
                else:
                    raise ValueError(
                        "Specified that job should be run on CUDA, but no CUDA devices are available. Aborting...")
            try:
                available_cuda_devices = ["cuda:{}".format(
                    device) for device in range(torch.cuda.device_count())]
                if cuda is True:
                    devices = available_cuda_devices
                elif cuda == "auto":
                    devices = available_cuda_devices
                elif isinstance(cuda, str):
                    devices = [cuda]
                elif isinstance(cuda, list):
                    devices = cuda

                for device in devices:
                    assert isinstance(device, str)
                    assert device.startswith("cuda:")
                    assert device in available_cuda_devices

            except AssertionError:
                raise ValueError("Specified cuda device(s) {} not in available cuda device(s): {}. Check Spelling or Numbering".format(
                    cuda, available_cuda_devices))
        else:
            print(
                "CUDA is set to {}, using cpu as fallback".format(cuda))
            devices = ["cpu"]

        return devices
    

class ExperimentArray:
    def __init__(self, opts):
        self.opts = opts
        self.opts.wandb_group = uuid.uuid4().hex

        self.val_seeds = range(opts.val_seed, opts.val_seed + opts.n_folds)
        self.performance_reports = None

    def run(self, eval_only=False):
        self.performance_reports = []
        for val_seed in tqdm(self.val_seeds):
            subopts = copy.deepcopy(self.opts)
            subopts.val_seed = val_seed
            fold_experiment = Experiment(subopts)
            self.performance_reports.append(fold_experiment.run(eval_only))

        return self.performance_reports
        