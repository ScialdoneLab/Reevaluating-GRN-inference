from src.experiment import ExperimentArray
import pandas as pd
from src.utils import opts
from src.experiment import Experiment
import wandb
import argparse
import dataclasses

class TrainingPipeline:
    def __init__(self, opts):
        self.experimentarray = ExperimentArray(opts)

    def run(self, eval_only=False):
        performance_reports = self.experimentarray.run(eval_only)

        metrics = list(performance_reports[0].keys())

        df = pd.DataFrame()

        for metric in metrics:
            values = []
            for run in performance_reports:
                values.append(run[metric])

            df[metric] = values

        df_t = df.transpose()
        df_t["mean"] = df_t.mean(axis=1)

        return df_t
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--n_conv_layers', type=int)
    parser.add_argument('--dropout_ratio', type=float)
    parser.add_argument('--mplayer', type=str)
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--layer_ratio', type=int)
    parser.add_argument('--decoder', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--es_patience', type=int)
    parser.add_argument('--val_mode', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--negative_sampling', type=str)
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--groundtruth', type=str)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--binarize_prediction', type=str, default="False")
    parser.add_argument('--unmask_topk', type=str, default="True")
    parser.add_argument('--fixed_t', type=str, default = "None")
    
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    args_dict["cuda"] = False if args_dict["cuda"] == "False" else True
    args_dict["binarize_prediction"] = False if args_dict["binarize_prediction"] == "False" else True
    args_dict["unmask_topk"] = False if args_dict["unmask_topk"] == "False" else True
    args_dict["fixed_t"] = None if args_dict["fixed_t"] == "None" else float(args_dict["fixed_t"])

    print(args_dict)

    return args_dict
    
if __name__ == "__main__":
    
    _opts = opts(**parse_args())
    if(_opts.wandb_tracking):
        run = wandb.init(project=_opts.wandb_project, entity="scialdonelab", save_code=True, group=_opts.wandb_group, 
                   config=wandb.helper.parse_config(dataclasses.asdict(_opts), exclude=('root', 'model_path', 'wandb_tracking', 'wandb_project', 'wandb_save_model', 'wandb_group')))
        #wandb.define_metric(f"val_{_opts.val_metric}", summary=_opts.val_mode)
    
    experiment = Experiment(_opts)
    experiment.run()
    
    if(_opts.wandb_tracking): 
        wandb.finish()