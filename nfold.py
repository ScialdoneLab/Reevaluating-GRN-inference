import pandas as pd
from src.utils import opts
import wandb
import dataclasses
import copy
from src.experiment import Experiment
import argparse

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True, help='Wandb run ID to use for 10-fold validation')
    parser.add_argument('--missing_only', action='store_true', help='Only run missing val_seeds (skip check if not set)')
    parser.add_argument('--fold', type=int, default=10, help='Number of folds for validation (default: 10)')
    args = parser.parse_args()
    
    api = wandb.Api()
    top_run = api.run(f"scialdonelab/GRN_inference/{args.id}")
        
    _opts = copy.deepcopy(top_run.config)
    val_seeds = range(_opts["val_seed"] + 1, _opts["val_seed"] + 1 + args.fold)
    
    for val_seed in val_seeds:
        

        if args.missing_only:
            skip_this_seed = False
            existing_runs = api.runs("scialdonelab/GRN_inference", {"group": f"10fold_run_{top_run.id}"})
            for run in existing_runs:

                run_val_seed = run.config.get("val_seed", None)
                if run_val_seed == val_seed:

                    roc_auc = run.summary.get("roc_auc_score", None)
                    if roc_auc is not None:
                        skip_this_seed = True
                        break
            if skip_this_seed:
                print(f"Skipping val_seed {val_seed} as a completed run already exists.")
                continue
        try:
            _opts["val_seed"] = val_seed

            run = wandb.init(project="GRN_inference", entity="scialdonelab", save_code=True, group=f'10fold_run_{top_run.id}', 
                    config=_opts)
        
            experiment = Experiment(opts(**_opts))
            experiment.run()

        except Exception as e:
            print(f"Error occurred for val_seed {val_seed}: {str(e)}")
            
        finally:
            wandb.finish()