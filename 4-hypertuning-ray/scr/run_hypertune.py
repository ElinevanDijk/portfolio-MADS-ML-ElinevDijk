import os
import sys

# Add project root to Python path so scr/ can be imported
current = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current)               
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scr.data import get_dataloader
from scr.train import train_one_epoch, validate_one_epoch

import ray
from ray import tune
import torch
from model import CNN, ModelConfig
from pathlib import Path
import os
import json
import sys

def parse_tune_config(config_dict):
    parsed = {}
    for k, v in config_dict.items():
        if isinstance(v, dict):
            if "choice" in v:
                parsed[k] = tune.choice(v["choice"])
            elif "uniform" in v:
                low, high = v["uniform"]
                parsed[k] = tune.uniform(low, high)
            elif "loguniform" in v:
                low, high = v["loguniform"]
                parsed[k] = tune.loguniform(low, high)
            else:
                # Recursively parse nested dict
                parsed[k] = parse_tune_config(v)
        else:
            parsed[k] = v
    return parsed

def train_ray(config, checkpoint_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = ModelConfig(
        filters= config["filters"],
        units1= config["units1"],
        units2= config["units2"],
        dropout_dense_rate= config["dropout_dense_rate"],
        dropout_conv_rate= config["dropout_conv_rate"],
        use_batchnorm= config["use_batchnorm"],
        use_maxpooling= config["use_maxpooling"],
        num_classes= 2,
        num_blocks= config["num_blocks"]
    )

    model = CNN(model_config).to(device)

    train_loader, val_loader = get_dataloader(batch_size= 16)
    optimizer = torch.optim.Adam(model.parameters(), lr= config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    patience = config["patience"]
    max_epochs = config["epochs"]
    counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        tune.report({"patience": patience,
                     "Max_epochs": max_epochs,
                     "train_loss": train_loss, 
                     "train_accuracy": train_accuracy, 
                     "val_loss": val_loss, 
                     "val_accuracy": val_accuracy
                     })
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            counter = 0
            best_model_state = model.state_dict()

            checkpoint_dir = f"./checkpoint_epoch_{epoch}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(best_model_state, path)
        else:
            counter += 1
            if counter >= patience:
                print(f"[Early stop] No improvement for {patience} epochs. Stopping at epoch {epoch}")
                break
    
        

def short_trial_name(trial):
    return f"trial_{trial.trial_id}"

if __name__ == "__main__":
    if len(sys.argv) <2:
        print("Wrong, use: python run_hypertune.py config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        raw_config = json.load(f)

    config = parse_tune_config(raw_config)

    ray.init(runtime_env={"working_dir": "."}, num_cpus = 4)

    results_dir = Path("ray_results").resolve()
    storage_path = f"file:///{results_dir.as_posix()}"

    is_tune_search = any(
        isinstance(v, dict) and ("grid_search" in v or "choice" in v or "uniform" in v or "loguniform" in v)
        for v in raw_config.values()
    )

    if is_tune_search:
        analysis = tune.run(
            train_ray,
            resources_per_trial= {"cpu": 1, "gpu": 0},
            config= config,
            metric= "val_accuracy",
            mode= "max",
            num_samples= 40,
            trial_dirname_creator=short_trial_name,
            storage_path=storage_path

        )

        print("Best config found:", analysis.best_config)
    else:
        tune.run(
            train_ray,
            resources_per_trial= {"cpu": 1, "gpu": 0},
            config= config,
            metric= "val_accuracy",
            mode= "max",
            num_samples= 1,
            trial_dirname_creator=short_trial_name,
            storage_path=storage_path

        )



