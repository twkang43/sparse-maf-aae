import os
import shutil
import json
from types import SimpleNamespace
import argparse
from data_preprocessing import get_voraus_ad

import torch
from execution import Execution
from utils import set_device

HOME = os.getcwd()
DEVICE = set_device.set_device()

def main(args, config):
    print(f"Current cuda device : {DEVICE}")

    # Get Data & Extract a dimension of feature space
    train_loader, test_loader = get_voraus_ad.get_voraus_ad(config.batch_size, (args.features, args.frequency, args.subset_size))
    save_name = f"voraus-AD-{args.features}-{args.frequency}hz-{args.subset_size}_{str(config.epochs)}_{str(config.window_size)}"

    # Select a model
    Model = Execution.AAE(args, config, (train_loader, test_loader))
    is_valid_exec = False
        
    # Train a model
    if args.exec == "train" or args.exec == "all":
        print("---------- Train Mode ----------")
        is_valid_exec = True
        Model.train()

        # Save a model
        if not os.path.exists("model_save"):
            os.makedirs("model_save")
        os.chdir("model_save")
        model_name = f"{save_name}.pth"
        description_name = f"{save_name}.json"

        if os.path.exists(save_name):
            shutil.rmtree(save_name)
        os.mkdir(save_name)
        os.chdir(save_name)
        torch.save(Model.generator.state_dict(), model_name)

        hyperparameters = {
                            "features"         : args.features,
                            "frequency"        : args.frequency,
                            "subset_size"      : args.subset_size,
                            "batch_size"       : config.batch_size,
                            "window_size"      : config.window_size, 
                            "win_stride"       : config.win_stride,
                          }
        with open(description_name, 'w') as f:
            json.dump(hyperparameters, f, ensure_ascii=False, indent=4)
        os.chdir(HOME)

    # Evaluate a model
    if args.exec == "eval" or args.exec == "all":
        print("---------- Evaluation Mode ----------")
        is_valid_exec = True

        # Load a model
        if not os.path.exists("model_save"):
            raise Exception("Path 'model_save' does not exists")
        os.chdir("model_save")
        os.chdir(save_name)

        model_name = save_name + ".pth"
        Model.generator.load_state_dict(torch.load(model_name))
        os.chdir(HOME)

        Model.eval()
        
    if is_valid_exec == False:
        raise Exception("Invalid Execution Mode")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse MAF-AAE")

    parser.add_argument("--exec", help="execution mode", type=str, default="all", choices=["train", "eval", "all"])
    parser.add_argument("--features", help="Sampling features", type=str, default="torques", choices=["encoders", "torques", "currents", "machine"])
    parser.add_argument("--frequency", help="Sampling frequency", type=int, default=50, choices=[10, 25, 50, 100])
    parser.add_argument("--subset_size", help="Subset size for training", type=float, default=1.0, choices=[1.0, 0.5, 0.25, 0.125])

    args = parser.parse_args()

    # Load configurations of models
    config_path = os.path.join(HOME, "execution", "configuration.json")
    with open(config_path, 'r') as file:
        configs = json.load(file)
        config_name = f"voraus-AD-{args.features}-{args.frequency}hz-{args.subset_size}"
        dataset_config = configs.get(config_name)
    config = SimpleNamespace(**dataset_config) if dataset_config else None

    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")

    for key in config.__dict__:
        print(f"{key} : {getattr(config, key)}")
    print("------------------------------------")

    main(args, config)