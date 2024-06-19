import os
import torch
import numpy
import random
from data_preprocessing import voraus_ad

HOME = os.getcwd()
DETERMINISTIC_CUDA = False

def get_voraus_ad(batch_size, parameters):
    features, frequency, subset_size = parameters
    print("------------------------------------")

    dataset_path = os.path.join(HOME, "dataset", "voraus-AD", "voraus-ad-dataset-100hz.parquet")
    configuration = voraus_ad.Configuration(columns=features, frequencyDivider=(100//frequency), trainGain=subset_size, seed=177, normalize=True, pad=True, scale=2)

    # Make the training reproducible.
    torch.manual_seed(configuration.seed)
    torch.manual_seed(configuration.seed)
    torch.cuda.manual_seed_all(configuration.seed)
    numpy.random.seed(configuration.seed)
    random.seed(configuration.seed)
    if DETERMINISTIC_CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader, test_loader = voraus_ad.load_torch_dataloaders(
        dataset             = dataset_path, 
        batch_size          = batch_size,
        columns             = voraus_ad.Signals.groups()[configuration.columns],
        seed                = configuration.seed,
        frequency_divider   = configuration.frequency_divider,
        train_gain          = configuration.train_gain,
        normalize           = configuration.normalize,
        pad                 = configuration.pad,
    )

    print("------------------------------------")
    return train_loader, test_loader