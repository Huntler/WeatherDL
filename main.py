import math
from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
import sys

from submodules.TimeSeriesDL.model.base_model import BaseModel
from data.dataset import Dataset
from utils.config import config
from utils.plotter import show_images


config_dict = None
device = "cpu"
precision = torch.float32


def prepare_data(mode: str):
    match mode:
        case "train":
            dataset = Dataset(**config_dict["dataset_args"])
            split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]
            
            trainset, valset = torch.utils.data.random_split(dataset, split_sizes)
            dl_1 = DataLoader(trainset, **config_dict["dataloader_args"])
            dl_2 = DataLoader(valset, **config_dict["dataloader_args"])
            return dl_1, dl_2

        case "test":
            # TODO
            return None

def prepare_model() -> BaseModel:    
    # create model
    model_name = config_dict["model_name"]
    model: BaseModel = config.get_model(model_name)(**config_dict["model_args"])

    # define log path in config and move the current hyperparameters to
    # this driectory
    if config_dict["evaluation"] is None:
        config_dict["evaluation"] = model.log_path
        config.store_args(f"{model.log_path}/config.yml", config_dict)

    print(f"Prepared model: {model.log_path}")
    return model

def train():
    # prepare data
    train, val = prepare_data(mode="train")
    test = prepare_data(mode="test")
    model = prepare_model()

    # train model, save kernel before and afterwards to show differences
    untrained_kernel = model.get_filter()
    model.learn(train, val, test, epochs=config_dict["train_epochs"])
    trained_kernel = model.get_filter()

    # save model and kernel images
    model.save_to_default()
    show_images([untrained_kernel, trained_kernel], ["untrained", "trained"], size=(6, 3))
    plt.savefig(f"{model.log_path}/kernel_diff.png")

def test():
    test = prepare_data(mode="test")
    # TODO

if __name__ == "__main__":
    freeze_support()
    
    config_path = sys.argv[1]
    config_dict = config.get_args(config_path)
    
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    if config_dict["evaluation"] == "None":
        train()
        test()

    else:
        print("Evaluation of config:", config_dict["evaluation"])
        test()