import math
from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
import sys
import os
import copy

from submodules.TimeSeriesDL.model.base_model import BaseModel
from data.dataset import Dataset
from utils.config import config
from utils.plotter import plot_temperatures, show_images


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
            test_config = copy.deepcopy(config_dict["dataset_args"])
            test_config["d_type"] = "test"
            testset = Dataset(**test_config)
            dl = DataLoader(testset)
            return dl

def prepare_model() -> BaseModel:    

    # load model flag
    load_flag = False if config_dict["evaluation"] == "None" else True
    log = config_dict["model_args"]["log"]
    config_dict["model_args"]["log"] = False if load_flag else log

    # create model
    model_name = config_dict["model_name"]
    model: BaseModel = config.get_model(model_name)(**config_dict["model_args"], precision=precision)
    model.use_device(device)

    # define log path in config and move the current hyperparameters to
    # this driectory
    if not load_flag:
        config_dict["evaluation"] = model.log_path
        config.store_args(f"{model.log_path}/config.yml", config_dict)
        print(f"Prepared model: {model_name}")
        return model
    
    path = config_dict["evaluation"]
    model_versions = []
    for file in os.listdir(path):
        if ".torch" in file:
            model_versions.append(f"{path}/{file}")
    model_versions.sort(reverse=True)

    print(model_versions[0])
    model.load(model_versions[0])

    print(f"Loaded model: {model_name} ({path})")
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
    model = prepare_model()

    real_temp = []
    pred_temp = []

    y = []
    for X, temp in test:
        # after predicting the first temperature, use it for future predictions
        if len(y) == config_dict["dataset_args"]["sequence_length"]:
            print(X[:, :, -1, 2], y)
            X[:, :, -1, 2] = torch.tensor([y])
            y += model.predict(X, as_list=True)
        else:
            _ = model.predict(X, as_list=True)
            y.append(temp.numpy()[0][0])
            
        if len(y) > config_dict["dataset_args"]["sequence_length"]:
            y.pop(0)
        
        real_temp.append(temp.numpy()[0])
        pred_temp.append(y[-1])

    plot_temperatures([pred_temp, real_temp], ["pred", "real"], size=(20, 10))
    path = config_dict["evaluation"]
    plt.savefig(f"{path}/temperature_city_4.png")

if __name__ == "__main__":
    freeze_support()
    
    config_path = sys.argv[1]
    config_dict = config.get_args(config_path)

    # perform some config checks
    ch_in = config_dict["model_args"]["ch_in"]
    seq_len = config_dict["dataset_args"]["sequence_length"]
    if ch_in != seq_len:
        raise RuntimeError(f"Input channel ({ch_in}) and sequence length ({seq_len}) must match.")
    
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float32# torch.float16 if device == "cuda" else torch.float32

    if config_dict["evaluation"] == "None":
        train()
        test()

    else:
        print("Evaluation of config:", config_dict["evaluation"])
        test()