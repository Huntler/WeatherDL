from statistics import mode
from submodules.TimeSeriesDL.utils.config import config
from model.cnn_model import CNNModel


config.register_model("CNN-Model", CNNModel)
