from statistics import mode
from submodules.TimeSeriesDL.utils.config import config
from model.cnn_model import CNNLSTMModel


config.register_model("CNN-LSTM-Model", CNNLSTMModel)
