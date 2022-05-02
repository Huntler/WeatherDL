from statistics import mode
from model.pure_cnn_model import PURECNNModel
from submodules.TimeSeriesDL.utils.config import config
from model.cnn_model import CNNLSTMModel


config.register_model("CNN-LSTM-Model", CNNLSTMModel)
config.register_model("PURE-CNN-Model", PURECNNModel)
