from typing import Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
from submodules.TimeSeriesDL.model.base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, lr: float = 3e-3, lr_decay: float = 0.99, adam_betas: Tuple[float] = [0.9, 0.999],
                 log: bool = True) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/CNN_LSTM_Model/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(CNNModel, self).__init__()

        # TODO

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)
