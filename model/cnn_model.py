from argparse import ArgumentError
from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
from submodules.TimeSeriesDL.model.base_model import BaseModel


# IDEA: use a 3D filter which has the shape #cities x #features x 1
#       then visualize the filter after training, if a weight is close to 0, then
#       this feature is not relevant for predicting the temperature ahead


class CNNLSTMModel(BaseModel):
    def __init__(self, ch_in: int = 1, ch_out: int = 1, kernel_size: int = 5, stride: int = 1, padding: int = 0,
                 lr: float = 3e-3, lr_decay: float = 0.99, adam_betas: Tuple[float] = [0.9, 0.999],
                 lstm_hidden: int = 64, init_method: str = "zeros", precision: torch.dtype = torch.float32, 
                 out_act: str = "linear", log: bool = True) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/CNN_Model/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(CNNLSTMModel, self).__init__()

        self.__precision = precision
        self.__output_activation = out_act
        self.__init_method = init_method

        # CNN hyperparameters
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

        # input shape is:
        # sequence_length x n_cities x n_features
        self.__conv_1 = torch.nn.Conv2d(
            in_channels=ch_in, 
            out_channels=ch_out, 
            kernel_size=self.__kernel_size, 
            stride=self.__stride, 
            padding=self.__padding,
            dtype=self.__precision
        )

        # lstm hyperparameters
        self.__hidden_dim = lstm_hidden

        self.__lstm_1 = torch.nn.LSTMCell(
            1,
            self.__hidden_dim,
            dtype=self.__precision
        )

        # create the dense layers and initilize them based on our hyperparameters
        self.__linear_1 = torch.nn.Linear(
            self.__hidden_dim,
            32,
            dtype=self.__precision
        )

        self.__linear_2 = torch.nn.Linear(
            32,
            1,
            dtype=self.__precision
        )

        # initlize all weights (except of the LSTM ones)
        self._init_layers()

        # init loss function, optimizer and scheduler
        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    def _init_layers(self) -> None:
        match self.__init_method:
            case "xavier":
                self.__conv_1.weight = torch.nn.init.xavier_normal_(
                    self.__conv_1.weight)
                self.__linear_1.weight = torch.nn.init.xavier_normal_(
                    self.__linear_1.weight)
                self.__linear_2.weight = torch.nn.init.xavier_normal_(
                    self.__linear_2.weight)

            case "zeros":
                self.__conv_1.weight = torch.nn.init.zeros_(self.__conv_1.weight)
                self.__linear_1.weight = torch.nn.init.zeros_(self.__linear_1.weight)
                self.__linear_2.weight = torch.nn.init.zeros_(self.__linear_2.weight)

            case "ones":
                self.__conv_1.weight = torch.nn.init.ones_(self.__conv_1.weight)
                self.__linear_1.weight = torch.nn.init.ones_(self.__linear_1.weight)
                self.__linear_2.weight = torch.nn.init.ones_(self.__linear_2.weight)

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def get_filter(self) -> np.array:
        k_filter = np.array(self.__conv_1.weight.detach().numpy())
        k_filter = np.mean(k_filter, axis=(0, 1))
        return k_filter

    def forward(self, X) -> torch.tensor:
        batch_size, sequence_length, n_samples, features = X.shape

        x = self.__conv_1(X)
        x = x[:, :, 0]

        # pass data through CNN
        x = torch.relu(x)

        # reset LSTM's cell states
        h = torch.zeros(batch_size, self.__hidden_dim, dtype=self.__precision)
        c = torch.zeros(batch_size, self.__hidden_dim, dtype=self.__precision)

        # shape of x should be: (sequence_length, batch_size, 1)
        x = torch.swapaxes(x, 0, 1)

        # pass each timestep as batch into LSTM
        for i in range(x.size(0)):
            h, c = self.__lstm_1(x[i], (h, c))
        x = torch.relu(h)

        # forward pass the LSTM's output through a couple of dense layers
        x = self.__linear_1(x)
        x = torch.relu(x)
        
        # output from the last layer
        x = self.__linear_2(x)

        if self.__output_activation == "relu":
            output = torch.relu(x)
        elif self.__output_activation == "sigmoid":
            output = torch.sigmoid(x)
        elif self.__output_activation == "tanh":
            output = torch.tanh(x)
        elif self.__output_activation == "linear":
            output = x
        else:
            raise ArgumentError(
                "Wrong output actiavtion specified (relu | sigmoid | tanh).")

        return output
