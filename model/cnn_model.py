from typing import Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
from submodules.TimeSeriesDL.model.base_model import BaseModel


# IDEA: use a 3D filter which has the shape #cities x #features x 1
#       then visualize the filter after training, if a weight is close to 0, then
#       this feature is not relevant for predicting the temperature ahead


class CNNModel(BaseModel):
    def __init__(self, kernel_size: int = 5, stride: int = 1, padding: int = 0,
                 lr: float = 3e-3, lr_decay: float = 0.99, adam_betas: Tuple[float] = [0.9, 0.999],
                 lstm_hidden: int = 64, xavier: bool = False, precision: torch.dtype = torch.float32, 
                 log: bool = True) -> None:
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
        super(CNNModel, self).__init__()

        self.__precision = precision
        self.__xavier = xavier

        # CNN hyperparameters
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding

        # input shape is:
        # sequence_length x n_cities x n_features
        self.__conv_1 = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
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
        if self.__xavier:
            self.__conv_1.weight = torch.nn.init.xavier_normal_(
                self.__conv_1.weight)
            self.__linear_1.weight = torch.nn.init.xavier_normal_(
                self.__linear_1.weight)
            self.__linear_2.weight = torch.nn.init.xavier_normal_(
                self.__linear_2.weight)
        else:
            self.__conv_1.weight = torch.nn.init.zeros_(self.__conv_1.weight)
            self.__linear_1.weight = torch.nn.init.zeros_(self.__linear_1.weight)
            self.__linear_2.weight = torch.nn.init.zeros_(self.__linear_2.weight)

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, X) -> torch.tensor:
        batch_size, sequence_length, n_samples, features = X.shape

        # pass data through CNN
        x = torch.empty(batch_size, sequence_length, 1)
        for i in range(sequence_length):
            _x = torch.unsqueeze(X[:, i], 1)
            _x = self.__conv_1(_x)
            x[:, i] = _x[:, 0, 0]

        x = torch.relu(x)

        # reset LSTM's cell states
        h = torch.zeros(batch_size, self.__hidden_dim, dtype=self.__precision)
        c = torch.zeros(batch_size, self.__hidden_dim, dtype=self.__precision)

        # shape of x should be: (sequence_length, batch_size, 1)
        x = torch.swapaxes(x, 0, 1)

        # pass each timestep as batch into LSTM
        for i in range(sequence_length):
            h, c = self.__lstm_1(x[i], (h, c))

        # feed the last output of the LSTM back into the LSTM to predict 
        # one step ahead
        x = torch.unsqueeze(h[:, -1], -1)
        h, c = self.__lstm_1(x, (h, c))
        x = torch.relu(h)

        # forward pass the LSTM's output through a couple of dense layers
        x = self.__linear_1(x)
        x = torch.relu(x)
        
        x = self.__linear_2(x)
        x = torch.relu(x)

        return x
