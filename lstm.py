import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1,
                 dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        # x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size),
                                             strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output


class PriceChangeLSTM(object):

    def __init__(self, df, epochs=100, hidden_layer_size=32, num_layers=2, dropout=0.2,
                 lr=0.01):
        self.df = df
        self.model = LSTMModel(input_size=1, hidden_layer_size=hidden_layer_size,
                               num_layers=num_layers,
                               output_size=1, dropout=dropout)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98),
                                    eps=1e-9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40,
                                                   gamma=0.1)
        self.epochs = epochs

        data_change = (df.Close.to_numpy() - np.roll(df.Close.to_numpy(), 1))[1:]
        data_x, self.data_x_unseen = prepare_data_x(data_change, window_size=20)
        data_y = prepare_data_y(data_change, window_size=20)

        # split dataset
        train_split_size = 0.9
        split_index = int(data_y.shape[0] * train_split_size)
        data_x_train = data_x[:split_index]
        data_x_val = data_x[split_index:]
        self.data_y_train = data_y[:split_index]
        self.data_y_val = data_y[split_index:]
        self.data_x_train = np.expand_dims(data_x_train, 2)
        self.data_x_val = np.expand_dims(data_x_val, 2)

        dataset_train = TimeSeriesDataset(self.data_x_train, self.data_y_train)
        dataset_val = TimeSeriesDataset(self.data_x_val, self.data_y_val)

        self.train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
        self.val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=True)

    def run_epoch(self, dataloader, is_training=False):
        epoch_loss = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        for idx, (x, y) in enumerate(dataloader):
            if is_training:
                self.optimizer.zero_grad()

            batchsize = x.shape[0]

            out = self.model(x)
            loss = self.criterion(out.contiguous(), y.contiguous())

            if is_training:
                loss.backward()
                self.optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = self.scheduler.get_last_lr()[0]

        return epoch_loss, lr

    def train(self):
        log = []
        for epoch in range(self.epochs):
            loss_train, lr_train = self.run_epoch(self.train_dataloader,
                                                  is_training=True)
            loss_val, lr_val = self.run_epoch(self.val_dataloader)
            self.scheduler.step()
            log.append({
                "epoch": epoch + 1,
                "loss_train": loss_train,
                "loss_test": loss_val,
                "lr": lr_train,
            })
            print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                  .format(epoch + 1, self.epochs, loss_train, loss_val, lr_train))
        return log

    def evaluate(self):
        self.model.eval()

        predicted_val = np.array([])

        for idx, (x, y) in enumerate(self.val_dataloader):
            out = self.model(x)
            out = out.cpu().detach().numpy()
            predicted_val = np.concatenate((predicted_val, out))

        high_low = np.where(self.data_y_val > 0, 1, -1)[1:]
        high_low_pred = np.where(predicted_val > 0, 1, -1)[1:]
        cm = confusion_matrix(high_low, high_low_pred)
        cm = [[y.item() for y in x] for x in cm]

        return {
            "confusion_matrix": cm,
            "classification_report": classification_report(high_low, high_low_pred,
                                                           output_dict=True)
        }

    def predict_next_day(self):
        # predict the change for the next trading day
        self.model.eval()

        x = torch.tensor(self.data_x_unseen).float().unsqueeze(0).unsqueeze(2)
        # this is the data type and shape required, [batch, sequence, feature]
        prediction = self.model(x)
        prediction = prediction.cpu().detach()
        return prediction.tolist()[0]
