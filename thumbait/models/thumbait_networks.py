import torch
from torch import nn
import torchvision.models as M
from enum import Enum
from logger import get_logger


logger = get_logger(__name__)


class Thumbait18(nn.Module):
    def __init__(
        self,
        image_data_loader,
        device,
        kwargs={
            "num_classes": 1,
            "input_size": 300,
            "hidden_size": 128,
            "num_layers": 2,
        },
    ):
        super(Thumbait18, self).__init__()

        self.num_classes = kwargs.get("num_classes", 1)
        self.input_size = kwargs.get("input_size", 100)
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_layers = kwargs.get("num_layers", 2)
        self.image_data_loader = image_data_loader
        self.device = device


        self.resnet = M.resnet18(pretrained=False)
        self.resnet.fc = torch.nn.Identity()

        self.lstm_title = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # lstm

        self.fc_1 = nn.Linear(
            512 + self.hidden_size, self.hidden_size
        )  # fully connected 1
        self.fc = nn.Linear(
            self.hidden_size, self.num_classes
        )  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, text, image):
        # Propagate input through LSTM
        output_title, (hn_title, cn_title) = self.lstm_title(
            text
        )  # lstm with input, hidden, and internal state

        hn_title = hn_title[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next
        imgs = torch.cat(
            [self.image_data_loader.dataset[img].unsqueeze(dim=0) for img in image]
        ).to(self.device)

        out_img = self.resnet(imgs)
        out = torch.cat((hn_title, out_img), axis=1)
        out = self.relu(out)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class Thumbait50(nn.Module):
    def __init__(
        self,
        image_data_loader,
        device,
        kwargs={
            "num_classes": 1,
            "input_size": 300,
            "hidden_size": 128,
            "num_layers": 2,
        },
    ):
        super(Thumbait50, self).__init__()

        self.num_classes = kwargs.get("num_classes", 1)
        self.input_size = kwargs.get("input_size", 100)
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_layers = kwargs.get("num_layers", 2)
        self.image_data_loader = image_data_loader
        self.device = device

        self.resnet = M.resnet50(pretrained=False)
        self.resnet.fc = torch.nn.Identity()

        self.lstm_title = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # lstm

        self.fc_1 = nn.Linear(
            2048 + self.hidden_size, self.hidden_size
        )  # fully connected 1
        self.fc = nn.Linear(
            self.hidden_size, self.num_classes
        )  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, text, image):
        # Propagate input through LSTM
        output_title, (hn_title, cn_title) = self.lstm_title(
            text
        )  # lstm with input, hidden, and internal state

        hn_title = hn_title[-1].view(
            -1, self.hidden_size
        )  # reshaping the data for Dense layer next
        imgs = torch.cat(
            [self.image_data_loader.dataset[img][0].unsqueeze(dim=0) for img in image]
        ).to(self.device)
        out_img = self.resnet(imgs)
        out = torch.cat((hn_title, out_img), axis=1)
        out = self.relu(out)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out
