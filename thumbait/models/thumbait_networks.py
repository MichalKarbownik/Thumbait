import torch
from torch import nn
import torchvision.models as M
from enum import Enum
from logger import get_logger


logger = get_logger(__name__)


class ThumbLight(nn.Module):
    def __init__(self, kwargs, *args):
        super(ThumbLight, self).__init__()

        self.num_classes = kwargs.get("num_classes", 1)
        self.resnet_size = kwargs.get("resnet_size", 512)
        self.text_input_size = kwargs.get("input_size", 10000)
        self.hidden_sizes = kwargs.get("hidden_sizes", [256, 64])

        if self.resnet_size == 512:
            self.resnet = M.resnet18(pretrained=False)
        elif self.resnet_size == 2048:
            self.resnet = M.resnet50(pretrained=False)

        self.resnet.fc = torch.nn.Identity()

        self.fc_text = nn.Linear(self.text_input_size, self.hidden_sizes[0])

        self.fc_1 = nn.Linear(
            self.resnet_size + self.hidden_sizes[0], self.hidden_sizes[0]
        )  # fully connected 1
        self.fc_2 = nn.Linear(
            self.hidden_sizes[0], self.hidden_sizes[1]
        )  # fully connected 1

        self.fc_3 = nn.Linear(
            self.hidden_sizes[1], self.num_classes
        )  # fully connected last layer
        self.relu = nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, text, image):
        text = self.fc_text(text)
        # imgs = torch.cat([self.image_data_loader.dataset[img][0].unsqueeze(dim=0) for img in image]).cuda()
        image = self.resnet(image)

        out = torch.cat((text, image), axis=1)

        out = self.relu(out)
        out = self.fc_1(out)  # first Dense
        out = self.drop(out)
        out = self.relu(out)
        out = self.fc_2(out)  # first Dense
        out = self.drop(out)
        out = self.relu(out)  # relu
        out = self.fc_3(out)
        return out


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
