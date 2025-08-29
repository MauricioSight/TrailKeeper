import torch
from torch import nn, device as TorchDevice

from logging import Logger
from modeling.structure.pytorch_base import PytorchModelStructure


class CNN(PytorchModelStructure):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(CNN, self).__init__(config, logger, device)

        input_size_w    = config.get('modeling', {}).get('structure', {}).get('input_size_w')
        input_size_h    = config.get('modeling', {}).get('structure', {}).get('input_size_h')
        num_layers      = config.get('modeling', {}).get('structure', {}).get('num_layers')
        kernel_size     = config.get('modeling', {}).get('structure', {}).get('kernel_size')
        channels        = config.get('modeling', {}).get('structure', {}).get('channels')
        dropout         = config.get('modeling', {}).get('structure', {}).get('dropout')
        hidden_size     = config.get('modeling', {}).get('structure', {}).get('hidden_size')
        output_size     = config.get('modeling', {}).get('structure', {}).get('output_size')


        emb_size = channels * (input_size_w//(2**num_layers)) * (input_size_h//(2**num_layers))

        layers = []
        for i in range(num_layers):
            in_channels = 1 if i == 0 else channels
            layers += [
                nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.9),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ]
        self.feature_extraction_layer = nn.Sequential(*layers)

        self.binary_classification_layer_fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=emb_size, out_features=hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

        self.binary_classification_layer_fc2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature_extraction_layer(x)
        x = torch.flatten(x, 1)
        x = self.binary_classification_layer_fc1(x)
        x = self.binary_classification_layer_fc2(x)
        x = torch.sigmoid(x)
        return x
