import os
import torch
import torch.nn as nn
from torchsummary import summary

"""
Authors of model: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

Paper: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class alexnet(nn.Module):
    def __init__(self, num_classes: int=1000)->None:
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),#Convolution
            nn.ReLU(),#Activation Function
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),#Normalisation
            nn.MaxPool2d(kernel_size=3, stride=2),#MaxPooling
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256 , out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384 , out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384 , out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)   
        )

        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = alexnet()
    print(model) 
    print("\nModel's Summary")
    summary(model, (3, 227, 227))

    print("\nDevice: ", device)
   