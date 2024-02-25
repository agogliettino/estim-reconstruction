import numpy as np
import torch.nn as nn
import torch
import pdb
import src.config as cfg

"""
Contains various CNN models for image denoising.
"""

class CAE(nn.Module):

    """
    CNN to enhance linear reconstructed image based on 
    "Neural Networks for Efficient Bayesian Decoding of Natural Images 
    from Retinal Neurons"
    """
    def __init__(self):
        super(CAE, self).__init__()

        """ Downsample """

        # Layer 1 
        self.conv0 = nn.Sequential(
                     nn.Conv2d(in_channels=1,out_channels=64,
                               kernel_size=7,stride=2,padding=3),
                     nn.BatchNorm2d(64),
                     nn.ReLU()
                     )
        
        # Layer 2
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=64,out_channels=128,
                               kernel_size=5,stride=2,padding=2),
                     nn.BatchNorm2d(128),
                     nn.ReLU(),
                     nn.Dropout2d(p=cfg.DROPOUT_PROB)
                     )

        # Layer 3
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=128,out_channels=256,
                               kernel_size=3,stride=2,padding=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(),
                     nn.Dropout2d(p=cfg.DROPOUT_PROB)
                     )

        # Layer 4 
        self.conv3 = nn.Sequential(
                     nn.Conv2d(in_channels=256,out_channels=256,
                               kernel_size=3,stride=2,padding=1),
                     nn.BatchNorm2d(256),
                     nn.ReLU(),
                     nn.Dropout2d(p=cfg.DROPOUT_PROB)
                     )

        """ Upsample """

        # Layer 5
        self.conv4 = nn.Sequential(
                     nn.Upsample(scale_factor=2),
                     nn.Conv2d(in_channels=256,out_channels=256,
                              kernel_size=3,stride=1,padding='same'),
                     nn.BatchNorm2d(256),
                     nn.ReLU(),
                     nn.Dropout(p=cfg.DROPOUT_PROB)
                     )

        # Layer 6 
        self.conv5 = nn.Sequential(
                     nn.Upsample(scale_factor=2),
                     nn.Conv2d(in_channels=256,out_channels=128,
                               kernel_size=3,stride=1,padding='same'),
                     nn.BatchNorm2d(128),
                     nn.ReLU(),
                     nn.Dropout(p=cfg.DROPOUT_PROB)
                     )

        # Layer 7
        self.conv6 = nn.Sequential(
                     nn.Upsample(scale_factor=2),
                     nn.Conv2d(in_channels=128,out_channels=64,
                               kernel_size=5,stride=1,padding='same'),
                     nn.BatchNorm2d(64),
                     nn.ReLU(),
                     nn.Dropout(p=cfg.DROPOUT_PROB)
                     )

        # Layer 8 (and out)
        self.conv7 = nn.Sequential(
                     nn.Upsample(scale_factor=2),
                     nn.Conv2d(in_channels=64,out_channels=1,
                               kernel_size=7,stride=1,padding='same'),
                     nn.BatchNorm2d(1),
                     )

    def forward(self,x):

        x1 = self.conv0(x)

        x2 = self.conv1(x1)

        x3 = self.conv2(x2)

        x4 = self.conv3(x3)

        x5 = self.conv4(x4)

        x6 = self.conv5(x5)

        x7 = self.conv6(x6)

        y = self.conv7(x7)

        return y