"""
Dataset loader class for minibatch gradient descent.
"""
import numpy as np
import torch
import pdb

class Dataset():

    def __init__(self,data_x,data_y):
        """
        Constructor

        Parameters:
            data_x: full path (plus file) to the x
            data_y: full path (plus file) to the y

        Returns:
            None
        """
        self.X = torch.tensor(np.load(data_x)[:,None,...]).to(torch.float32)
        y = np.load(data_y)[:,None,...]

        # Get to floating point and mean subtract
        y = y / 255
        y -= np.mean(y.ravel())
        self.Y = torch.tensor(y).to(torch.float32)

    def __getitem__(self,index):
        """
        Gets instance of the data, returns x,y pair.

        Parameters:
            index: index ...
        
        Returns:
            x,y pair
        """
        x = self.X[index,...]
        y = self.Y[index,...]

        return x,y

    def __len__(self):
        """ Returns number of samples in partition """
        return self.X.shape[0]