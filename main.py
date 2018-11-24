#!/usr/bin/env python
import torch
import torch.utils.data
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from IPython import embed

from dataset import MovieData, MovieDataset
from deepset import DeepSet, MoviePhi, MovieRho

class DeepCasts(object):
    def __init__(self, lr=1e-3):
        self.lr = lr 
        self.data = MovieData()
        self.num_cast = self.data.num_cast

        self.train_set = MovieDataset(self.data, True)
        self.test_set = MovieDataset(self.data, False)
        print("Training size: %d" % len(self.train_set))
        print("Testing  size: %d" % len(self.test_set))

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=8, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=8, shuffle=True)

        self.phi = MoviePhi(self.num_cast)
        self.rho = MovieRho()
        self.model = DeepSet(self.phi, self.rho)
        if torch.cuda.is_available():
            self.model.cuda() 
        
        self.criterion = F.mse_loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epoch(self, epoch, log_interval=20):
        self.model.train()
        running_loss = 0.0 

        for i, (x, t) in enumerate(self.train_loader, 1):
            if torch.cuda.is_available():
                x, t = x.cuda(), t.cuda() 
            
            self.optimizer.zero_grad()
            pred = self.model(x) 
            loss = self.criterion(pred, t) 
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % log_interval == 0:
                print('Training [{:2d}, {:5d}]: loss: {:.3f}'.format(
                        epoch, i, running_loss / log_interval))
                running_loss = 0.0 

    def test(self):
        self.model.eval()
        test_loss = 0.0 
        with torch.no_grad():
            for x, t in self.test_loader:
                if torch.cuda.is_available():
                    x, t = x.cuda(), t.cuda() 
                pred = self.model(x)
                test_loss += self.criterion(pred, t, reduction='sum').item()
        size = len(self.test_loader.dataset)
        test_loss /= size 
        print('Test: loss: {:.3f}\n'.format(test_loss)) 
        return test_loss 
        


if __name__ == '__main__':
    deepCasts = DeepCasts()
    for i in range(1, 10):
        deepCasts.train_epoch(i)
        deepCasts.test()
