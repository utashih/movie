import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import *
from MovieDataset import MovieDataset
from Model import MovieModel
import pandas as pd
import os.path as osp

def main():
    df = prepare_dataframe()
    # df.to_csv('the-movies-dataset/processed.csv')
    genres_idx, actor_idx, director_idx, company_idx = build_vocab(df)
    train_set = MovieDataset(df, genres_idx, actor_idx, director_idx, company_idx, train=True)
    test_set = MovieDataset(df, genres_idx, actor_idx, director_idx, company_idx, train=False)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=8)
    print("Training size: %d" % len(train_set))
    print("Testing  size: %d" % len(test_set))

    num_epoch = 10
    log_interval = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MovieModel(genres_idx, actor_idx, director_idx, company_idx)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, (budget, runtime, genres, actors, companies, director, revenue) in enumerate(train_loader):
            budget.to(device)
            runtime.to(device)
            genres.to(device)
            actors.to(device)
            companies.to(device)
            director.to(device)
            revenue.to(device)

            pred = model(budget, runtime, genres, actors, companies, director)
            loss = criterion(pred, revenue)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1) % log_interval == 0:
                print('[{:>2}, {:>4}] loss: {:.4}'.format(epoch+1, i+1, running_loss / log_interval))
                running_loss = 0

if __name__ == '__main__':
    main()