from typing import List, Any

import torch
import torch.utils.data
import pandas as pd


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, df, genres_idx, actor_idx, director_idx, company_idx, train=True):
        num_train = int(len(df) * 0.9)
        if train:
            self.df = df[:num_train]
        else:
            self.df = df[num_train:]
        self.genres_idx = genres_idx
        self.actor_idx = actor_idx
        self.director_idx = director_idx
        self.company_idx = company_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        movie = self.df.iloc[index]
        budget = torch.tensor([movie.budget], dtype=torch.float32)
        runtime = torch.tensor([movie.runtime], dtype=torch.float32)
        director = torch.tensor([self.director_idx[movie.director['id']]], dtype = torch.int64)

        genres = []
        actors = []
        companies = []
        for genre in movie.genres:
            genres.append(self.genres_idx[genre['id']])
        for actor in movie.cast:
            actors.append(self.actor_idx[actor['id']])
            if len(actors) > 10:
                break
        for company in movie.production_companies:
            companies.append(self.company_idx[company['id']])
        genres = torch.tensor(genres, dtype = torch.int64)
        actors = torch.tensor(actors, dtype = torch.int64)
        companies = torch.tensor(companies, dtype = torch.int64)

        revenue = torch.tensor([movie.revenue], dtype=torch.float32)

        return budget, runtime, genres, actors, companies, director, revenue
