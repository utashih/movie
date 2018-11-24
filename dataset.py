#!/usr/bin/env python
import ast
import torch
import torch.utils.data
import pandas as pd
from IPython import embed

        
MAX_NUM_CAST = 10 



class MovieData(object):
    def __init__(self):
        movies_credit = {}
        movies_metadata = {}
        movies = {}

        credits = pd.read_csv('./the-movies-dataset/credits.csv')#('./credits_small.csv')#

        for movie in credits.itertuples(): 
            casts = ast.literal_eval(movie.cast)
            cast_ids = list(map(lambda cast: cast['id'], casts))
            movies_credit[movie.id] = {
                "cast": cast_ids
            }

        metadata = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False) #('./movies_metadata_small.csv', low_memory=False)#

        for movie in metadata.itertuples():
            try:
                if movie.revenue == 0: 
                    continue
                movies_metadata[int(movie.id)] = {
                    "title": movie.title,
                    "revenue": movie.revenue
                }
            except:
                continue

        for movie_id, movie in movies_metadata.items():
            if movie_id in movies_credit:
                movies[movie_id] = movie
                movies[movie_id]["cast"] = movies_credit[movie_id]["cast"]

        N = len(movies) 
        revenue = torch.zeros(N, dtype=torch.float32)
        cast = torch.zeros((N, MAX_NUM_CAST), dtype=torch.int64)
        for i, movie in enumerate(movies.values()):
            revenue[i] = movie["revenue"] / 10000000.0
            j = 0 
            for cid in movie["cast"]:
                if cid < 100000:
                    cast[i, j] = cid 
                    j += 1 
                    if j >= MAX_NUM_CAST:
                        break
            #for j in range(min(MAX_NUM_CAST, len(movie["cast"]))):
            #    cast[i, j] = movie["cast"][j]

        self.revenue = revenue
        self.cast = cast
        self.N = N
        self.num_cast = 1 + int(torch.max(cast))


class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, data, train=True):
        num_train = 7000
        if train:
            self.revenue = data.revenue[:num_train]
            self.cast = data.cast[:num_train]
        else:
            self.revenue = data.revenue[num_train:]
            self.cast = data.cast[num_train:]
        self.num_cast = data.num_cast 

    def __len__(self):
        return len(self.revenue)

    def __getitem__(self, index):
        cast = self.cast[index]
        revenue = self.revenue[index]
        return cast, revenue
