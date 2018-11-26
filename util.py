import pandas as pd
import numpy as np
import torch
import json
import ast
import os.path as osp

def filter_director(crews):
    # return [crew for crew in crews if crew['job'] == 'Director']
    for crew in crews:
        if crew['job'] == 'Director':
            return crew

def filter_director(crews):
    # return [crew for crew in crews if crew['job'] == 'Director']
    for crew in crews:
        if crew['job'] == 'Director':
            return crew
def filter_director(crews):
    # return [crew for crew in crews if crew['job'] == 'Director']
    for crew in crews:
        if crew['job'] == 'Director':
            return crew
def prepare_dataframe():
    PATH = 'the-movies-dataset'
    credits = pd.read_csv(osp.join(PATH, 'credits.csv'), dtype={'id': int}, na_values = '[]')
    metadata = pd.read_csv(osp.join(PATH, 'movies_metadata.csv'), na_values = '[]',
                           usecols=['budget', 'genres', 'id', 'popularity', 'production_companies',
                                    'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count'])
    keywords = pd.read_csv(osp.join(PATH, 'keywords.csv'))
    metadata['id'] = pd.to_numeric(metadata['id'], errors='coerce', downcast="integer")
    metadata['budget'] = pd.to_numeric(metadata['budget'], errors='coerce')
    metadata['revenue'] = pd.to_numeric(metadata['revenue'], errors='coerce')
    metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce')
    metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce')
    metadata['popularity'] = pd.to_numeric(metadata['popularity'], errors='coerce')
    metadata['runtime'] = pd.to_numeric(metadata['runtime'], errors='coerce')
    metadata.dropna(inplace=True)
    metadata = metadata[metadata.revenue > 10000]
    credits.dropna(inplace=True)

    df = metadata.set_index('id').join(credits.set_index('id'), on='id', how='inner')
    df = df.join(keywords.set_index('id'), on='id', how='left')
    df.cast = df.cast.apply(ast.literal_eval)
    df.crew = df.crew.apply(ast.literal_eval)
    df.genres = df.genres.apply(ast.literal_eval)
    df.keywords = df.keywords.apply(ast.literal_eval)
    df.production_companies = df.production_companies.apply(ast.literal_eval)
    df.budget[df.budget == 0] = df.budget[df.budget != 0].mean()
    df['director'] = df.crew.apply(filter_director)
    df.dropna(subset=['director'], inplace=True)
    df = df.reset_index()

    return df

def build_vocab(df):
    genres_idx = {}
    actor_idx = {}
    director_idx = {}
    company_idx = {}

    for genres in df.genres:
        for genre in genres:
            if genre['id'] not in genres_idx:
                genres_idx[genre['id']] = len(genres_idx)

    for cast in df.cast:
        for actor in cast:
            if actor['id'] not in actor_idx:
                actor_idx[actor['id']] = len(actor_idx)

    for director in df.director:
        if director['id'] not in director_idx:
            director_idx[director['id']] = len(director_idx)

    for companies in df.production_companies:
        for company in companies:
            if company['id'] not in company_idx:
                company_idx[company['id']] = len(company_idx)

    return genres_idx, actor_idx, director_idx, company_idx
