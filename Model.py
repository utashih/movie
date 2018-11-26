import torch
import torch.nn as nn

class MovieModel(nn.Module):
    def __init__(self, genres_idx, actor_idx, director_idx, company_idx):
        super(MovieModel, self).__init__()
        genres_dim = 32
        actors_dim = 32
        companies_dim = 16
        directors_dim = 16

        self.deepset_genres = DeepSetLayer(len(genres_idx), output_dim=genres_dim)
        self.deepset_actors = DeepSetLayer(len(actor_idx), output_dim=actors_dim)
        self.deepset_companies = DeepSetLayer(len(company_idx), output_dim=companies_dim)
        self.embedding = nn.Embedding(len(director_idx), directors_dim)
        self.fc1 = nn.Linear(2 + genres_dim + actors_dim + companies_dim + directors_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 8)
        self.fc6 = nn.Linear(8, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, budget, runtime, genres, actors, companies, director):
        genres_out = self.deepset_genres(genres)
        actors_out = self.deepset_actors(actors)
        companies_out = self.deepset_companies(companies)
        director_out = self.embedding(director).squeeze(1)
        encode = torch.cat([budget, runtime, genres_out, actors_out, companies_out, director_out], dim=1)
        out = self.tanh(self.fc1(encode))
        out = self.tanh(self.fc2(out))
        out = self.tanh(self.fc3(out))
        out = self.tanh(self.fc4(out))
        out = self.tanh(self.fc5(out))
        out = self.relu(self.fc6(out))

        return out



class DeepSetLayer(nn.Module):
    def __init__(self, vocab_size, output_dim = 32):
        super().__init__()
        self.phi = MoviePhi(vocab_size)
        self.rho = MovieRho(output_dim = output_dim)

    def forward(self, x):
        x = self.phi(x)
        x = torch.mean(x, dim=1, keepdim=False)
        out = self.rho(x)
        return out


class MoviePhi(nn.Module):
    def __init__(self, num_cast, output_dim = 64):
        super().__init__()
        self.num_cast = num_cast
        self.emb = nn.Embedding(self.num_cast, output_dim)
        self.fc1 = nn.Linear(output_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.emb(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        out = torch.tanh(x)
        return out


class MovieRho(nn.Module):
    def __init__(self, input_dim = 64, output_dim = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)

        out = self.fc2(x)
        return out


