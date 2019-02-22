import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t

from torch import FloatTensor as FT


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
#        initializing the weights of embeddings
        self.u_embeddings.weight = nn.Parameter(t.cat([t.zeros(1, embedding_dim), FT(vocab_size - 1, embedding_dim).uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)]))
        self.v_embeddings.weight = nn.Parameter(t.cat([t.zeros(1, embedding_dim), FT(vocab_size - 1, embedding_dim).uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)]))
        
        self.loss = nn.CrossEntropyLoss()
        self.embedding_dim = embedding_dim
        
    def forward(self, u_pos, v_pos):
        embed_u = self.u_embeddings(u_pos)

        return self.loss(t.matmul(embed_u, t.t(self.v_embeddings.weight.data)), v_pos)
    
    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()



