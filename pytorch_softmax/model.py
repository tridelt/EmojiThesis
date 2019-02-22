import torch
import torch.nn as nn
import torch.nn.functional as F

class Skipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim
        self.init_emb()
        
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)
        
    def forward(self, u_pos, v_pos, v_neg):
        embed_u = self.u_embeddings(u_pos)
        embed_v = self.v_embeddings(v_pos)
        neg_embed_v = self.v_embeddings(v_neg)
        
        #ploss = torch.mul(embed_u, embed_v).sum(dim=1).sigmoid().log()
        ploss = F.logsigmoid(torch.mul(embed_u, embed_v).sum(dim=1))
        
        #nloss = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze().sum(dim=1).neg().sigmoid().log()
        nloss = F.logsigmoid(torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze().sum(dim=1).neg())
        
        return -(ploss + nloss).mean()
    
    def input_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()









































