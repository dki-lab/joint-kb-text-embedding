import torch
import torch.nn as nn
import torch.nn.functional as F

class skipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, window, negative):
        super(skipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.window = window
        self.negative = negative
        self.embed_dim = embed_dim

        self.emb0_lookup = nn.Embedding(self.vocab_size, self.embed_dim, sparse=True)
        self.emb1_lookup = nn.Embedding(self.vocab_size, self.embed_dim, sparse=True)
        self.emb0_lookup.weight.data.uniform_(-0.5/self.embed_dim, 0.5/self.embed_dim)
        #self.emb0_lookup.weight.data[args.vocab_size].fill_(0)
        self.emb1_lookup.weight.data.zero_()
        self.window = self.window
        self.negative = self.negative
        #self.use_cuda = args.cuda
        self.pad_idx = self.vocab_size

    def forward(self, data):
        word_idx = data[:, 0]
        ctx_idx = data[:, 1]
        neg_indices = data[:, 2:2+self.negative]
        neg_mask = data[:, 2+self.negative:].float()

        w_embs = self.emb0_lookup(word_idx)
        c_embs = self.emb1_lookup(ctx_idx)
        n_embs = self.emb1_lookup(neg_indices)

        # print('w_embs = {}'.format(w_embs))
        # print('c_embs = {}'.format(c_embs))
        # print('c_embs = {}'.format(c_embs))

        pos_ips = torch.sum(w_embs * c_embs, 1)
        neg_ips = torch.bmm(n_embs, torch.unsqueeze(w_embs,1).permute(0,2,1))[:,:,0]
        neg_ips = neg_ips * neg_mask

        # print('pos_ips = {}'.format(pos_ips))
        # print('neg_ips = {}'.format(neg_ips))
        
        # Neg Log Likelihood
        pos_loss = torch.sum( -F.logsigmoid(pos_ips) )
        neg_loss = torch.sum( -F.logsigmoid(-neg_ips) )

        return pos_loss, neg_loss
