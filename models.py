import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, args, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embd_size)
        if args.pre_embd is not None:
            print('pre embedding weight is set')
            self.embedding.weight = nn.Parameter(args.pre_embd_w, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class QA_LSTM(nn.Module):
    def __init__(self, args):
        super(QA_LSTM, self).__init__()
        self.word_embd = WordEmbedding(args)
        self.lstm = nn.LSTM(args.embd_size, args.hidden_size, batch_first=True, bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)

    def reduce_mean(self, x):
        bs = x.size(0)
        x = x.contiguous()
        x = x.view(bs, -1) # (bs, L*2H)
        x = torch.mean(x, 1)
        x = x.unsqueeze(1) # (bs, 1)
        return x

    def forward(self, q, a):
        # embedding
        q = self.word_embd(q) # (bs, L, E)
        a = self.word_embd(a) # (bs, L, E)
        # LSTM
        q, _h = self.lstm(q) # (bs, L, 2H)
        a, _h = self.lstm(a) # (bs, L, 2H)

        # mean/maxpooling
        q = self.reduce_mean(q) # (bs,)
        a = self.reduce_mean(a) # (bs,)

        return self.cos(q, a)
