import copy
import random
import torch
import torch.nn as nn
from gensim.models.keyedvectors import KeyedVectors
from utils import save_pickle, load_pickle, load_data, load_vocabulary, Config, load_embd_weights, to_var
from utils import make_vector, padding
from models import QA_LSTM

PAD = '<PAD>'
id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
w2i = {w: i for i, w in enumerate(id_to_word.values(), 1)}
w2i[PAD] = 0
vocab_size = len(w2i)
print('vocab_size', vocab_size)

data = load_data('./V2/InsuranceQA.question.anslabel.token.500.pool.solr.train.encoded', id_to_word, label_to_ans_text)

embd_size = 300
hidden_size = 100
args = {
    'vocab_size': vocab_size,
    'embd_size': embd_size,
    'hidden_size': hidden_size,
    'pre_embd': None
}
args = Config(**args)

# print('loading a word2vec binary...')
# model_path = './GoogleNews-vectors-negative300.bin'
# word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
# print('loaded!')
# pre_embd = load_embd_weights(word2vec, vocab_size, args.embd_size, w2i)
# save_pickle(pre_embd, 'pre_embd.pickle')
# args.pre_embd = pre_embd
cos = nn.CosineSimilarity(dim=1)
def loss_fn(pos_sim, neg_sim):
    loss = 2.0 - pos_sim + neg_sim
    if loss.data[0] < 0:
        loss.data[0] = 0
    return loss

def train(model, data, optimizer, n_epochs=10, batch_size=32):
    # data = copy.deepcopy(data)
    print('len(data)', len(data))
    for epoch in range(n_epochs):
        random.shuffle(data)
        for i in range(0, len(data)-batch_size, batch_size):
            batch = data[i:i+batch_size]
            q = [d[0] for d in batch]
            max_q_len = max([len(qq) for qq in q])
            q = make_vector(q, w2i, max_q_len)
            p = [d[1] for d in batch]
            n = [d[2] for d in batch]
            max_ans_len = max(max([len(pp) for pp in p]), max([len(nn) for nn in n]))
            p = make_vector(p, w2i, max_ans_len)
            n = make_vector(n, w2i, max_ans_len)
            pos_sim = model(q, p)
            neg_sim = model(q, n)
            loss = loss_fn(pos_sim, neg_sim)
            print('loss:', loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


model = QA_LSTM(args)
if torch.cuda.is_available():
    model.cuda()
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train(model, data[:10000], optimizer)
