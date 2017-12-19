'''
    LSTM-based Deep Learning Models for Non-factoid Answer Selection
    Ming Tan, Cicero dos Santos, Bing Xiang, Bowen Zhou, ICLR 2016
    https://arxiv.org/abs/1511.04108
'''

import random
from tqdm import tqdm
import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors
from utils import load_data, load_data2, load_vocabulary, Config, load_embd_weights
from utils import make_vector
from models import QA_LSTM

PAD = '<PAD>'
id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
w2i = {w: i for i, w in enumerate(id_to_word.values(), 1)}
w2i[PAD] = 0
vocab_size = len(w2i)
print('vocab_size:', vocab_size)

train_data = load_data('./V2/InsuranceQA.question.anslabel.token.500.pool.solr.train.encoded', id_to_word, label_to_ans_text)
test_data = load_data2('./V2/InsuranceQA.question.anslabel.token.500.pool.solr.test.encoded', id_to_word, label_to_ans_text)
print('n_train:', len(train_data))
print('n_test:', len(test_data))

margin       = 0.2
embd_size    = 300
hidden_size  = 141
args = {
    'max_sent_len': 200,
    'margin'      : margin,
    'vocab_size'  : vocab_size,
    'embd_size'   : embd_size,
    'hidden_size' : hidden_size,
    'pre_embd'    : None
}
args = Config(**args)

print('loading a word2vec binary...')
model_path = './GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)
print('loaded!')
pre_embd = load_embd_weights(word2vec, vocab_size, args.embd_size, w2i)
# save_pickle(pre_embd, 'pre_embd.pickle')
args.pre_embd = pre_embd


def loss_fn(pos_sim, neg_sim):
    loss = args.margin - pos_sim + neg_sim
    if loss.data[0] < 0:
        loss.data[0] = 0
    return loss


def train(model, data, optimizer, n_epochs=4, batch_size=2):
    model.train()
    for epoch in range(n_epochs):
        print('epoch', epoch)
        random.shuffle(data) # TODO use idxies
        for i in tqdm(range(0, len(data)-batch_size, batch_size)):
            batch = data[i:i+batch_size]
            q_batch = [d[0] for d in batch]
            max_q_len = max([len(q) for q in q_batch])
            q_batch = make_vector(q_batch, w2i, max_q_len)

            pos_batch = [d[1] for d in batch]
            max_pos_len = min(args.max_sent_len, max([len(p) for p in pos_batch]))
            pos_batch = make_vector(pos_batch, w2i, max_pos_len)

            # sample one negative candidate until getting plus loss value or reaching max sampling count
            for neg_sampling_ct in range(50): # see 2.3
                neg_batch = [random.choice(d[2]) for d in batch] # choose one negative sample
                max_neg_len = min(args.max_sent_len, max([len(n) for n in neg_batch]))
                neg_batch = make_vector(neg_batch, w2i, max_neg_len)
                pos_sim = model(q_batch, pos_batch)
                neg_sim = model(q_batch, neg_batch)
                loss = loss_fn(pos_sim, neg_sim)
                if loss.data[0] != 0:
                    # print('loss:', loss.data[0])
                    # print('margin is less than m, lets update')
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, data):
    acc, total = 0, 0
    for d in data:
        q = d[0]
        print('q', ' '.join(q))
        labels = d[1]
        cands = d[2]

        # preprare answer labels
        label_indices = [cands.index(l) for l in labels if l in cands]

        # build data
        q = make_vector([q], w2i, len(q))
        cands = [label_to_ans_text[c] for c in cands] # id to text
        max_cand_len = min(args.max_sent_len, max([len(c) for c in cands]))
        cands = make_vector(cands, w2i, max_cand_len)

        # predict
        scores = [model(q, c.unsqueeze(0)).data[0] for c in cands]
        pred_idx = np.argmax(scores)
        if pred_idx in label_indices:
            print('correct')
            acc += 1
        else:
            print('wrong')
        total += 1
    print('Acc:', 100*acc/total)


model = QA_LSTM(args)
if torch.cuda.is_available():
    model.cuda()
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

train(model, train_data, optimizer)
test(model, test_data)
