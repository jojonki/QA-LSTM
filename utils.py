import pickle
import numpy as np
import torch
from torch.autograd import Variable


PAD = '<PAD>' # TODO


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)


def load_vocabulary(vocab_path, label_path):
    id_to_word = {}
    with open(vocab_path) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            if d[0] not in id_to_word:
                id_to_word[d[0]] = d[1]

    label_to_ans = {}
    label_to_ans_text = {}
    with open(label_path) as f:
        lines = f.readlines()
        for l in lines:
            label, answer = l.rstrip().split('\t')
            if label not in label_to_ans:
                label_to_ans[label] = answer
                label_to_ans_text[label] = [id_to_word[t] for t in answer.split(' ')]
    return id_to_word, label_to_ans, label_to_ans_text


# def load_data_elmwise(fpath, id_to_word, label_to_ans_text):
#     data = []
#     with open(fpath) as f:
#         lines = f.readlines()
#         for l in lines:
#             d = l.rstrip().split('\t')
#             q = [id_to_word[t] for t in d[1].split(' ')] # question
#             poss = [label_to_ans_text[t] for t in d[2].split(' ')] # ground-truth
#             negs = [label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]] # candidate-pool without ground-truth
#             for pos in poss:
#                 for neg in negs:
#                     data.append((q, pos, neg))
#     return data


def load_data(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            poss = [label_to_ans_text[t] for t in d[2].split(' ')] # ground-truth
            negs = [label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]] # candidate-pool without ground-truth
            for pos in poss:
                data.append((q, pos, negs))
    return data


def load_data2(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines[12:]:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            # poss = [label_to_ans_text[t] for t in d[2].split(' ')] # ground-truth
            # cands = [label_to_ans_text[t] for t in d[3].split(' ')] # candidate-pool
            poss = [t for t in d[2].split(' ')] # ground-truth
            cands = [t for t in d[3].split(' ')] # candidate-pool
            data.append((q, poss, cands))
    return data


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def load_embd_weights(word2vec, vocab_size, embd_size, w2i):
    embedding_matrix = np.zeros((vocab_size, embd_size))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, idx in w2i.items():
        # words not found in embedding index will be all-zeros.
        if word in word2vec.wv:
            embedding_matrix[idx] = word2vec.wv[word]
            found_ct += 1
    print(found_ct, 'words are found in word2vec. vocab_size is', vocab_size)
    return torch.from_numpy(embedding_matrix).type(torch.FloatTensor)


def padding(data, max_sent_len, pad_token):
    pad_len = max(0, max_sent_len - len(data))
    data += [pad_token] * pad_len
    data = data[:max_sent_len]
    return data


def make_vector(data, w2i, seq_len):
    ret_data = [padding([w2i[w] for w in d], seq_len, w2i[PAD]) for d in data]
    return to_var(torch.LongTensor(ret_data))


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
