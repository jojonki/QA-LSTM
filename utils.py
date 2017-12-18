import pickle
import numpy as np
import torch


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


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
