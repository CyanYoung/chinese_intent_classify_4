import pickle as pk

import numpy as np

import torch
import torch.nn.functional as F

from preprocess import clean

from represent import sent2ind

from util import map_item


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


device = torch.device('cpu')

seq_len = 30

bos = '<'

path_word_ind = 'feat/word_ind.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'trm': 'model/trm.pkl'}

models = {'trm': torch.load(map_item('trm', paths), map_location=device)}


def predict(text, name):
    text = bos + text
    pad_seq = sent2ind(text, word_inds, seq_len, keep_oov=True)
    sent = torch.LongTensor([pad_seq]).to(device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = F.softmax(model(sent), dim=1)
    probs = probs.numpy()[0]
    sort_probs = sorted(probs, reverse=True)
    sort_inds = np.argsort(-probs)
    sort_preds = [ind_labels[ind] for ind in sort_inds]
    formats = list()
    for pred, prob in zip(sort_preds, sort_probs):
        formats.append('{} {:.3f}'.format(pred, prob))
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        text = clean(text)
        print('trm: %s' % predict(text, 'trm'))
