import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mul_att(layers, x, y):
    querys, keys, vals, fuse = layers
    c = list()
    for i in range(len(querys)):
        q, k, v = querys[i](y), keys[i](x), vals[i](x)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c_i = torch.matmul(a, v)
        c.append(c_i)
    x = torch.cat(c, dim=-1)
    return fuse(x)


class Att(nn.Module):
    def __init__(self, embed_mat, pos_mat, class_num, head, stack):
        super(Att, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.pos = pos_mat
        self.querys, self.keys, self.vals = [[[nn.Linear(embed_len, 200)] * head] * stack] * 3
        self.fuses = [nn.Linear(200 * head, 200)] * stack
        self.lals = [nn.Sequential(nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200))] * stack
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, class_num))

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos
        for i in range(len(self.querys)):
            r = x
            layers = [self.querys[i], self.keys[i], self.vals[i], self.fuses[i]]
            x = mul_att(layers, x, x)
            x = F.layer_norm(x + r, x.size()[1:])
            r = x
            x = self.lals[i](x)
            x = F.layer_norm(x + r, x.size()[1:])
        x = x[:, -1, :]
        return self.dl(x)
