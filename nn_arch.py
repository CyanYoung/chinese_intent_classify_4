import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trm(nn.Module):
    def __init__(self, embed_mat, pos_mat, class_num, head, stack):
        super(Trm, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.pos = pos_mat
        self.head, self.stack = head, stack
        self.querys, self.keys, self.vals = [[[nn.Linear(embed_len, 200)] * head] * stack] * 3
        self.merges = [nn.Linear(200 * head, 200)] * stack
        self.lals = [nn.Sequential(nn.Linear(200, 200),
                                   nn.ReLU(),
                                   nn.Linear(200, 200))] * stack
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, class_num))

    def mul_att(self, i, x):
        c = list()
        for j in range(self.head):
            q, k, v = self.querys[i][j](x), self.keys[i][j](x), self.vals[i][j](x)
            scale = math.sqrt(k.size(-1))
            d = torch.matmul(q, k.permute(0, 2, 1)) / scale
            a = F.softmax(d, dim=-1)
            c_i = torch.matmul(a, v)
            c.append(c_i)
        x = torch.cat(c, dim=-1)
        return self.merges[i](x)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos
        for i in range(self.stack):
            r = x
            x = self.mul_att(i, x)
            x = F.layer_norm(x + r, x.size()[1:])
            r = x
            x = self.lals[i](x)
            x = F.layer_norm(x + r, x.size()[1:])
        x = x[:, -1, :]
        return self.dl(x)
