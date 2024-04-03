# -*- coding:utf-8 -*-
# author: tiger
# datetime:2023/9/6 14:14
# software: PyCharm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))

        #xavier初始化参数
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.Q = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.Q.data, gain=1.414)
        self.V = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.V.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)#F*(F,F')=F'
        q = torch.mm(input, self.Q)
        v = torch.mm(input, self.V)
        N = h.size()[0]#F'

        #torch.cat() 拼接张量，后面这串都是注意力机制原理，看不懂啊
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), q.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #输出加权注意力
        attention = F.dropout(attention, self.dropout, training=self.training)
        #矩阵相乘
        h_prime = torch.matmul(attention, v)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATRegress(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GATRegress, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        #torch.Size([3, 6])
        #x = F.dropout(x, self.dropout, training=self.training)
        #torch.Size([3, 6])
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #torch.Size([3, 70])
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #torch.Size([3, 6])
        x = x.flatten()
        return x

class CNNWithPooling(nn.Module):
    def __init__(self):
        super(CNNWithPooling, self).__init__()
        # 添加卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=[1,2])

    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.pool(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super(BiLSTM, self).__init__()
        # LSTM层：input_size: 每个x的特征个数，hidden_size:隐藏层输出的维度， num_layer:lstm单元的个数
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.lstm = nn.LSTM(input_sz, hidden_sz, bidirectional=True)


    def forward(self, X,init_states=None):
        batch_size = X.shape[0]
        the_input = X.transpose(0, 1)
        if init_states is None:
            h_t, c_t = (torch.randn(1*2, batch_size, self.hidden_size,device=X.device),
                        torch.randn(1*2, batch_size, self.hidden_size,device=X.device))
        else:
            h_t, c_t = init_states

        the_output, (_, _) = self.lstm(the_input, (h_t, c_t))
        return the_output, (_, _)

# class GATLSTM(nn.Module):
#     def __init__(self, n_nodes, n_features, n_hid, n_output, dropout, alpha, n_heads):
#         super(GATLSTM, self).__init__()
class MM(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.gcns = nn.ModuleList([GATRegress(6, 18, 6, 0.1, 0.1, 10) for _ in range(t)])
        self.lstm = BiLSTM(96,18*16)#96,
        self.linear = nn.Linear(int(18*5/6*16*2*2), 1)#960
        self.cov1 = CNNWithPooling()#

    def forward(self, x, adj,state=None):  # t,n
        x_gcns = [i(x[j,:],adj) for j,i in enumerate(self.gcns)]  # t*n
        #5*3*6
        x_gcns=torch.stack(x_gcns)
        # torch.Size([5, 18])

        x_gcns = x_gcns.view(5,1, 3,-1)
        # torch.Size([5, 1,3,6])
        xx_gcns = self.cov1(x_gcns)
        #torch.Size([5, 16,2,3])
        xx_gcns = xx_gcns.view(1,5, -1)
        #torch.Size([1,5,96])

        
        # 对第二维度进行倒置
        flipped_tensor = torch.flip(xx_gcns, dims=[1])

        # 将原始张量和倒置后的张量连接起来
        xx_gcns = torch.cat((xx_gcns, flipped_tensor), dim=1)
        # torch.Size([1, 10, 96])

        output, (hn, cn) = self.lstm(xx_gcns,state)
        # torch.Size([1, 10, 576])

        xx=output.view(6,-1)
        #torch.Size([6, 960])
        m = self.linear(xx)

        return m, (hn, cn)


