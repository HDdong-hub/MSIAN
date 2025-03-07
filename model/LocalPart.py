from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import GCN


class SemCHGraphConv(nn.Module):
    """
    Semantic channel-wise graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=False):
        super(SemCHGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj.unsqueeze(0).repeat(out_features, 1, 1)
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(out_features, len(self.m[0].nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # input: (B, T, J, C)
        h0 = torch.matmul(input, self.W[0]).unsqueeze(2).transpose(2, 4)  # B * T * C * J * 1
        h1 = torch.matmul(input, self.W[1]).unsqueeze(2).transpose(2, 4)  # B * T * C * J * 1

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)  # C * J * J
        adj[self.m] = self.e.view(-1)
        adj = F.softmax(adj, dim=2)

        E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E = E.unsqueeze(0).repeat(self.out_features, 1, 1)  # C * J * J

        output = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
        output = output.transpose(2, 4).squeeze(2)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LocalPart(nn.Module):
    def __init__(self, adj, input_dim, output_dim, dropout=None):
        super(LocalPart, self).__init__()

        num_joints = adj.shape[0]

        if num_joints == 22:
            distal_joints = [3, 7, 11, 15, 16, 20, 21]
            joints_left = [0, 1, 2, 3, 17, 18, 19, 20, 21]
            joints_right = [4, 5, 6, 7, 12, 13, 14, 15, 16]

        elif num_joints == 18:
            distal_joints = [6, 7, 11, 16, 17]
            joints_left = [0, 3, 6, 9, 12, 14, 16]
            joints_right = [1, 4, 7, 10, 13, 15, 17]

        else:
            raise KeyError("The dimension of adj matrix is wrong!")

        adj_sym = torch.zeros_like(adj)
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    adj_sym[i][j] = 1
                if i in joints_left:
                    index = joints_left.index(i)
                    adj_sym[i][joints_right[index]] = 1.0
                if i in joints_right:
                    index = joints_right.index(i)
                    adj_sym[i][joints_left[index]] = 1.0

        adj_1st_order = adj.matrix_power(1)
        for i in np.arange(num_joints):
            if i in distal_joints:
                adj_1st_order[i] = 0

        adj_2nd_order = adj.matrix_power(2)
        for i in np.arange(num_joints):
            if i not in distal_joints:
                adj_2nd_order[i] = 0

        adj_3rd_order = adj.matrix_power(3)
        for i in np.arange(num_joints):
            if i not in distal_joints:
                adj_3rd_order[i] = 0

        adj_con = adj_1st_order + adj_2nd_order + adj_3rd_order

        self.gcn_sym = SemCHGraphConv(input_dim, output_dim, adj_sym)
        self.bn_1 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.gcn_con = SemCHGraphConv(input_dim, output_dim, adj_con)
        self.bn_2 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.relu = nn.ReLU()

        self.gcn_dis = GCN.GraphConvolution(input_dim, output_dim, node_n=num_joints)
        self.bn_3 = nn.BatchNorm2d(output_dim, momentum=0.1)

        self.cat_conv = nn.Conv2d(3 * output_dim, output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        x = self.gcn_sym(input)
        y = self.gcn_con(input)

        pos_1 = input.unsqueeze(-2)
        pos_2 = input.unsqueeze(-3)
        pos_rel = pos_1 - pos_2
        dis = torch.pow(pos_rel, 2).sum(2)
        dis = torch.sqrt(dis) + 1e-8
        joint_dis = self.gcn_dis(dis)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        z = joint_dis.permute(0, 3, 1, 2)

        x = self.relu(self.bn_1(x))
        y = self.relu(self.bn_2(y))
        z = self.relu(self.bn_3(z))

        output = torch.cat((x, y, z), dim=1)
        output = self.cat_bn(self.cat_conv(output))

        if self.dropout is not None:
            output = self.dropout(self.relu(output))
        else:
            output = self.relu(output)
        output = output.permute(0, 2, 3, 1)

        return output
