from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
from model.LocalPart import LocalPart
from model.GlobalPart import MultiGlobalPart
from utils import graph_utils


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, node_n=48, is_ga=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.is_ga = is_ga
        if self.is_ga == True:
            self.weight_q = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features))
        self.I = torch.eye(self.weight.shape[0])

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        if self.is_ga == True:
            stdv = math.sqrt(6.0 / (self.weight_q.size(0) + self.weight_q.size(1)))
            self.weight_q.data.uniform_(-stdv, stdv)

    def forward(self, input, y0=None, layer_nums=0):

        if self.is_ga is True:
            hamilton = None
        else:
            hamilton = self.weight

        if layer_nums != 0:
            lamda, alpha = 1.5, 0.2
            beta = math.log(lamda / layer_nums + 1)
            support = (1 - alpha) * torch.matmul(self.att, input) + alpha * y0
            output = torch.matmul(support, beta * hamilton + (1 - beta) * self.I.to(self.weight.device))
        else:
            support = torch.matmul(input, hamilton)
            output = torch.matmul(self.att, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

adj = graph_utils.adj_mx_from_skeleton()

class IntegrationGraph(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout):
        super(IntegrationGraph, self).__init__()

        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)

        self.local_part = LocalPart(adj, input_dim, hid_dim, p_dropout)
        self.global_part = MultiGlobalPart(adj, input_dim, input_dim // 4, dropout=p_dropout)

        self.cat_conv = nn.Conv2d(3 * output_dim,  output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)

    def forward(self, x):
        b,  t = x.shape[0], x.shape[3]

        residual = x
        x_ = self.local_part(x)
        y_ = self.global_part(x)
        x = torch.cat((residual, x_, y_), dim=-1)

        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x)))
        x = x.permute(0, 2, 3, 1)

        return x.view(b, -1, t).contiguous()



class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.node_gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.node_bn1 = nn.BatchNorm1d(node_n * in_features)
        self.edge_gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.edge_bn1 = nn.BatchNorm1d(node_n * in_features)

        self.abs_gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.abs_bn1 = nn.BatchNorm1d(node_n * in_features)

        self.node_gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.node_bn2 = nn.BatchNorm1d(node_n * in_features)
        self.edge_gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.edge_bn2 = nn.BatchNorm1d(node_n * in_features)

        self.abs_gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.abs_bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fuse = nn.Conv1d(in_features, in_features, kernel_size=3, stride=3)
        self.fuse_bn = nn.BatchNorm1d(node_n * in_features)

        self.graph_att = IntegrationGraph(adj, in_features, in_features, p_dropout)


    def forward(self, x, node0=None, edge0=None, abs0=None, layer_nums=0):  #去掉node0，edge0    node0=None, edge0=None,

        b, n, t = x.shape
        x = x.view(b, 3, -1, t).contiguous()


        if n == 66:
            parents = [8, 0, 1, 2, 8, 4, 5, 6, 9, 8, 9, 10, 8, 12, 13, 14, 14, 8, 17, 18, 19, 19]
            parents_abs = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        elif n == 54:
            parents = [2, 2, 5, 0, 1, 5, 3, 4, 5, 5, 5, 8, 9, 10, 12, 13, 14, 15]
            parents_abs = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

        node = x.view(b, -1, t).contiguous()

        edge = x - x[:, :, parents, :]
        edge = edge.view(b, -1, t).contiguous()

        abs_pos = x - x[:, :, parents_abs, :]
        abs_pos = abs_pos.view(b, -1, t).contiguous()

        graph = self.graph_att(x)

        if layer_nums < 3:
            node = self.node_gc1(node)
            node = self.node_bn1(node.view(b, -1)).view(b, n, t).contiguous()
            node = self.act_f(node)
            node = self.do(node)
            node_init = node

            edge = self.edge_gc1(edge)
            edge = self.edge_bn1(edge.view(b, -1)).view(b, n, t).contiguous()
            edge = self.act_f(edge)
            edge = self.do(edge)
            edge_init = edge

            abs_pos = self.abs_gc1(abs_pos)
            abs_pos = self.abs_bn1(abs_pos.view(b, -1)).view(b, n, t).contiguous()
            abs_pos = self.act_f(abs_pos)
            abs_pos = self.do(abs_pos)
            abs_pos_init = abs_pos

            node = self.node_gc2(node, node_init, layer_nums + 1)
            node = self.node_bn2(node.view(b, -1)).view(b, n, t).contiguous()
            node = self.act_f(node)
            node = self.do(node)

            edge = self.edge_gc2(edge, edge_init, layer_nums + 1)
            edge = self.edge_bn2(edge.view(b, -1)).view(b, n, t).contiguous()
            edge = self.act_f(edge)
            edge = self.do(edge)

            abs_pos = self.abs_gc2(abs_pos, abs_pos_init, layer_nums + 1)
            abs_pos = self.abs_bn2(abs_pos.view(b, -1)).view(b, n, t).contiguous()
            abs_pos = self.act_f(abs_pos)
            abs_pos = self.do(abs_pos)

            node = node.view(b, 3, -1, t).contiguous()
            edge = edge.view(b, 3, -1, t).contiguous()
            abs_pos = abs_pos.view(b, 3, -1, t).contiguous()
            graph = graph.view(b, 3, -1, t).contiguous()

            b, c, v, t = x.shape

            fusion = torch.zeros((b, c, 3 * v, t)).to(x.device)
            node_idx = np.arange(0, 3 * v, 3)
            abs_idx = np.arange(1, 3 * v, 3)
            graph_idx = np.arange(2, 3 * v, 3)

            fusion[:, :, node_idx, :] = node
            fusion[:, :, abs_idx, :] = abs_pos
            fusion[:, :, graph_idx, :] = graph
            fusion = fusion.view(b, -1, t).permute(0, 2, 1).contiguous()
            fusion = self.fuse(fusion)
            fusion = self.fuse_bn(fusion.view(b, -1)).view(b, t, -1).permute(0, 2, 1).contiguous()

            if layer_nums == 0:
                return fusion + x.view(b, -1, t).contiguous(), node_init, edge_init, abs_pos_init
            else:
                return fusion + x.view(b, -1, t).contiguous()

        else:

            node = self.node_gc1(node, node0, layer_nums)
            node = self.node_bn1(node.view(b, -1)).view(b, n, t).contiguous()
            node = self.act_f(node)
            node = self.do(node)

            edge = self.edge_gc1(edge, edge0, layer_nums)
            edge = self.edge_bn1(edge.view(b, -1)).view(b, n, t).contiguous()
            edge = self.act_f(edge)
            edge = self.do(edge)


            abs_pos = self.abs_gc1(abs_pos, abs0, layer_nums)
            abs_pos = self.abs_bn1(abs_pos.view(b, -1)).view(b, n, t).contiguous()
            abs_pos = self.act_f(abs_pos)
            abs_pos = self.do(abs_pos)

            node = self.node_gc2(node, node0, layer_nums + 1)
            node = self.node_bn2(node.view(b, -1)).view(b, n, t).contiguous()
            node = self.act_f(node)
            node = self.do(node)

            edge = self.edge_gc2(edge, edge0, layer_nums + 1)
            edge = self.edge_bn2(edge.view(b, -1)).view(b, n, t).contiguous()
            edge = self.act_f(edge)
            edge = self.do(edge)

            abs_pos = self.abs_gc2(abs_pos, abs0, layer_nums + 1)
            abs_pos = self.abs_bn2(abs_pos.view(b, -1)).view(b, n, t).contiguous()
            abs_pos = self.act_f(abs_pos)
            abs_pos = self.do(abs_pos)

            node = node.view(b, 3, -1, t).contiguous()
            edge = edge.view(b, 3, -1, t).contiguous()
            abs_pos = abs_pos.view(b, 3, -1, t).contiguous()
            graph = graph.view(b, 3, -1, t).contiguous()

            b, c, v, t = x.shape

            fusion = torch.zeros((b, c, 3 * v, t)).to(x.device)
            node_idx = np.arange(0, 3 * v, 3)
            abs_idx = np.arange(1, 3 * v, 3)
            graph_idx = np.arange(2, 3 * v, 3)

            fusion[:, :, node_idx, :] = node
            fusion[:, :, abs_idx, :] = abs_pos
            fusion[:, :, graph_idx, :] = graph
            fusion = fusion.view(b, -1, t).permute(0, 2, 1).contiguous()
            fusion = self.fuse(fusion)
            fusion = self.fuse_bn(fusion.view(b, -1)).view(b, t, -1).permute(0, 2, 1).contiguous()

            return fusion + x.view(b, -1, t).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48, seq_len=40):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):

        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        node0, edge0, abs0 = None, None, None

        for i in range(self.num_stage):
            if i == 0:
                y, node0, edge0, abs0 = self.gcbs[i](y)
            else:
                y = self.gcbs[i](y, node0, edge0, abs0, 2*i)
        y = self.gc7(y)
        y = y + x

        return y
