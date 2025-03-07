from __future__ import absolute_import, division

import torch
from torch import nn


class GlobalPart(nn.Module):
    def __init__(self, adj, in_channels, inter_channels=None):
        super(GlobalPart, self).__init__()

        self.adj = adj
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

        if self.inter_channels == self.in_channels // 2:
            self.g_channels = self.in_channels
        else:
            self.g_channels = self.inter_channels

        assert self.inter_channels > 0

        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.g_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        adj_shape = self.adj.shape
        self.C_k = nn.Parameter(torch.zeros(adj_shape, dtype=torch.float))

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.g_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)
        phi_x = phi_x.expand(-1, -1, h, -1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        attention = self.leakyrelu(f.view(b, h, w))

        attention = torch.add(self.softmax(attention), self.C_k)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.g_channels, *x.size()[2:])

        return y


class MultiGlobalPart(nn.Module):
    def __init__(self, adj, in_channels, inter_channels, dropout=None):
        super(MultiGlobalPart, self).__init__()

        self.num_non_local = in_channels // inter_channels

        attentions = [GlobalPart(adj, in_channels, inter_channels) for _ in range(self.num_non_local)]
        self.attentions = nn.ModuleList(attentions)

        self.cat_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x_size = x.shape
        x = x.contiguous()
        x = x.view(-1, *x_size[2:])
        x = x.permute(0, 2, 1)

        x = torch.cat([self.attentions[i](x) for i in range(len(self.attentions))], dim=1)

        x = x.permute(0, 2, 1).contiguous()

        x = x.view(*x_size)

        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x)))

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)

        return x


class SingleGlobalPart(nn.Module):
    def __init__(self, adj, in_channels, output_channels, dropout=None):
        super(SingleGlobalPart, self).__init__()

        self.attentions = GlobalPart(adj, in_channels, output_channels//2)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x_size = x.shape
        x = x.contiguous()
        x = x.view(-1, *x_size[2:])

        x = x.permute(0, 2, 1)

        x = self.attentions(x)

        x = x.permute(0, 2, 1).contiguous()

        x = x.view(*x_size)

        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn(x))

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)

        return x
