import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction


        self.jo_conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.jo_conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.jo_conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.jo_conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.bo_conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.bo_conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.bo_conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.bo_conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.jm_conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.jm_conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.jm_conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.jm_conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)

        self.bm_conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.bm_conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.bm_conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.bm_conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)


        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, jo, bo, jm, bm, jo_A=None, jo_alpha=1, bo_A=None, bo_alpha=1, jm_A=None, jm_alpha=1, bm_A=None, bm_alpha=1):


        print('x.sum:', jo.sum())        

        jo1, jo2, jo3 = self.jo_conv1(jo).mean(-2), self.jo_conv2(jo).mean(-2), self.jo_conv3(jo)
        print('jo1:', jo1.sum())
        jo1 = self.tanh(jo1.unsqueeze(-1) - jo2.unsqueeze(-2))
        print('jo2:', jo1.sum())
        jo1 = self.jo_conv4(jo1) * jo_alpha + (jo_A.unsqueeze(0).unsqueeze(0) if jo_A is not None else 0)  # N,C,V,V
        print('jo3:', jo1.sum())
        jo1 = torch.einsum('ncuv,nctv->nctu', jo1, jo3)

        bo1, bo2, bo3 = self.bo_conv1(bo).mean(-2), self.bo_conv2(bo).mean(-2), self.bo_conv3(bo)
        bo1 = self.tanh(bo1.unsqueeze(-1) - bo2.unsqueeze(-2))
        bo1 = self.bo_conv4(bo1) * bo_alpha + (bo_A.unsqueeze(0).unsqueeze(0) if bo_A is not None else 0)  # N,C,V,V
        bo1 = torch.einsum('ncuv,nctv->nctu', bo1, bo3)

        jm1, jm2, jm3 = self.jm_conv1(jm).mean(-2), self.jm_conv2(jm).mean(-2), self.jm_conv3(jm)
        jm1 = self.tanh(jm1.unsqueeze(-1) - jm2.unsqueeze(-2))
        jm1 = self.jm_conv4(jm1) * jm_alpha + (jm_A.unsqueeze(0).unsqueeze(0) if jm_A is not None else 0)  # N,C,V,V
        jm1 = torch.einsum('ncuv,nctv->nctu', jm1, jm3)

        bm1, bm2, bm3 = self.bm_conv1(bm).mean(-2), self.bm_conv2(bm).mean(-2), self.bm_conv3(bm)
        bm1 = self.tanh(bm1.unsqueeze(-1) - bm2.unsqueeze(-2))
        bm1 = self.bm_conv4(bm1) * bm_alpha + (bm_A.unsqueeze(0).unsqueeze(0) if bm_A is not None else 0)  # N,C,V,V
        bm1 = torch.einsum('ncuv,nctv->nctu', bm1, bm3)

        print('x.sum:', jo1.sum())  

        return jo1, bo1, jm1, bm1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]

        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.jo_down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.jo_down = lambda x: x
        else:
            self.jo_down = lambda x: 0
        if self.adaptive:
            self.jo_PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.jo_A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.jo_alpha = nn.Parameter(torch.zeros(1))
        self.jo_bn = nn.BatchNorm2d(out_channels)
        bn_init(self.jo_bn, 1e-6)
        self.jo_relu = nn.ReLU(inplace=True)


        if residual:
            if in_channels != out_channels:
                self.bo_down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.bo_down = lambda x: x
        else:
            self.bo_down = lambda x: 0
        if self.adaptive:
            self.bo_PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.bo_A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.bo_alpha = nn.Parameter(torch.zeros(1))
        self.bo_bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bo_bn, 1e-6)
        self.bo_relu = nn.ReLU(inplace=True)

        if residual:
            if in_channels != out_channels:
                self.jm_down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.jm_down = lambda x: x
        else:
            self.jm_down = lambda x: 0
        if self.adaptive:
            self.jm_PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.jm_A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.jm_alpha = nn.Parameter(torch.zeros(1))
        self.jm_bn = nn.BatchNorm2d(out_channels)
        bn_init(self.jm_bn, 1e-6)
        self.jm_relu = nn.ReLU(inplace=True)


        if residual:
            if in_channels != out_channels:
                self.bm_down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.bm_down = lambda x: x
        else:
            self.bm_down = lambda x: 0
        if self.adaptive:
            self.bm_PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.bm_A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.bm_alpha = nn.Parameter(torch.zeros(1))
        self.bm_bn = nn.BatchNorm2d(out_channels)
        bn_init(self.bm_bn, 1e-6)
        self.bm_relu = nn.ReLU(inplace=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)


    def forward(self, jo, bo, jm, bm):

        jo_y = None
        bo_y = None
        jm_y = None
        bm_y = None


        if self.adaptive:
            jo_A = self.jo_PA
        else:
            jo_A = self.jo_A.cuda(jo.get_device())

        if self.adaptive:
            bo_A = self.bo_PA
        else:
            bo_A = self.bo_A.cuda(bo.get_device())

        if self.adaptive:
            jm_A = self.jm_PA
        else:
            jm_A = self.jm_A.cuda(jm.get_device())

        if self.adaptive:
            bm_A = self.bm_PA
        else:
            bm_A = self.bm_A.cuda(bm.get_device())

        for i in range(self.num_subset):
            jo_z, bo_z, jm_z, bm_z = self.convs[i](jo, bo, jm, bm, jo_A[i], self.jo_alpha, bo_A[i], self.bo_alpha, jm_A[i], self.jm_alpha, bm_A[i], self.bm_alpha)
            print('A: ', jo_A[i].sum())
            print('z: ', jo_z.sum())
            jo_y = jo_z + jo_y if jo_y is not None else jo_z
            bo_y = bo_z + bo_y if bo_y is not None else bo_z
            jm_y = jm_z + jm_y if jm_y is not None else jm_z
            bm_y = bm_z + bm_y if bm_y is not None else bm_z

        jo_y = self.jo_bn(jo_y)
        jo_y += self.jo_down(jo)
        jo_y = self.jo_relu(jo_y)

        bo_y = self.bo_bn(bo_y)
        bo_y += self.bo_down(bo)
        bo_y = self.bo_relu(bo_y)

        jm_y = self.jm_bn(jm_y)
        jm_y += self.jm_down(jm)
        jm_y = self.jm_relu(jm_y)

        bm_y = self.bm_bn(bm_y)
        bm_y += self.bm_down(bm)
        bm_y = self.bm_relu(bm_y)

        return jo_y, bo_y, jm_y, bm_y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)

        self.jo_tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.jo_relu = nn.ReLU(inplace=True)
        if not residual:
            self.jo_residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.jo_residual = lambda x: x

        else:
            self.jo_residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


        self.bo_tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.bo_relu = nn.ReLU(inplace=True)
        if not residual:
            self.bo_residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.bo_residual = lambda x: x

        else:
            self.bo_residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


        self.jm_tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.jm_relu = nn.ReLU(inplace=True)
        if not residual:
            self.jm_residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.jm_residual = lambda x: x

        else:
            self.jm_residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)


        self.bm_tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.bm_relu = nn.ReLU(inplace=True)
        if not residual:
            self.bm_residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.bm_residual = lambda x: x

        else:
            self.bm_residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, jo, bo, jm, bm):

        # x1: tensor(-3.0518e-05, device='cuda:0', grad_fn=<SumBackward0>)
        # a.sum: tensor(-3.0518e-05, device='cuda:0', grad_fn=<SumBackward0>)
        # b.sum: tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        # A:  tensor(25., device='cuda:0', grad_fn=<SumBackward0>)
        # z:  tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        # a.sum: tensor(-3.0518e-05, device='cuda:0', grad_fn=<SumBackward0>)
        # b.sum: tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        # A:  tensor(24., device='cuda:0', grad_fn=<SumBackward0>)
        # z:  tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        # a.sum: tensor(-3.0518e-05, device='cuda:0', grad_fn=<SumBackward0>)
        # b.sum: tensor(0.0001, device='cuda:0', grad_fn=<SumBackward0>)
        # A:  tensor(20., device='cuda:0', grad_fn=<SumBackward0>)
        # z:  tensor(0.0001, device='cuda:0', grad_fn=<SumBackward0>)     

        print('x1:', jo.sum())
        jo_, bo_, jm_, bm_ = self.gcn1(jo, bo, jm, bm)
        print('x2:', jo_.sum())

        jo = self.jo_relu(self.jo_tcn1(jo_) + self.jo_residual(jo))
        bo = self.bo_relu(self.bo_tcn1(bo_) + self.bo_residual(bo))
        jm = self.jm_relu(self.jm_tcn1(jm_) + self.jm_residual(jm))
        bm = self.bm_relu(self.bm_tcn1(bm_) + self.bm_residual(bm))

        return jo, bo, jm, bm


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point

        self.jo_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.bo_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.jm_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.bm_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.jo_fc = nn.Linear(base_channel*4, num_class)
        self.bo_fc = nn.Linear(base_channel*4, num_class)
        self.jm_fc = nn.Linear(base_channel*4, num_class)
        self.bm_fc = nn.Linear(base_channel*4, num_class)

        nn.init.normal_(self.jo_fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.bo_fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.jm_fc.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.bm_fc.weight, 0, math.sqrt(2. / num_class))

        if drop_out:
            self.drop_jo = nn.Dropout(drop_out)
        else:
            self.drop_jo = lambda x: x

        if drop_out:
            self.drop_bo = nn.Dropout(drop_out)
        else:
            self.drop_bo = lambda x: x

        if drop_out:
            self.drop_jm = nn.Dropout(drop_out)
        else:
            self.drop_jm = lambda x: x

        if drop_out:
            self.drop_bm = nn.Dropout(drop_out)
        else:
            self.drop_bm = lambda x: x

        bn_init(self.jo_bn, 1)
        bn_init(self.bo_bn, 1)
        bn_init(self.jm_bn, 1)
        bn_init(self.bm_bn, 1)


    def forward(self, jo):

          

        bo = torch.zeros_like(jo).to(jo.device)
        for v1, v2 in ntu_pairs:
            bo[:, :, v1 - 1] = jo[:, :, v1 - 1] - jo[:, :, v2 - 1]

        jm = torch.zeros_like(jo).to(jo.device)
        jm[:, :-1] = jo[:, 1:] - jo[:, :-1]

        bm = torch.zeros_like(jo).to(jo.device)
        bm[:, :-1] = bo[:, 1:] - bo[:, :-1]


        if len(jo.shape) == 3:
            N, T, VC = jo.shape
            jo = jo.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            bo = bo.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            jm = jm.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
            bm = bm.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = jo.size()
        

        print('jo.sum:', jo.sum())
        jo = jo.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        jo = self.jo_bn(jo)
        print('jo.sum:', jo.sum())      
        jo = jo.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        bo = bo.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        bo = self.bo_bn(bo)
        bo = bo.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        jm = jm.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        jm = self.jm_bn(jm)
        jm = jm.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        bm = bm.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        bm = self.bm_bn(bm)
        bm = bm.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        
        jo, bo, jm, bm = self.l1(jo, bo, jm, bm)
        #print('jo.sum:', jo.sum())
        jo, bo, jm, bm = self.l2(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l3(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l4(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l5(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l6(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l7(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l8(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l9(jo, bo, jm, bm)
        jo, bo, jm, bm = self.l10(jo, bo, jm, bm)


        # N*M,C,T,V
        c_new = jo.size(1)

        jo = jo.view(N, M, c_new, -1)
        jo = jo.mean(3).mean(1)
        jo = self.drop_jo(jo)
        jo = self.jo_fc(jo)

        bo = bo.view(N, M, c_new, -1)
        bo = bo.mean(3).mean(1)
        bo = self.drop_bo(bo)
        bo = self.bo_fc(bo)

        jm = jm.view(N, M, c_new, -1)
        jm = jm.mean(3).mean(1)
        jm = self.drop_jm(jm)
        jm = self.jm_fc(jm)

        bm = bm.view(N, M, c_new, -1)
        bm = bm.mean(3).mean(1)
        bm = self.drop_bm(bm)
        bm = self.bm_fc(bm)

        x = 0.6 * jo + 0.6 * bo + 0.4 * jm + 0.4 * bm

        return x, jo, bo, jm, bm
