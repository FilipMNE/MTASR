import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import models.setting as setting
# import setting as setting
 


# class MixA_Module(nn.Module):
#     """ Spatial-Skin attention module"""
#
#     def __init__(self):
#         super(MixA_Module, self).__init__()
#         self.softmax = nn.Softmax(dim=-1)
#         self.AVGpool = nn.AdaptiveAvgPool1d(1)
#         self.MAXpool = nn.AdaptiveMaxPool1d(1)
#         self.tlam = nn.Sequential(
#             nn.Conv1d(1, 1, 3, 1, 1),
#             nn.Sigmoid()
#         )
#         self.res = nn.Sequential(
#             nn.Conv1d(16, 1, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x, peak):
#         """
#             inputs :
#                 x : input feature maps( B X C X L)
#                 peak : peak feature maps( B X L)
#             returns :
#                 out : attention value
#                 spatial attention: W x H
#         """
#         m_batchsize, C, L = x.size()
#         # B_C_Tf = x.view(m_batchsize, C, -1)
#         B_L_C = x.permute(0, 2, 1)
#
#         # B_L_C_AVG = torch.sigmoid(self.AVGpool(B_L_C)).view(m_batchsize, L)
#         # B_L_C_MAX = torch.sigmoid(self.MAXpool(B_L_C)).view(m_batchsize, L)
#         # B_L_C_Fusion = B_L_C_AVG + B_L_C_MAX + peak
#
#         B_L_C_AVG = self.AVGpool(B_L_C).transpose(1, 2)
#         B_L_C_conv = self.tlam(B_L_C_AVG).squeeze(1)
#         B_L_C_Fusion = B_L_C_conv + peak + self.res(x).squeeze(1)
#
#         Attention_weight = self.softmax(B_L_C_Fusion)
#         # Attention_weight = Attention_weight.view(m_batchsize, L)
#         Attention_weight = Attention_weight.unsqueeze(1).expand_as(x)
#         # mask1 mul
#         output = x * Attention_weight + x
#         return output, B_L_C_Fusion

class MixA_Module(nn.Module):
    def __init__(self):
        super(MixA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # self.q_w = nn.Conv1d(1, 1, 3, 1, 1)
        self.v_w = nn.Conv1d(16, 1, 3, 1, 1)

    def forward(self, x, peak):
        # q = self.q_w(F.adaptive_avg_pool1d(x.transpose(1, 2), 1).transpose(1, 2))
        v = self.v_w(x)
        Attention_weight = self.softmax(torch.sigmoid(v) + peak)
        output = x * Attention_weight + x
        return output, Attention_weight


class BVPFeatrueExtraction(nn.Module):

    def __init__(self):
        super(BVPFeatrueExtraction, self).__init__()
        self.conv_peak = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            # nn.Conv1d(16, 16, 3, 1, 1),
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(inplace=True),
        )
        self.peak_main = nn.Sequential(
            nn.Conv1d(16, 8, 3, 1, 1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 4, 3, 1, 1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(inplace=True)
        )

        self.peak_residual = nn.Sequential(
            nn.Conv1d(16, 4, 3, 1, 1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(inplace=True)
        )

        self.peak_out = nn.Sequential(
            nn.Conv1d(4, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.MixA_Module = MixA_Module()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(16, 32, 9, 1, 4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 9, 1, 4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True)

        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(32, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x_16 = self.conv_peak(x)  # B, 16, 350
        x_peak_main = self.peak_main(x_16)  # B, 4, 350
        x_peak_residual = self.peak_residual(x_16)  # B, 4, 350
        x_peak_p = x_peak_main + x_peak_residual
        x_peak_out = self.peak_out(x_peak_p)
        x_peak = x_peak_out[:, 0, :]  # B, 210

        x_16_SA, attention_16 = self.MixA_Module(x_16, x_peak_out)  # B, 16, 350

        x_32 = self.conv_1(x_16_SA)

        x_64 = self.conv_2(F.max_pool1d(x_32, 5, 5))

        x_128 = self.conv_3(F.max_pool1d(x_64, 5, 5))

        return x_16, x_32, x_64, x_128, x_peak


class TLAM(nn.Module):
    def __init__(self, input_length):
        super(TLAM, self).__init__()
        self.atten = nn.Sequential(
            nn.Conv1d(input_length, input_length // 2, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(input_length // 2, input_length, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, Y):
        atten = self.atten(F.adaptive_avg_pool1d(Y.transpose(1, 2), 1)).transpose(1, 2)
        atten = atten.expand_as(Y)
        return atten * Y + Y


class CLAM(nn.Module):
    def __init__(self, input_channels):
        super(CLAM, self).__init__()
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_channels, input_channels // 2, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(input_channels // 2, input_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, Y):
        atten = self.atten(Y)
        return atten * Y + Y


class CLMMAM(nn.Module):
    def __init__(self, input_channels):
        super(CLMMAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.atten = nn.Sequential(
            nn.Conv1d(input_channels, input_channels // 2, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(input_channels // 2, input_channels, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, Y):
        max = self.max_pool(Y)
        avg = self.avg_pool(Y)
        atten_max = self.atten(max)
        atten_avg = self.atten(avg)
        atten = self.sigmoid(atten_avg + atten_max)
        return atten * Y + Y


class ECAM(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super(ECAM, self).__init__()

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.atten = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, Y):
        Y_gap = self.gap(Y)
        atten_gap = self.atten(Y_gap.transpose(-1, -2)).transpose(-1, -2)
        atten = self.sigmoid(atten_gap)
        return atten * Y + Y


# Understanding and Learning Discriminant Features based on Multiattention 1DCNN for Wheelset Bearing Fault Diagnosis
class JAM(nn.Module):
    def __init__(self, in_c):
        super(JAM, self).__init__()
        self.eam_conv1x1 = nn.Conv1d(in_c, 1, 1, 1, bias=False)
        self.eam_conv3x3 = nn.Conv1d(in_c, in_c, 3, 1, 1, bias=False)
        self.ecam = ECAM(in_c)

    def forward(self, Y):
        F_ecm_1 = torch.sigmoid(self.eam_conv1x1(Y))
        F_ecm_2 = F.leaky_relu(self.eam_conv3x3(Y))
        F_ecm = F_ecm_1 * F_ecm_2
        ECM_out = F_ecm + Y
        return self.ecam(ECM_out)


class Task(nn.Module):
    '''
    The Task class is a feature refiner. It takes the features that were extracted by BVPFeatrueExtraction (x_16, x_32, x_64, x_128) 
    and passes them through lightweight processing blocks to combine and strengthen the information

    It has 4 small blocks, and each block:
        1. Applies a special attention layer (ECAM, which emphasizes important features)
        2. Runs a 1D convolution to adjust feature size
        3. (Optionally) Downsamples the signal with MaxPool1d
        Then, it adds the result to the corresponding input feature and sends it to the next block.

        It’s gradually building up deeper features by combining and transforming earlier features (x_32, x_64, etc.).
    '''
    def __init__(self):
        super(Task, self).__init__()
        self.block_1 = nn.Sequential(
            # TLAM(2100),
            ECAM(16),
            nn.Conv1d(16, 32, 1, 1),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(inplace=True)
        )
        self.block_2 = nn.Sequential(
            # TLAM(2100),
            ECAM(32),
            nn.Conv1d(32, 64, 1, 1),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)  # (Optionally) Downsamples the signal with MaxPool1d
        )

        self.block_3 = nn.Sequential(
            # TLAM(420),
            ECAM(64),
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_4 = nn.Sequential(
            ECAM(128),
            nn.Conv1d(128, 256, 1, 1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, x_16, x_32, x_64, x_128):
        # x_32_a = self.block_1(x_16) + x_32
        x_64_a = self.block_2(x_32) + x_64
        x_128_a = self.block_3(x_64_a) + x_128
        x_256_a = self.block_4(x_128_a)
        return x_256_a


class MetaStress(nn.Module):
    def __init__(self, classes_num=2, hidden_size=128, num_layers=2):
        '''
        "param classes_num: Number of output classes (stressed vs. not stressed)
        '''
        super(MetaStress, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # This module is a deep 1D convolutional neural network designed to extract rich, multi-level features 
        # from a BVP signal (Blood Volume Pulse, a type of rPPG/PPG signal). 
        # It combines peak detection, residual learning, attention, and progressively deeper convolutions to encode signal characteristics.
        # It's the first stage of the MetaStress model, producing features used for stress classification and HR estimation.
        # In MetaStress, this module is used to preprocess the PPG input, turning it into feature tensors.
        self.bvp = BVPFeatrueExtraction() # TODO slozena CNN arhitektura

        self.state_task = Task() # donekle objasnjena u klasi Task
        # self.classifier = nn.Linear(512, classes_num)

        # nn.Sequential is a PyTorch utility that allows you to chain layers/modules together in a sequence. 
        # Data flows through them in the order they are defined, one after another. It's like a pipeline.
        self.classifier = nn.Sequential(
            # A fully connected (dense) layer.
            # Input: a tensor of size [batch_size, 512] — output from BVPFeatrueExtraction module.
            # Output: a tensor of size [batch_size, hidden_size], where hidden_size=512
            # it performs output = x * W^T + b, where W and b are learnable parameters (weight and bias).
            nn.Linear(512, setting.classifier_hidden_size),
            # nn.Linear(512, 256),

            # A non-linear activation function.
            # LeakyReLu(x) = x, if x>0 else 0.01*x
            nn.LeakyReLU(),

            # Another fully connected layer, this time projecting the hidden representation to the number of output classes - 2 for binary classification
            # Output shape: [batch_size, classes_num=2]
            nn.Linear(setting.classifier_hidden_size, classes_num),
        )

        self.hr_task = Task()
        self.hr_estimator = nn.Linear(256, 1)
        # self.hr_estimator = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 1),
        # )

    # forward function - defines what happens when you pass input data through the model.
    def forward(self, x, net_type="both"):
        '''
        The goal of this function is to:
            Extract features from the input signal using the bvp module
            Use those features to either:
                - Classify stress (net_type="classify")
                - Estimate heart rate (HR) (net_type="hr")
                - Do both (net_type="both")
        '''
        
        # This line passes the input x (a signal like PPG) into the BVPFeatrueExtraction module, 
        # which processes it and returns 4 levels of features (x_16 to x_128) and a peak signal.
        x_16, x_32, x_64, x_128, peak = self.bvp(x)

        if net_type == "classify":
            state_out = self.state_task(x_16, x_32, x_64, x_128)
            out = self.classifier(F.adaptive_avg_pool1d(state_out, 1).squeeze(-1))
        elif net_type == "hr":
            hr_out = self.hr_task(x_16, x_32, x_64, x_128)
            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))
        else:
            state_out = self.state_task(x_16, x_32, x_64, x_128)

            hr_out = self.hr_task(x_16, x_32, x_64, x_128)

            # out = self.classifier(F.adaptive_avg_pool1d(state_out + hr_out, 1).squeeze(-1))
            out = self.classifier(F.adaptive_avg_pool1d(torch.cat((state_out, hr_out), 1), 1).squeeze(-1))
            # out = self.classifier(F.adaptive_avg_pool1d(self.mmtm(state_out, hr_out), 1).squeeze(-1))

            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))

        if net_type == "classify":
            return out, peak
        elif net_type == "hr":
            return hr, peak
        else:
            return out, hr, peak

    # Nigdje se ne koristi
    # This function initializes the hidden state and cell state for the LSTM layers.
    def init_hidden(self, bs, device):
        h0 = Variable(torch.zeros(self.num_layers, bs, self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, bs, self.hidden_size).to(device))
        return h0, c0


if __name__ == '__main__':
    net = MetaStress()
    in_p = torch.randn((5, 1, 2100))
    out, hr, peak = net(in_p)
    print(out.shape)
    print(hr.shape)
    print(peak.shape)


    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(net))
    #
    # in_p = torch.randn((5, 32, 420))
    # Jnet = JAM(32)
    # out = Jnet(in_p)
    # print(out.shape)
