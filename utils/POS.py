import numpy as np
from numpy.linalg import inv
import math

from utils.filter import butter_bandpass_filter, detrend

PRE_STEP_ASF = False
PRE_STEP_CDF = False


def CDF(C, B):
    C_ = np.matmul(inv(np.diag(np.mean(C, 1))), C) - 1
    F = np.fft.fft(C_)
    S = np.dot(np.expand_dims(np.array([-1 / math.sqrt(6), 2 / math.sqrt(6), -1 / math.sqrt(6)]), axis=0), F)
    W = np.real((S * S.conj()) / np.sum((F * F.conj()), 0)[None, :])
    W[:, 0:B[0]] = 0
    W[:, B[1] + 1:] = 0

    F_ = F * W
    iF = (np.fft.ifft(F_) + 1).real
    C__ = np.matmul(np.diag(np.mean(C, 1)), iF)
    C = C__.astype(np.float)
    return C


def ASF(C):
    alpha = .002
    delta = 0.0001
    C_ = np.dot(inv(np.diag(np.mean(C, 1))), C) - 1
    L = C.shape[1]
    F = np.fft.fft(C_) / L
    W = delta / (1e-12 + np.abs(F[0, :]))
    W = W.astype(np.complex)

    W[np.abs(F[0, :]) < alpha] = 1

    W = np.stack((W, W, W), axis=0)
    F_ = F * W

    C__ = np.dot(np.diag(np.mean(C, 1)), (np.fft.ifft(F_) + 1))

    C__ = C__.astype(np.float)
    return C__


class Pulse():
    def __init__(self, framerate, signal_size):
        self.framerate = float(framerate) # video framerate -> 35
        self.signal_size = signal_size # the number of frames -> 6300
        self.minFreq = 0.7  # frequency limits (0.7 Hz to 3.5 Hz), corresponding to typical human heart rates (≈ 42–210 bpm).
        self.maxFreq = 3.5  #
        self.fft_spec = []

    # def get_pulse(self, mean_rgb):
    #     seg_t = 3.2
    #     l = int(self.framerate * seg_t)
    #     H = np.zeros(self.signal_size)
    #
    #     B = [int(0.8 // (self.framerate / l)), int(4 // (self.framerate / l))]
    #
    #     for t in range(0, (self.signal_size - l + 1)):
    #         # pre processing steps
    #         C = mean_rgb[t:t + l, :].T
    #
    #         if PRE_STEP_CDF:
    #             C = CDF(C, B)
    #
    #         if PRE_STEP_ASF:
    #             C = ASF(C)
    #
    #         # POS
    #         mean_color = np.mean(C, axis=1)
    #         diag_mean_color = np.diag(mean_color)
    #         diag_mean_color_inv = np.linalg.inv(diag_mean_color)
    #         Cn = np.matmul(diag_mean_color_inv, C)
    #         projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
    #         S = np.matmul(projection_matrix, Cn)
    #         std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
    #         P = np.matmul(std, S)
    #         H[t:t + l] = H[t:t + l] + (P - np.mean(P))
    #
    #     return H

    def get_pulse(self, mean_rgb): # mean_rgb.shape -> (6300, 3)
        C = mean_rgb.T # transpose, shape becomes (3, 6300)
        # Each row is one color channel (R, G, B), and columns represent time.

        # POS
        mean_color = np.mean(C, axis=1) # mean po bojama, # shape: (3,)
        diag_mean_color = np.diag(mean_color) # dijagonalna matrica (samo su dijagonale != 0) # 3×3 diagonal matrix
        diag_mean_color_inv = np.linalg.inv(diag_mean_color) # multiplikativni inverz dijagonalne matrice

        # This step removes global brightness variation — normalizing each channel by its mean.
        # After this, Cn contains relative color changes rather than absolute intensity.
        # produces normalized color signal.
        Cn = np.matmul(diag_mean_color_inv, C) # Cn.shape -> (3, 6300)

        # This projects the 3D color signal onto a 2D space using a transformation matrix designed to:
        #   Emphasize chrominance (color differences),
        #   Suppress brightness variation (luminance).
        # This is the heart of the POS algorithm.
        # This is part of the CHROM method — a linear projection to isolate pulse-related variations.
        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
        S = np.matmul(projection_matrix, Cn) # S.shape -> (2, 6300)

        # This step normalizes the two channels of S to have the same standard deviation.
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S) # P.shape -> (1, 6300)

        # detrend
        # Removes the mean — leaving only the oscillating (AC) component,
        # This is the final pulse signal.
        H = P - np.mean(P)

        # H is a 1D NumPy array (length ~6300), representing color-based pulse signal over time.
        return H

    def get_rfft_hr(self, signal):
        signal_size = len(signal)
        signal = signal.flatten()
        fft_data = np.fft.rfft(signal)  # FFT
        fft_data = np.abs(fft_data)

        freq = np.fft.rfftfreq(signal_size, 1. / self.framerate)  # Frequency data

        inds = np.where((freq < self.minFreq) | (freq > self.maxFreq))[0]
        fft_data[inds] = 0
        bps_freq = 60.0 * freq
        max_index = np.argmax(fft_data)
        fft_data[max_index] = fft_data[max_index] ** 2
        self.fft_spec.append(fft_data)
        HR = bps_freq[max_index]
        return HR
