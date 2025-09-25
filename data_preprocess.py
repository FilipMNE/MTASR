from scipy import signal
from scipy import interpolate
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import math
import os
import yaml
import csv

from utils.HRV import HRV
from utils.filter import butter_bandpass_filter, detrend

yaml_file = "./setting.yaml"
cfg = yaml.safe_load(open(yaml_file, 'r'))

# podesi root_dir, model za face detection, duzinu segmenta i framerate
root_dir = "/Volumes/stari/UBFC-Phys" # zavisi koristis li eksterni disk (stari ili novi) ili lokalni disk, "/Volumes/externiMacNovi/UBFC-Phys" ili "/Volumes/stari/UBFC-Phys" ili "/Users/filipjovanovic/Desktop/OneAI/VISION/MTASR_rad_trening"
face_detection_model = 'faceBoxes' # "yolo" ili "faceBoxes"
segment_id = 'segment_30s' # "no_segmentation" ili "segment_10s" ili "segment_30s"
framerate_id = 'framerate_1' # "framerate_1" ili "framerate_5"

rppg_dir = f"{root_dir}/{face_detection_model}/{segment_id}/{framerate_id}/rppg"
bvp_dir = "/Users/filipjovanovic/Desktop/OneAI/VISION/MTASR_rad_trening/bvp_info_values" # gdje se nalaze folderi 's1', 's2',... koji sadrze bvp.csv

drop_num = cfg[segment_id][framerate_id]['drop_num']
step = cfg[segment_id]['step']
clip_len = cfg[segment_id]['clip_len']
segment_len = cfg[segment_id]['segment_len']
number_of_segments = math.ceil(180 / segment_len)
ubfc_npy_filename = cfg[segment_id][framerate_id]['ubfc_npy_filename']


def outlier(ppg_signal):
    ppg_signal_new = ppg_signal
    ppg_mean = np.mean(ppg_signal)
    ppg_std = np.std(ppg_signal)
    outlier = np.logical_or(ppg_signal > ppg_mean + 2 * ppg_std, ppg_signal < ppg_mean - 2 * ppg_std)
    index = np.arange(len(ppg_signal))
    tck = interpolate.splprep(index[~outlier], ppg_signal[~outlier])
    for x in index[outlier]:
        ppg_signal_new[x] = interpolate.splev(x, tck)
    return ppg_signal_new


def missing_fix(signal):
    signal_new = signal
    missing = np.zeros(signal.shape, dtype=np.bool8)
    missing[signal == 0] = True
    if np.sum(missing) == 0:
        # return move_signal_filter(signal)
        return signal
    missing[signal != 0] = False
    index = np.arange(len(signal))
    try:
        tck = interpolate.splrep(index[~missing], signal[~missing])
        for x in index[missing]:
            signal_new[x] = interpolate.splev(x, tck)
    except Exception as e:
        print(e)
        # return move_signal_filter(signal)
        return signal
    # return move_signal_filter(signal_new)
    return signal_new


def move_signal_filter(x):
    return savgol_filter(x, 11, 3)


def find_peak(ppg_signal):
    '''
    uses SciPy's find_peaks function to detect local maxima in the input signal
    The distance=10 argument ensures that two detected peaks must be at least 10 frames apart. This prevents detecting multiple close-by fluctuations as separate peaks.
    The [0] extracts just the indices of the peaks from the tuple returned by find_peaks.
    For each peak index found, the corresponding position in index_arr is set to 1.
    This results in a binary array: 1s at peak positions, 0s elsewhere.
    This array serves as a binary mask showing where the peaks are in the original signal.
    '''
    # ppg_signal = signal.resample(ppg_signal, math.floor(len(ppg_signal) / 2))
    peak_ = signal.find_peaks(ppg_signal, distance=10)[
        0]  # 300ms Reference: Robust PPG Peak Detection Using Dilated Convolutional Neural Networks
    index_arr = np.zeros((len(ppg_signal)), dtype="uint8")
    for index in peak_:
        index_arr[index] = 1
    return index_arr


def z_score(ppg_signal):
    '''What it does:
        Centers the signal to zero mean.
        Scales the signal to unit variance (standard deviation = 1).
    Why it's important:
        Standardizing makes signals comparable (rPPG vs. BVP).
    Especially helpful before:
        Peak detection → reduces bias from overall signal amplitude.
        Gradient calculation → stabilizes signal dynamics.
        ML input → models train better on standardized data.
    '''
    ppg_mean = np.mean(ppg_signal)
    ppg_signal = ppg_signal - ppg_mean
    ppg_std = np.std(ppg_signal)

    return ppg_signal / ppg_std


def get_data(task, dir_, segment, save_datas):
    # read rppg and bvp signals from csv files
    person_rppg_dir = os.path.join(rppg_dir, dir_) # za rppg.csv
    person_bvp_dir = os.path.join(bvp_dir, dir_) # za bvp.csv
    if segment_id == 'no_segmentation':
        ppg_signal_path = os.path.join(person_rppg_dir, r"rppg_png_{1}_T{0}.csv").format(task, dir_)
    else:
        ppg_signal_path = os.path.join(person_rppg_dir, r"rppg_{1}_T{0}_seg{2}.csv").format(task, dir_, segment)
    ppg_signal = pd.read_csv(ppg_signal_path, header=None)
    bvp_signal = pd.read_csv(os.path.join(person_bvp_dir, r"bvp_{1}_T{0}.csv").format(task, dir_), header=None)
    
    # convert from pandas DataFrame of shape (6300,1) to numpy array (6300,)
    ppg_signal = ppg_signal.to_numpy().squeeze(-1)
    bvp_signal = bvp_signal.to_numpy().squeeze(-1)

    # segment traje segment_len sekundi, bvp signal je 64Hz, pa je segment duzine 64*segment_len
    segment_int = int(segment)
    start_segment_bvp = segment_int * 64 * segment_len
    end_segment_bvp = (segment_int + 1) * 64 * segment_len
    bvp_signal = bvp_signal[start_segment_bvp:end_segment_bvp]

    # Resample BVP to match PPG length using Fourier method along the given axis. Because a Fourier method is used, the signal is assumed to be periodic.
    # Necessary for fair comparison/sync.
    bvp_signal = signal.resample(bvp_signal, len(ppg_signal)) 

    if segment == "00": # uklanja prvih nekoliko frejmova samo ako se obradjuje prvi segment, ne mora da uklanja iz sredine snimka
        ppg_signal = ppg_signal[drop_num:] # Remove initial (5?) seconds of frames
        bvp_signal = bvp_signal[drop_num:]

    bvp_signal = z_score(bvp_signal) # standardizes signal (zero mean, unit variance) -> objasnjeno u komentarima u funckiji
    # butter_bandpass_filter:
    # It keeps only the frequencies between 0.7 and 2.5 Hz, and attenuates all other frequency components — like noise or unrelated physiological signals.
    # Removes low-frequency drift (e.g. head movement, lighting changes).
    # Removes high-frequency noise (sensor noise, sudden spikes).
    # Preserves only the physiological band of interest for heart rate (0.7–3.5 Hz).
    bvp_signal = butter_bandpass_filter(bvp_signal, 35, 0.7, 2.5) # isolates heart rate band (0.7–2.5 Hz) assuming 35Hz sampling

    # determine level from task number and test/ctrl group
    # 2 - neutral task (task 1) for both ctrl and test group
    # 1 - test group (hard scenario) tasks 2 and 3
    # 0 - ctrl group (easy scenario) tasks 2 and 3
    df_info = pd.read_csv(os.path.join(person_bvp_dir, r'info_{0}.txt'.format(dir_)), header=None)
    if task == 1:
        level = 2
    else:
        if df_info.values[2][0] == "test":
            level = 1
        else:
            level = 0

    label = 1 if task == 2 or task == 3 else 0

    # Breaks signals into sliding windows
    # clip_len: length of each segment (e.g., 2100 frames / 35 fps = 60 seconds worth of data)
    # step: how much to shift window forward (for overlap)
    # total_times: total number of windows
    total_times = (len(ppg_signal) - clip_len) / step + 1
    for times in range(math.floor(total_times)):
        # if segment == '00':
        #     print(times+1, ' put usao u petlju')
        # print(times, ' put usao u petlju')
        # Extract signals from current window
        ppg_start_index = int(times * step)
        ppg_end_index = int(ppg_start_index + clip_len)
        ppg_signal_ = ppg_signal[ppg_start_index: ppg_end_index]
        bvp_signal_ = bvp_signal[ppg_start_index: ppg_end_index]

        vpg_signal_ = np.gradient(ppg_signal_) # first derivative of rPPG (useful for detecting peaks)
        bvp_signal_ = z_score(bvp_signal_) # Standardizes BVP again

        # Peaks are detected using scipy.signal.find_peaks().
        # Inter-beat intervals (called RRi, in milliseconds) are calculated based on time between peaks.
        bvp_hrv_ = HRV(bvp_signal_, 35, 14) # distance=14 -> objasnjeno u komentarima u HRV konstruktoru

        # This method calculates the mean heart rate from the RR intervals:
        # Formula used: HR = 60 / (RRi / 1000) — converts each RR interval from milliseconds to beats per minute (BPM).
        # Then averages those values to get the mean heart rate over the signal segment.
        HR = bvp_hrv_.get_hr()

        # Final signal normalization and feature extraction
        ppg_signal_ = z_score(ppg_signal_)
        vpg_signal_ = z_score(vpg_signal_)

        peak_ = find_peak(ppg_signal_)
        vpg_peak_ = find_peak(vpg_signal_)

        # dir_, task, level, label --> metadata
        # ppg_singal_ --> na njemu treniram model
        # peak_, HR --> ground truths
        # vpg_signal_, vpg_peak_ --> nigdje se ne koriste
        datas = [dir_, task, level, label, ppg_signal_, peak_, vpg_signal_, vpg_peak_, HR]

        # print("Dodao", datas[:2], "u save_datas")
        save_datas.append(datas)


if __name__ == '__main__':
    havent_done = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15',
                   's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29',
                   's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's43',
                   's44', 's45', 's46', 's47', 's48', 's49', 's50', 's51', 's52', 's53', 's54', 's55', 's56']
    save_datas = []
    for dir_ in havent_done:
        for task in [1, 2, 3]:
            print("Preprocessing data for person", dir_, "task T%d" % task, "all segments")
            for segment in range(number_of_segments):
                if segment < 10:
                    segment = "0" + str(segment)
                else:
                    segment = str(segment)
                get_data(task, dir_, segment, save_datas)

    np.save(f"ubfc_npy_files/{face_detection_model}/{ubfc_npy_filename}.npy", save_datas)
    print(f"Data saved to ubfc_npy_files/{face_detection_model}/{ubfc_npy_filename}.npy")
