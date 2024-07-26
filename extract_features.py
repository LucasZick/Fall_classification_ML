import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def extract_features(file):
    data = pd.read_csv(file, header=None)

    sv_max = data[1].max()
    sv_min = data[1].min()
    sv_range = sv_max - sv_min

    peaks, _ = find_peaks(data[1])

    num_peaks = len(peaks)
    if num_peaks > 0:
        peak_heights = data[1].iloc[peaks]
        avg_peak_height = peak_heights.mean()
        if len(peaks) > 1:
            avg_distance_between_peaks = np.mean(np.diff(peaks))
        else:
            avg_distance_between_peaks = 0
    else:
        avg_peak_height = 0
        avg_distance_between_peaks = 0

    return sv_max, sv_range, num_peaks, avg_peak_height, avg_distance_between_peaks
