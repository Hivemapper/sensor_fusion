from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, fs, cutoff=1, order=2):
    nyq = 0.5 * fs  # Define Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y
