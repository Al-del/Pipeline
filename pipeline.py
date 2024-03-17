import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
class pipeliune:
    def __init__(self, data, fs=300,cutoff=1):
        self.data = data
        self.fs = fs
        self.cutoff = cutoff

    def butter_highpass(self, order=5):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
        return b, a


    def butter_highpass_filter(self, order=5):
        b, a = self.butter_highpass(self.cutoff, self.fs, order=order)
        y = signal.filtfilt(b, a, self.data)
        plt.plot(range(len(y)), y)
        plt.title("Filtered Signal")
        plt.show()
        return y
    def notch_filter(self, f0, Q):
        nyq = 0.5 * self.fs
        w0 = f0 / nyq
        b, a = signal.iirnotch(w0, Q)
        y_highpass = self.butter_highpass_filter()
        y = signal.filtfilt(b, a, y_highpass)
        plt.plot(range(len(y)), y)
        plt.title("Notch Filtered Signal")
        plt.show()
        return y
    def ica(self):
        notch_filt= self.notch_filter()
        ica = FastICA(n_components=3)
        S_ = ica.fit_transform(notch_filt)
        A_ = ica.mixing_
        plt.plot(range(len(S_)), S_)
        plt.title("ICA")
        
