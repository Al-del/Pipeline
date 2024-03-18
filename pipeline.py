import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from ok import load
class pipeliune:
    def __init__(self, raw, fs=250, cutoff=1):
        self.raw = raw
        self.fs = fs
        self.cutoff = cutoff

    def butter_highpass_filter(self, order=5):
        data = self.raw.get_data()
        y = mne.filter.filter_data(data, self.fs, l_freq=self.cutoff, h_freq=None)
        return y

    def notch_filter(self, f0, Q):
        data_highpass = self.butter_highpass_filter()
        y = mne.filter.notch_filter(data_highpass, self.fs, freqs=f0, notch_widths=Q)
        return y

    def ica(self):
        raw_notch_filtered = self.notch_filter(50, 2)
        #Plot raw_notch_filtered
        ica = ICA(n_components=20, random_state=97, max_iter='auto')
        ica.fit(raw_notch_filtered)
raw=load()
pipeline = pipeliune(raw)
pipeline.ica()
