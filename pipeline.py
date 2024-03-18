import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from ok import load
import mne_icalabel
class pipeliune:
    def __init__(self, raw, fs=250, cutoff=1):
        self.raw = raw
        self.fs = fs
        self.cutoff = cutoff

    def butter_highpass_filter(self, order=5):
        filt_raw = self.raw.copy().filter(l_freq=1.0, h_freq=None)
        return filt_raw

    def notch_filter(self):
        data_highpass = self.butter_highpass_filter()
        filt_raw = data_highpass.copy().notch_filter([48, 52], picks="eeg", filter_length="auto", phase="zero", verbose=True)
        return filt_raw

    def ica(self):
        raw_notch_filtered = self.notch_filter()
        #Plot raw_notch_filtered
        filt_raw = raw_notch_filtered.set_eeg_reference("average")
        ica = ICA(
            n_components=15,
            max_iter="auto",
            method="infomax",
            random_state=97,
            fit_params=dict(extended=True),
        )
        ica.fit(filt_raw)
    def IC_label(self):
        a=self.ica()
        ic_labels = mne_icalabel.label_components(a, ica, method="iclabel")
        print(ic_labels["labels"])
        ica.plot_properties(raw, picks=[0, 12], verbose=False)
        labels = ic_labels["labels"]
        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        ]
        print(f"Excluding these ICA components: {exclude_idx}")
        return a

raw=load()
#raw.plot()
pip=pipeliune(raw)
pip.IC_label()