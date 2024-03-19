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

    def IC_label(self):
        raw_notch_filtered = self.notch_filter()
        filt_raw = raw_notch_filtered.set_eeg_reference("average")

        all_excluded_components = []

        for i in range(int(filt_raw.times[-1])):
            raw_cropped = filt_raw.copy().crop(tmin=i, tmax=i+1)

            ica = ICA(
                n_components=2,
                max_iter="auto",
                method="infomax",
                random_state=97,
                fit_params=dict(extended=True),
            )
            ica.fit(raw_cropped)

            # Apply the ICA to the raw data
            raw_ica_applied = ica.apply(raw_cropped.copy())


            ic_labels = mne_icalabel.label_components(raw_ica_applied, ica, method="iclabel")
            print(ic_labels["labels"])
            #ica.plot_properties(raw_ica_applied, verbose=False)
            labels = ic_labels["labels"]
            exclude_idx = [
                idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
            ]
            print(f"Excluding these ICA components: {exclude_idx}")
            excluded_components = ica.get_components()[:, exclude_idx]

            all_excluded_components.append(excluded_components)

        return ica

raw=load()
#raw.plot()
pip=pipeliune(raw)
a=pip.IC_label()
