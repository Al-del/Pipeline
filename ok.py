#Load .fif file
import mne
import os

from matplotlib import pyplot as plt


def load(ok=False):
    def load_fif_file(file_path):
        raw = mne.io.read_raw_fif(file_path, preload=True)
        return raw
    file_path="."
    for file in os.listdir(file_path):
        if file.endswith(".fif"):
            raw = load_fif_file(file)
            if ok==True:
                for channel in raw.ch_names:
                    channel_data = raw.copy().pick_channels([channel]).get_data()
                    plt.figure()
                    plt.plot(channel_data[0])
                    plt.title(f'Channel: {channel}')
                    plt.show()
            return raw
a=load()
a.plot()