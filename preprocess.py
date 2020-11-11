import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import display
import librosa

def main():
    dest_directory = args.dest
    data = pd.read_csv(os.path.join(dest_directory, "UrbanSound8K/metadata/UrbanSound8K.csv"))
    print(data.head())
    print(data["fold"].value_counts())

    y, sr = librosa.load(os.path.join(dest_directory, "UrbanSound8K/audio/fold5/100032-3-0-0.wav"))
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40)
    print(melspectrogram.shape, chroma_stft.shape, chroma_cq.shape, chroma_cens.shape, mfccs.shape)

    # MFCC of dog bark
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess sound')
    parser.add_argument("--dest", type=str, default='./data')
    args = parser.parse_args()
    main()