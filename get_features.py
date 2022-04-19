import librosa
import numpy as np
import os
from scipy.io import savemat

num_frames = 350
n_mels = 64
n_samples = 153

path_to_audio = './librosa_data/audio/'

files_ = os.listdir(path_to_audio)

mel_ = np.array([])

for file_ in files_:
    if file_.endswith('ogg'):

        y, sr = librosa.load('{p_}{f_}'.format(p_=path_to_audio,f_=file_))
        mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
        # print(np.shape(mel1))
        if mel_.size == 0:
            mel_ = mel1
        else:
            mel_ = np.concatenate((mel_, mel1), axis=1)
        print(np.shape(mel_))

mel_ = np.transpose(mel_, (1,0))
mel_ = mel_[:53550,:]
print(np.shape(mel_))
mel_ = np.reshape(mel_, (n_samples, num_frames, n_mels))
mel_ = np.transpose(mel_, (1,0,2))
print(np.shape(mel_))
savemat('test.mat', {'mel': mel_})
# mel_ = mel1
#
# y, sr = librosa.load(librosa.ex('brahms'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# print(np.shape(mel1))
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('choice'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('fishin'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# # y, sr = librosa.load(librosa.ex('humpback'))
# y, sr = librosa.load('./librosa_data/audio/Kevin_MacLeod_-_Vibe_Ace.ogg')
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('libri1'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('libri2'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('libri3'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
#
#
# y, sr = librosa.load(librosa.ex('nutcracker'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('pistachio'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('robin'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('sweetwaltz'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))
#
# y, sr = librosa.load(librosa.ex('vibeace'))
# mel1 = librosa.feature.melspectrogram(y, sr, n_mels = n_mels)
# mel_ = np.concatenate((mel_, mel1), axis=1)
# print(np.shape(mel_))