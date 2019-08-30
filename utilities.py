import numpy as np

import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
import random
import matplotlib.pyplot as plt
dataset_folder = "dataset/"

class_labels = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]

mslen = 32000  


def take_audio(filename):
    return wav.read(filename)


def get_features(flatten=True, mfcc_len=13):
 
    data = []
    labels = []
    sampling_maxf = 0
    min_sample = int('9' * 10)
    s = 0
    cnt = 0
    cur_dir = os.getcwd()
    os.chdir('..')
    os.chdir(dataset_folder)
    x_test = []
    x_train = []
    y_train = []
    y_test = []
    for i, directory in enumerate(class_labels):
        print("started reading folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = take_audio(filename)
            

            sampling_maxf = max(sampling_maxf, fs)
            s_len = len(signal)
            # pad the signals to have same size if lesser than required
            # else slice them
            if s_len < mslen:
                signal_pad = mslen - s_len
                pad_rem = signal_pad % 2
                signal_pad /= 2
                # plt.figure(1)
            	# plt.title('Signal Wave...')
            	# plt.plot(signal)
            	# plt.show()
            	# break
            	#print(signal.shape)
                signal_pad = int(signal_pad)
                signal = np.pad(signal, (signal_pad, signal_pad + pad_rem), 'constant', constant_values=0)
            else:
                signal_pad = s_len - mslen
                pad_rem = signal_pad % 2
                signal_pad /= 2
                #mfcc_de_de = mfcc_de_de.reshape(mfcc_de_de.shape[0], mfcc_de_de.shape[1]*3)
           		#mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)
            	#print(mfcc_de_de.shape)
                signal_pad = int(signal_pad)
                signal = signal[signal_pad:signal_pad + mslen]
            min_sample = min(len(signal), min_sample)

           
            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)
            #mfcc_de_de = speechpy.feature.extract_derivative_feature(mfcc)    
            
            if flatten:
                mfcc = mfcc.flatten()

            # speaker = int(filename[0]+filename[1])
            # if speaker==12 or speaker==16:
            # 	x_test.append(mfcc_de_de)
            # 	y_test.append(i)
            # else:
            # 	x_train.append(mfcc_de_de)
            # 	y_train.append(i)

            #mfcc = stats.zscore(mfcc, axis=1, ddof=1)
            data.append(mfcc)
            #data.append(mfcc_de_de)
            labels.append(i)
            cnt += 1
        print("ended reading folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    rand_s = random.randrange(0, 50, 2)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=rand_s)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def display_metrics(y_pred, y_true):
    print(accuracy_score(y_pred=y_pred, y_true=y_true))
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))

# x_train, x_test, y_train, y_test = get_features(flatten=False)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
