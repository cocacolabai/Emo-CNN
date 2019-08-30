import numpy as np
import csv
import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats

dataset_folder = "dataset/"

class_labels = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]

mslen = 32000  # Empirically calculated for the given dataset


def read_wav(filename):
    return wav.read(filename)


def get_read_data(flatten=True, mfcc_len=39):
    """
    Read the files get the data perform the test-train split and return them to the caller
    :param mfcc_len: Number of mfcc features to take for each frame
    :param flatten: Boolean specifying whether to flatten the data or not
    :return: 4 arrays, x_train x_test y_train y_test
    """
    data = []
    labels = []
    max_fs = 0
    min_sample = int('9' * 10)
    s = 0
    cnt = 0
    cur_dir = os.getcwd()
    os.chdir('..')
    os.chdir(dataset_folder)
    data = []
    labels = [] 
    
    for i, directory in enumerate(class_labels):
        print(i)
        print("some folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            with open('../../merge_this.csv') as csvDataFile:
                csvReader = csv.reader(csvDataFile)
            
                fs, signal = read_wav(filename)
                max_fs = max(max_fs, fs)
                s_len = len(signal)
                # pad the signals to have same size if lesser than required
                # else slice them
                if s_len < mslen:
                    pad_len = mslen - s_len
                    pad_rem = pad_len % 2
                    pad_len /= 2
                    pad_len = int(pad_len)
                    signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
                else:
                    pad_len = s_len - mslen
                    pad_rem = pad_len % 2
                    pad_len /= 2
                    pad_len = int(pad_len)
                    signal = signal[pad_len:pad_len + mslen]
                min_sample = min(len(signal), min_sample)

                mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)
                for row in csvReader:
                    if str(row[0]) == str(filename):
                        #print(filename)
                        #print("taken")
                        #print(row[0])
                        extra_f = np.array(row[1:1678]).astype(np.float)
                        #print(len(extra_f))
                        extra_f = extra_f.reshape(int(len(extra_f)/39),39)
                
                hybrid = np.concatenate((mfcc, extra_f), axis=0)   
                if flatten:
                    mfcc = mfcc.flatten()
                
                hybrid = stats.zscore(hybrid, axis=1, ddof=1)
                data.append(hybrid)
                labels.append(i)
                cnt += 1
            
            csvDataFile.close()
        print("ended that folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)



#x_train, x_test, y_train, y_test = get_read_data(flatten=False)

#print(x_train[5][150][:])
#print(np.min(x_train))
