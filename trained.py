import numpy as np
import sys
import scipy.io.wavfile as wav
import os
import speechpy
import pandas as pd
from keras.utils import np_utils
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, Conv3D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D, MaxPooling3D
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#from train_DNN_new import get_model
from keras.models import Model
from collections import defaultdict
from math import sqrt
import random
from sklearn.preprocessing import MinMaxScaler
#from read_util import get_read_data
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_model_pca(model_name, input_shape, fs):
    model = Sequential()
    if model_name == 'CNN':
        model.add(Conv2D(8, (fs, fs), strides=(1, 1), 
                         input_shape=(input_shape[0], input_shape[1], 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2D(8, (fs, fs)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        
        model.add(Conv2D(8, (fs, fs)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        
        model.add(Conv2D(8, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    
    elif model_name == 'LSTM':
        model.add(LSTM(128, input_shape=(input_shape[0], input_shape[1])))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='tanh'))
    
    model.add(Dense(len(class_labels), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae', 'mse'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #print(model.summary())
    return model

def find_hash(xs, k):
    num_audi = len(xs)
    audi_size = len(xs[0])
    rand_mat = np.empty((k,audi_size,))
    for i in range(k):
        rand_vec = np.random.choice([1, -1], size=(audi_size,), p=[1./2, 1./2])
        rand_mat[i] = rand_vec
    rand_mat = rand_mat.transpose()

    sim_mat_real = np.dot(xs, rand_mat)
    sim_mat = (sim_mat_real >= 0).astype(int)
    return sim_mat

def read_wav(filename):
    return wav.read(filename)


def get_data_pca(flatten=True, mfcc_len=39):
    data = []
    labels = []
    max_fs = 0
    min_sample = int('9' * 10)
    s = 0
    cnt = 0
    cur_dir = os.getcwd()
    os.chdir('..')
    os.chdir(dataset_folder)
    for i, directory in enumerate(class_labels):
        print("some folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
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
                signal = signal[pad_len	:pad_len + mslen]
            min_sample = min(len(signal), min_sample)

            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)
            #mfcc_de_de = speechpy.feature.extract_derivative_feature(mfcc)
            
            #hybrid = stats.zscore(hybrid, axis=1, ddof=1)
            
            if flatten:
                mfcc = mfcc.flatten()
            data.append(mfcc)
            #data.append(mfcc_de_de)
            labels.append(i)
            cnt += 1
        print("done that folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    return({"data": data, "labels": labels})



# model_op = []
print("innnnnnnnnnnnnnnnnnnnnnn")
dataset_folder = "dataset/"
class_labels = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]
mslen = 32000 
filt_size = 3
print("reading")
dataset = get_data_pca(flatten=False)
print("done reading")
data = dataset["data"]
data = np.array(data)
const = 0.21224846
data = data.reshape(data.shape[0], 199, 39, 1)       
label = dataset["labels"]
label = np.array(label)

            
model = get_model_pca("CNN", [199,39], filt_size)
model.load_weights('../models/best_model_199_fs_3_2_CNN.h5')

model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
features = model.predict(data, verbose=0)
print(features.shape)


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(features)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

label_emo = np.array(np.zeros(label.shape)).astype(str)   
emo = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]   
for i in range(len(emo)):
    indices = np.where(label == i)
    label_emo[indices] = emo[i]

principalDf['label'] = label_emo

principalDf['principal component 1'] = (principalDf['principal component 1'] - principalDf['principal component 1'].mean(axis=0)) / principalDf['principal component 1'].std(axis=0)
principalDf['principal component 2'] = (principalDf['principal component 2'] - principalDf['principal component 2'].mean(axis=0)) / principalDf['principal component 2'].std(axis=0)
principalDf['principal component 3'] = (principalDf['principal component 3'] - principalDf['principal component 3'].mean(axis=0)) / principalDf['principal component 3'].std(axis=0)

list1 = []
list2 = []
list3 = []

classes = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]
for target in classes:
    indicesToKeep = principalDf['label'] == target
    list1.append(np.average(np.array(principalDf.loc[indicesToKeep, 'principal component 1'])))
    list2.append(np.average(np.array(principalDf.loc[indicesToKeep, 'principal component 2'])))
    list3.append(np.average(np.array(principalDf.loc[indicesToKeep, 'principal component 3'])))



fig = plt.figure(figsize = (8,8))
list3[0] = const 
ax = fig.add_subplot(1,1,1, projection='3d') 


ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)

ax.set_title('CNN + 3 comp. PCA', fontsize = 20)
targets = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
index = 0
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['label'] == target
    
    ax.scatter(list1[index], list2[index], list3[index]
               , c = color
               , s = 50)
    index+=1
ax.legend(targets)
ax.grid()

plt.show()

print(list1)
print(list2)
print(list3)              

###get a 3D plot of all points
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1, projection='3d') 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_zlabel('Principal Component 3', fontsize = 15)

# ax.set_title('CNN + 3 comp. PCA', fontsize = 20)
# targets = ["Angry", "Boredom", "Disgust", "Fear", "Happy", "Sad", "Neutral"]
# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
# for target, color in zip(targets,colors):
#     indicesToKeep = principalDf['label'] == target
#     ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
#                , principalDf.loc[indicesToKeep, 'principal component 2']
#                , principalDf.loc[indicesToKeep, 'principal component 3']
#                , c = color
#                , s = 30)
# ax.legend(targets)
# ax.grid()

# plt.show()


#print(principalComponents[535])
#print(principalComponents[536])

# sim_mat = find_hash(features, 2)
# sim_mat = np.array(sim_mat)
# print(sim_mat.shape)
# print(sim_mat[0])

# scaler = MinMaxScaler()
# scaler.fit(sim_mat)
# sim_mat_norm = scaler.transform(sim_mat)

# print(sim_mat_norm.shape)
# print(sim_mat_norm[0])
