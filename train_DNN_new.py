import numpy as np
import sys
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, Conv3D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D, MaxPooling3D
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.metrics import classification_report
from utilities import get_data, class_labels
#from read import read_data
#from read_util import get_read_data, class_labels
from keras import backend as K

models = ["CNN", "LSTM"]


def get_model(model_name, input_shape, fs):
    model = Sequential()
    #below is the architecture of i-CNN
    if model_name == 'CNN':
        #first conv layer 
        model.add(Conv2D(8, (fs, fs), strides=(1, 1), 
                         input_shape=(input_shape[0], input_shape[1], 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        #second conv layer 
        model.add(Conv2D(8, (fs, fs)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        #third conv layer 
        model.add(Conv2D(8, (fs, fs)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        #fourth conv layer 
        model.add(Conv2D(8, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        #flatten and then use dropout layer  
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.8))
    
    
    model.add(Dense(len(class_labels), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae', 'mse'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    return model


def train_CNN_model(model, eps, x_train, x_test, y_train, y_test):
  
    best_acc = 0
    for i in tqdm(range(eps)):
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]  
        model.fit(x_train, y_train, batch_size=32, epochs=1)
        loss, acc, mae, mse = model.evaluate(x_test, y_test)
        if acc > best_acc:
            print('Updated best accuracy', acc)
            best_acc = acc
            model.save_weights(best_model_path)   

def evaluate_CNN_model(model, i, dim, filt_size, x_test, y_test):
	
	
	model.load_weights(best_model_path)
	loss, cat_acc, mae, mse = model.evaluate(x_test, y_test)

	y_predicted = np.argmax(np.array(model.predict(x_test)), axis=1)
	y_test = np.argmax(np.array(y_test), axis=1)
	
	target_names = ['class 1','class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']
	precision_recall = classification_report(y_predicted, y_test, target_names=target_names)
	#with open('../models/acc_matrix/matrix_2D_hybrid_znorm_'+str(dim)+'_fs_'+ str(filt_size)+'_' +str(i)+'_.txt', 'w') as f:
		#print(precision_recall, file=open(f, "a"))
		#print(cat_acc, file=open(f, "a"))
	
	print("for iters")
	print(precision_recall)
	print(cat_acc)
	print("###")
	
	return({"cat_acc": cat_acc, "mae": mae, "mse": mse})


if __name__ == "__main__":

	if len(sys.argv) != 3:
		sys.stderr.write('Model not added yet\n')
		sys.exit(-1)
	
	input = sys.argv[1]
	n = int(sys.argv[2]) - 1
	print('model given', models[n])
	num_iter = 3
	filt_size = 3
	eps = 80
	cat_acc = 0.0
	mae = 0.0
	mse = 0.0
	global x_train, y_train, x_test, y_test
	global best_model_path
	
	for i in range(num_iter):
		if n == 0 and input == 'read_util.py':
			print("###both###")
			x_train, x_test, y_train, y_test = get_read_data(flatten=False)
			y_train = np_utils.to_categorical(y_train)
			y_test = np_utils.to_categorical(y_test)
			dim = x_train[0].shape[0]
			print(dim)
			in_shape = x_train[0].shape
			x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
			x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
	    

		if n == 0 and input == 'utilities.py':
			print("###simple###")
			x_train, x_test, y_train, y_test = get_data(flatten=False)
			y_train = np_utils.to_categorical(y_train)
			y_test = np_utils.to_categorical(y_test)
			dim = x_train[0].shape[0]
			print(dim)
			in_shape = x_train[0].shape
			x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
			x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
	    
		if n == 0 and input == 'read.py':
			x_train, x_test, y_train, y_test = read_data()
			y_train = np_utils.to_categorical(y_train)
			y_test = np_utils.to_categorical(y_test)
			x_train = x_train.reshape(x_train.shape[0], 56, 39, 3)
			x_test = x_test.reshpe(x_test.shape[0], 56, 39, 3)
			dim = x_train.shape[1]
			print(dim)
			
	    
	
		print("---------")
		print(x_train[0].shape)
		
		model = get_model(models[n], x_train[0].shape, filt_size)
		
		

		#best_model_path = '../models/best_model_2D_hybrid_znorm_'+str(dim)+'_fs_'+ str(filt_size)+'_' +str(i)+ '_' + models[n] + '.h5'
		best_model_path = '../small_mfcc_'+str(dim)+'_fs_'+ str(filt_size)+'_' +str(i)+ '_' + models[n] + '.h5'
		#best_model_path = 'best_model_2D_199_fs_3_0_CNN.h5'
		train_CNN_model(model, eps, x_train, x_test, y_train, y_test)
		dict_op = evaluate_CNN_model(model, i, dim, filt_size, x_test, y_test)
		print(dict_op["cat_acc"])
		print(dict_op["mae"])
		print(dict_op["mse"])

		cat_acc += dict_op["cat_acc"]	
		mae += dict_op["mae"]
		mse += dict_op["mse"]
		
	print('avged Categorical Accuracy = ', cat_acc/num_iter)		
	print('avged MAE = ', mae/num_iter)	
	print('avged MSE = ', mse/num_iter)
	K.clear_session()
