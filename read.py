import numpy as np 
from numpy import genfromtxt
import csv
from sklearn.model_selection import train_test_split


#data = genfromtxt('../emo_384.csv',dtype=float, delimiter=',', names=True)
def read_data():


	data = []
	labels = [] 
	with open('../emo_6553.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			if(row[0] == 'name'):
				continue 
			category = row[0]
			letter = category[6]

			#print(len(row))
			if(letter == 'W'):
				labels.append(0)
			if(letter == 'L'):
				labels.append(1)
			if(letter == 'E'):
				labels.append(2)
			if(letter == 'A'):
				labels.append(3)
			if(letter == 'F'):
				labels.append(4)
			if(letter == 'T'):
				labels.append(5)
			if(letter == 'N'):
				labels.append(6)
			feature_vec = np.array(row[1:])
			feature_vec = feature_vec[:-1]
			#feature_vec = feature_vec.reshape(int(len(feature_vec)/2),2)	
			data.append(feature_vec)
	        
	data = np.array(data)
	labels = np.array(labels)
	data = data.astype(float)
	labels = labels.astype(int)
	
	x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
	return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

#x_train, x_test, y_train, y_test = read_data()