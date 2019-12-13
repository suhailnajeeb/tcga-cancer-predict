import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
import os

# to test generator, values = next(generator) in code

def ensureDir(filePath):
	
	''' This function checks if the folder at filePath exists.
		If not, it creates it. '''

	directory = os.path.dirname(filePath)

	if not os.path.exists(directory):
		os.makedirs(directory)

def generator(h5file, indexes, batch_size):
    X = []
    Y = []
    idx = 0
    while True:
        for index in indexes:
            if(idx==0):
                X = []
                Y = []
            RNA = np.expand_dims(h5file["RNASeq"][index], axis = 1)
            label = to_categorical(h5file["label"][index], num_classes = 33, dtype = np.uint8)
            X.append(RNA)
            Y.append(label)
            idx = idx + 1
            if(idx>=batch_size):
                idx = 0
                yield np.asarray(X),np.asarray(Y)