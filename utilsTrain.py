import h5py
import numpy as np

def generator(h5file, indexes, batch_size):
    X = []
    Y = []
    idx = 0
    while True:
        for index in indexes:
            if(idx==0):
                X = []
                Y = []
            RNA = h5file["RNASeq"][index]
            label = h5file["label"][index]
            X.append(RNA)
            Y.append(label)
            idx = idx + 1
            if(idx>=batch_size):
                idx = 0
                yield np.asarray(X),np.asarray(Y)