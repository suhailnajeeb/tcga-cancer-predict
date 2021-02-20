import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from labelMapping import diseasedict

figSavePath = '../figures/gene_array.jpg'
dbPath = '../data/HiSeqV2.h5'
db = h5py.File(dbPath, 'r')

idx = 1215
X = db["RNASeq"][idx]
X = np.multiply(X, 3)
Y = db["label"][idx]
classIdx2Label = {v: k for (k, v) in diseasedict.items()}
Y = classIdx2Label[Y]

plt.figure(figsize=(10,10), dpi=300)
plt.imshow(X,  cmap="gist_gray", interpolation='nearest')
# plt.title(Y)
plt.savefig(figSavePath, dpi=300)
plt.show()
