import os
import h5py
import numpy as np
from labelMapping import diseasedict

dbPath = '../data/HiSeqV2.h5'

db = h5py.File(dbPath, 'r')
y = db["label"][...]
totalSamples=y.shape[0]
nClasses = 33

classIdx2Label = {v: k for (k, v) in diseasedict.items()}
sampleCounts = [ (classIdx2Label[c],len(y[y == c])) for c in range(nClasses) ]
sortedCounts = sorted(sampleCounts, key=lambda x: x[1], reverse=True)

with open("../data/class_distribution.csv", 'w') as f:
    f.write("Class,Count,%\n")
    for label, count in sortedCounts:
        f.write("{label},{count},{perc:.2f} %\n".format(
            label=label, count=count, perc=100*count/totalSamples))
        print("{label},{count},{perc:.2f} %".format(
            label=label, count=count, perc=100*count/totalSamples))
