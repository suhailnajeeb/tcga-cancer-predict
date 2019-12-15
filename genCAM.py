import os
import h5py
import keras
import numpy as np
import matplotlib.pyplot as plt
import progressbar as progbar
from labelMapping import diseasedict
from keras.models import load_model
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency, visualize_activation, overlay

def genCAM(model, layerName,  filterIdx, inputData, modifier='guided'):
    """ generates class activation maps for given layer and given filter/node index """

    modelCopy = model
    layerIdx = utils.find_layer_idx(modelCopy, layerName)
    modelCopy.layers[layerIdx].activation = activations.linear
    modelCopy = utils.apply_modifications(modelCopy)

    cam = visualize_cam(modelCopy,
                        layerIdx,
                        filter_indices=filterIdx,
                        seed_input=inputData,
                        backprop_modifier=modifier)

    return cam


def rgb2gray(rgb, threshold=128):
    """ convert rgb CAM to grayscale and map and apply threshold """
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    gray[gray < threshold] = 0
    return gray

# input params
modelPath = '../models/DilatedCNN2D_002_02/best.hdf5'
layerName = 'global_max_pooling2d_1'
dbPath = '../data/HiSeqV2.h5'
classes = [6, 10, 5, 17, 25, 15, 18, 22]
figSavePath = '../figures/CAM_CNN2D_002_02b.jpg'

numSamplesPerClass = 1

db = h5py.File(dbPath, 'r')
nTotal = db["RNASeq"].shape[0]
X = np.arange(nTotal)
y = db["label"][...]

classIdx2Label = {v: k for (k, v) in diseasedict.items()}

# getting input samples for each class
vizData = {}
for cdx in classes:
    classSamples = X[y == cdx]
    if numSamplesPerClass >= len(classSamples) or numSamplesPerClass < 0:
        vizData[classIdx2Label[cdx]] = classSamples
    else:
        vizData[classIdx2Label[cdx]] = np.random.choice(classSamples, numSamplesPerClass, replace=False)

# generating CAM
print("generating CAMs ...")
model = load_model(modelPath)
classActivationMaps = {}
for c, dataIdx in vizData.items():
    x = db["RNASeq"][sorted(dataIdx)]
    x = np.expand_dims(x, axis=-1)

    classActivationMaps[c] = {  "cam": genCAM(model, layerName, diseasedict[c], x), 
                                "input": x[0, :, :, 0] } 

# plotting CAM
plt.figure(figsize=(16,12), dpi=200)
plotRows = 1 if len(classes)//2 < 1 else len(classes)//2
plotCols = len(classes)//plotRows
for idx, (c, maps) in enumerate(classActivationMaps.items()):
    ax = plt.subplot(plotCols, plotRows,  idx+1)
    ax.imshow(maps["cam"],  cmap='jet')
    
    # # showing original gene array, and CAM overlayed
    # ax.imshow(maps["input"])
    # ax.imshow(maps["cam"],  cmap='jet', alpha=0.75)

    plt.title(c)

print("saving figure at {}".format(figSavePath))
plt.savefig(figSavePath,dpi=200)

# plotting CAM
plt.figure(figsize=(16, 12), dpi=200)
plotRows = 1 if len(classes)//2 < 1 else len(classes)//2
plotCols = len(classes)//plotRows
for idx, (c, maps) in enumerate(classActivationMaps.items()):
    ax = plt.subplot(plotCols, plotRows,  idx+1)
    ax.imshow(rgb2gray(maps["cam"]))
    plt.title(c)

plt.show()
