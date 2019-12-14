import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from utilsTrain import generator, ensureDir
import os
from modelLib import makeModel
from tensorflow.keras.layers import Conv1D, Input, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config = config))

dbPath = 'C:\Data\data.h5'

weightsFolder = 'weights\\'
modelName = 'CNN1D_001'
bestModelPath = 'weights\\best.hdf5'
modelFolder = 'model\\'

ensureDir(weightsFolder)
ensureDir(modelFolder)
ensureDir(os.path.join(weightsFolder,modelName))

epochs = 10
epochStart = 0
patience = 50
batchSize = 32

db = h5py.File(dbPath, 'r')

nTotal = db["RNASeq"].shape[0]
nFeat = db["RNASeq"].shape[1]

n_classes = 33

X = np.arange(nTotal)
y = db["label"][...]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 42)

train_generator = generator(db, X_train, batch_size = 32)
test_generator = generator(db, X_test, batch_size = 32)

model = makeModel(modelName)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

check1=ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', save_best_only=True, mode='auto')
check2=ModelCheckpoint(bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
check3=EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=0, mode='auto')
check4=CSVLogger(os.path.join(modelFolder, modelName + '_trainingLog.csv'), separator=',', append=True)
check5=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience // 1.5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)

trained_model=model.fit_generator(train_generator, steps_per_epoch=(len(X_train) // batchSize), epochs=epochs, initial_epoch=epochStart,
											validation_data=test_generator, validation_steps=(len(X_test) // batchSize), callbacks=[check1, check2, check3, check4, check5],
											verbose=1)

db.close()

# getting filters and biases for conv layer 1
filters, biases = model.layers[1].get_weights()

# normalize values to 0~1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

import matplotlib.pyplot as plt

# plot first few filters

filter = 1 - filters[:,:,0].transpose()

plt.imshow(filter, cmap = 'gray', vmin = 0, vmax = 1)
plt.show()


for i in range(filters.shape[2]):
    f = 1 - filters[:,:,i].transpose()
    ax = plt.subplot(8,8,i+1)
    ax.imshow(f, cmap = 'gray', vmin = 0, vmax = 1)
    i = i + 1

#plt.savefig('filters.png')
plt.show()


