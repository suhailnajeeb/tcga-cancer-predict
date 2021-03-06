import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
modelName = 'CNN1D_003'
bestModelPath = 'weights\\best.hdf5'
modelFolder = 'model\\'

ensureDir(weightsFolder)
ensureDir(modelFolder)
ensureDir(os.path.join(weightsFolder,modelName))

epochs = 20
epochStart = 0
patience = 50
batchSize = 32

db = h5py.File(dbPath, 'r')

nTotal = db["RNASeq"].shape[0]
nFeat = db["RNASeq"].shape[1]

n_classes = 33

X = np.arange(nTotal)
y = db["label"][...]

skf = StratifiedKFold(n_splits = 5)
skf.get_n_splits(X,y)

kdx = 0
for train_index, test_index in skf.split(X, y):
    kdx += 1
    train_generator = generator(db, train_index, batch_size = 32)
    test_generator = generator(db, test_index, batch_size = 32)

    model = makeModel(modelName)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


    print("\nCross Validation Fold : %02d \n" % kdx)

    check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_fold_%02d" % kdx + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', save_best_only=True, mode='auto')
    check2 = ModelCheckpoint( bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
    check3 = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=int(patience), verbose=0, mode='auto')
    check4 = CSVLogger(os.path.join(modelFolder, modelName + '_fold_trainingLog.csv'), separator=',', append=True)
    check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(patience), verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)

    trained_model=model.fit_generator(train_generator, steps_per_epoch=(len(train_index) // batchSize), epochs=epochs, initial_epoch=epochStart,
											validation_data=test_generator, validation_steps=(len(test_index) // batchSize), callbacks=[check1, check2, check3, check4, check5],
											verbose=1)

db.close()

