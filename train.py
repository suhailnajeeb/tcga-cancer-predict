import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from utilsTrain import generator, ensureDir

from modelLib import makeModel
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
configTF = tf.ConfigProto()
configTF.gpu_options.allow_growth = True
sess = tf.Session(config=configTF)

# setting RNG seeds
tf.set_random_seed(2727)
np.random.seed(2727)

dbPath = '../data/HiSeqV2.h5'

rootModelDir = '../models/'
modelName = 'DilatedCNN2D_002'
modelFolder = os.path.join(rootModelDir, modelName)
weightsFolder = os.path.join(modelFolder, 'weights')
bestModelPath = os.path.join(modelFolder, 'best.hdf5')
ensureDir(weightsFolder)

epochs = 50
epochStart = 0
patience = 10
batchSize = 32

db = h5py.File(dbPath, 'r')

nTotal = db["RNASeq"].shape[0]
X = np.arange(nTotal)
y = db["label"][...]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 42)

train_generator = generator(db, X_train, batch_size = 32)
test_generator = generator(db, X_test, batch_size = 32)

if epochStart > 0:
	model = load_model(bestModelPath)
else:
	model = makeModel(modelName)
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adamax', metrics = ['categorical_accuracy'])

check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName +"_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), 
						monitor='val_loss', save_best_only=True, mode='auto')

check2 = ModelCheckpoint(bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')

check3 = EarlyStopping(monitor='val_loss', min_delta=0.01,
                       patience=patience*3, verbose=0, mode='auto')

check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog.csv'), 
					separator=',', append=True)

check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience // 1.5,
							 verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)

trained_model=model.fit_generator(train_generator, 
								steps_per_epoch=(len(X_train) // batchSize), 
								epochs=epochs, 
								initial_epoch=epochStart,
								validation_data=test_generator, 
								validation_steps=(len(X_test) // batchSize), 
								callbacks=[check1, check2, check3, check4, check5],
								verbose=1)

db.close()
