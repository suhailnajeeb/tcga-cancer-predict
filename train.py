import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from utilsTrain import generator
import os

################################################################
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto() 
# dynamically grow GPU memory 
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
'''

dbPath = 'C:\Data\data.h5'

weightsFolder = 'weights\\'
modelName = 'conv1D'
bestModelPath = 'weights\\best.hdf5'
modelFolder = 'model\\'

epochs = 100
epochStart = 0
patience = 50
batchSize = 32

################################################################

db = h5py.File(dbPath, 'r')

nTotal = db["RNASeq"].shape[0]
nFeat = db["RNASeq"].shape[1]

n_classes = 33

X = np.arange(nTotal)
y = db["label"][...]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

train_generator = generator(db, X_train, batch_size = 32)
test_generator = generator(db, X_test, batch_size = 32)

################################################################

from tensorflow.keras.layers import Conv1D, Input, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Model

input = Input(shape = (nFeat,1))

x1 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (input)
x2 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x1)
x3 = Dropout(0.5)(x2)
x4 = MaxPooling1D(pool_size = 2)(x3)
x5 = Flatten()(x4)
x6 = Dense(100, activation = 'relu')(x5)
out = Dense(n_classes, activation = 'softmax')(x6)

model = Model(inputs = input, outputs = out)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


################################################################

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:02d}-loss-{val_loss:.3f}.hdf5"), monitor='val_loss', save_best_only=True, mode='auto')
check2 = ModelCheckpoint(bestModelPath, monitor='val_loss', save_best_only=True, mode='auto')
#check3 = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience, verbose=0, mode='auto')
#check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog.csv'), separator=',', append=True)
#check5 = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience//1.5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-10)


trained_model = model.fit_generator(train_generator, steps_per_epoch=(len(X_train) // batchSize), epochs=epochs, initial_epoch=epochStart,
											validation_data= test_generator, validation_steps=(len(X_test) // batchSize), callbacks=[check1,check2],#,check3,check4,check5], 
											verbose=1)

db.close()