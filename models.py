from tensorflow.keras.layers import Conv1D, Input, Dense, Dropout, MaxPooling1D

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