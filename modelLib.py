from tensorflow.keras.layers import Conv1D, Input, Dense, Dropout, MaxPooling1D, Flatten
from tensorflow.keras.models import Model

modelArch = {}												
addModel = lambda f:modelArch.setdefault(f.__name__,f)

# build and return model
def makeModel(architecture,verbose=True):

	model = modelArch[architecture]()

	if verbose:
		print(model.summary(line_length=150))
	
	return model

@addModel
def CNN1D_001(nFeat = 20530, n_classes = 33):
    input = Input(shape = (nFeat,1))

    x1 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (input)
    x2 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x1)
    x3 = Dropout(0.5)(x2)
    x4 = MaxPooling1D(pool_size = 2)(x3)
    x5 = Flatten()(x4)
    x6 = Dense(100, activation = 'relu')(x5)
    out = Dense(n_classes, activation = 'softmax')(x6)

    model = Model(inputs = input, outputs = out)

    return model

@addModel
def CNN1D_002(nFeat = 20530, n_classes = 33):
    input = Input(shape = (nFeat,1))

    x1 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (input)
    x2 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x1)
    x3 = Dropout(0.5)(x2)
    x4 = MaxPooling1D(pool_size = 2)(x3)
    x5 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x4)
    x6 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x5)
    x7 = Dropout(0.5)(x6)
    x8 = MaxPooling1D(pool_size = 2)(x7)
    x9 = Flatten()(x8)
    x10 = Dense(100, activation = 'relu')(x9)
    out = Dense(n_classes, activation = 'softmax')(x10)

    model = Model(inputs = input, outputs = out)

    return model

@addModel
def CNN1D_003(nFeat = 20530, n_classes = 33):
    input = Input(shape = (nFeat,1))
    x0 = Dense(100, activation = 'relu')(input)
    x1 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (input)
    x2 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x1)
    x3 = Dropout(0.5)(x2)
    x4 = MaxPooling1D(pool_size = 2)(x3)
    x5 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x4)
    x6 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x5)
    x7 = Dropout(0.5)(x6)
    x8 = MaxPooling1D(pool_size = 2)(x7)
    x9 = Flatten()(x8)
    x10 = Dense(100, activation = 'relu')(x9)
    out = Dense(n_classes, activation = 'softmax')(x10)

    model = Model(inputs = input, outputs = out)

    return model