from keras.layers import Conv1D, Input, Dense, Dropout, MaxPooling1D, Flatten
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, SpatialDropout2D
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import ReLU, Softmax
from keras.models import Model

modelArch = {}												
addModel = lambda f:modelArch.setdefault(f.__name__,f)

# build and return model
def makeModel(architecture,verbose=True):

	model = modelArch[architecture]()

	if verbose:
		print(model.summary(line_length=150))
	
	return model

@addModel
def CNN1D_001(nFeat = 20530, nClasses = 33):
    input = Input(shape = (nFeat,1))

    x1 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (input)
    x2 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu') (x1)
    x3 = Dropout(0.5)(x2)
    x4 = MaxPooling1D(pool_size = 2)(x3)
    x5 = Flatten()(x4)
    x6 = Dense(100, activation = 'relu')(x5)
    out = Dense(nClasses, activation='softmax')(x6)

    model = Model(inputs = input, outputs = out)

    return model

@addModel
def DilatedCNN2D_001(inputDim=(116, 177, 1), nClasses=33, perms=100):

    def bn_block(x):
        return BatchNormalization()(x)

    def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
        x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
        x = bn_block(x)
        x = ReLU()(x)
        return x

    atrousRates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21)] 
    numFilters = [32, 32, 32, 32, 32, 32, 32, 32]
    featList = []

    i = Input(shape=inputDim)

    # learnabale remapping 
    x = Dense(perms)(i)
    x = bn_block(x)

    # convolutional blocks
    for idx, (nFilter, dilationRate) in enumerate(zip(numFilters, atrousRates)):
        x = conv_block(x, nFilter, (3,3), dilationRate)
        featList.append(x)

    # bottlenecking
    x = concatenate(featList)
    x = SpatialDropout2D(0.25)(x)
    x = conv_block(x, 128, (1,1))
    x = conv_block(x, 1, (1, 1))

    # classifying
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    o = Dense(nClasses, activation='softmax')(x)

    model = Model(inputs=i, outputs=o)

    return model

@addModel
def DilatedCNN2D_002(inputDim=(116, 177, 1), nClasses=33, perms=100):

    def bn_block(x):
        return BatchNormalization()(x)

    def conv_block(x, nb_filter, filter_size, atrous_rate=(1, 1)):
        x = Conv2D(nb_filter, filter_size, dilation_rate=atrous_rate, kernel_initializer='he_normal', padding='same')(x)
        x = bn_block(x)
        x = ReLU()(x)
        return x

    atrousRates = [(1,1), (1,1), (2,2), (3,3), (5,5), (8,8), (13,13), (21,21)] 
    numFilters = [32, 64, 64, 128, 64, 64, 64, 128]
    featList = []

    i = Input(shape=inputDim)

    # learnabale remapping 
    x = Dense(perms)(i)
    x = bn_block(x)

    # convolutional blocks
    for idx, (nFilter, dilationRate) in enumerate(zip(numFilters, atrousRates)):
        x = conv_block(x, nFilter, (3,3), dilationRate)
        if idx % 2 ==0 and idx > 0:
            x = MaxPooling2D()(x)

    # bottlenecking
    x = conv_block(x, 32, (3, 3))
    x = conv_block(x, 128, (1,1))
    x = SpatialDropout2D(0.25)(x)

    # classifying
    x = conv_block(x, nClasses, (1, 1))
    x = GlobalMaxPooling2D()(x)
    o = Softmax()(x)

    model = Model(inputs=i, outputs=o)

    return model

if __name__ == "__main__":
    m = DilatedCNN2D_002()
    print(m.summary(line_width=150))
