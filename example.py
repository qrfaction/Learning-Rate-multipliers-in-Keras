from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from multipliers import M_Nadam
# Loading mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshaping the data
# assuming 'channels_last' format
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Normalizing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

#
input_layer = Input(shape=(28,28,1))
x = Conv2D(32,3,activation='relu',padding='same',input_shape=(1,28,28),use_bias=False,name='c1')(input_layer)
x = Conv2D(32,3,activation='relu',padding='same',use_bias=False,name='c2')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu',use_bias=False,name='d1')(x)
x = Dropout(0.2)(x)
x = Dense(10,activation='softmax',name='d2')(x)
model = Model(inputs=input_layer, outputs=x)

#
# -------------------------------  方式一 --------------------------------------------
LR_mult_dict = {}
LR_mult_dict['c1']=1
LR_mult_dict['c2']=1
LR_mult_dict['d1']=2
LR_mult_dict['d2']=2

# -------------------------------  方式二 --------------------------------------------
LR_mult_dict = {layer.name:10 for layer in model.layers}



# Setting up optimizer
base_lr = 0.1
momentum = 0.9
optimizer = M_Nadam(lr=base_lr,multipliers=LR_mult_dict)

# callbacks
checkpoint = ModelCheckpoint('weights.h5', monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min')

#compiling
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# training
model.fit(x = X_train, y=Y_train,callbacks=[checkpoint], batch_size=100, epochs=2)

# testing
_,score =model.evaluate(x=X_test, y=Y_test, batch_size=100)
