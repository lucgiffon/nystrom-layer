"""
Simple script that shows how a nystrom layer should be implemented.

This doesn't aim at giving good results, just to show a simple implementation of the Nyström layer in a convolutional neural network.
"""

import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Flatten, Input, Lambda, concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras_kernel_functions import keras_linear_kernel


def datagen_fixed_batch_size(x, y, batch_size, x_sub=None, p_datagen=ImageDataGenerator()):
    """
    Wrap a data generator so that:
     - it always output batches of the same size
     - it gives a subsample along with each batch

    :param x: observation data
    :param y: label data
    :param x_sub: list of base of subsample (each base must be of size batch_size)
    :param p_datagen: the initial data generator to wrap
    :return:
    """
    if x_sub is None:
        x_sub = []
    for x_batch, y_batch in p_datagen.flow(x, y, batch_size=batch_size):
        if x_batch.shape[0] != batch_size:
            continue
        yield [x_batch] + x_sub, y_batch

def build_conv_model(input_shape):
    """
    Create a simple sequential convolutional model

    :param input_shape: tuple containing the expected input data shape
    :return: keras model object
    """

    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    return model

def init_number_subsample_bases(nys_size, batch_size):
    """
    Return the number of bases and the size of the zero padding for initialization of the model.

    :param nys_size: The number of subsample in the Nystrom approximation.
    :param batch_size: The batch size in the final model.
    :return: number of bases, size of the zero padding
    """
    remaining = nys_size % batch_size
    quotient = nys_size // batch_size
    if nys_size == 0 or batch_size == 0:
        raise ValueError
    if remaining == 0:
        return quotient, 0
    elif quotient == 0:
        return 1, batch_size - remaining
    else:
        return quotient + 1, batch_size - remaining

if __name__ == "__main__":
    # model meta parameters
    # ---------------------
    batch_size = 128
    epochs = 1
    num_classes = 10
    nys_size = 8

    # data preparation
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(x_train)

    # subsample for nystrom layer preparation
    # ---------------------------------------
    # keras needs all its input to have the same shape. The subsample input to the model is then divided in so called "bases" of the same size than the batch, all stored in a list.
    # The last base may not be full of samples so it mmust be padded with zeros. Those zeros will be cut off in the model computation.
    # If you have a suggestion on how to better implement it, feel free to suggest.
    nb_subsample_bases, zero_padding_base = init_number_subsample_bases(nys_size, batch_size)
    subsample_indexes = np.random.permutation(x_train.shape[0])[:nys_size]
    nys_subsample = x_train[subsample_indexes]
    zero_padding_subsample = np.zeros((zero_padding_base, *nys_subsample.shape[1:]))
    nys_subsample = np.vstack([nys_subsample, zero_padding_subsample])
    list_subsample_bases = [nys_subsample[i * batch_size:(i + 1) * batch_size] for i in range(nb_subsample_bases)]

    # convolution layers preparation
    # ------------------------------
    convmodel_func = build_conv_model(x_train[0].shape)  # type: keras.models.Sequential
    convmodel_func.add(Flatten())


    # processing of the input by the convolution
    # ------------------------------------------
    input_x = Input(shape=x_train[0].shape, name="x")
    conv_x = convmodel_func(input_x)

    # processing of the subsample by the convolution
    # ----------------------------------------------
    # definition of the list of input bases
    input_repr_subsample = [Input(batch_shape=(batch_size, *x_train[0].shape)) for _ in range(nb_subsample_bases)]

    if nb_subsample_bases > 1:
        input_subsample_concat = concatenate(input_repr_subsample, axis=0)
    else:
        input_subsample_concat = input_repr_subsample[0]

    # remove the zeros from the input subsamplebefore actual computation in the network
    slice_layer = Lambda(lambda input: input[:nys_size], output_shape=lambda shape: (nys_size, *shape[1:]))
    input_subsample_concat = slice_layer(input_subsample_concat)
    conv_subsample = convmodel_func(input_subsample_concat)

    # definition of the nystrom layer
    # -------------------------------
    kernel_function = lambda *args, **kwargs: keras_linear_kernel(*args, **kwargs, normalize=True)
    # kernel function as Lambda layer
    kernel_layer = Lambda(kernel_function, output_shape=lambda shapes: (shapes[0][0], nys_size))
    kernel_vector = kernel_layer([conv_x, conv_subsample])
    # weight matrix of the nystrom layer
    input_classifier = Dense(nys_size, use_bias=False, activation='linear')(kernel_vector) # metric matrix of the Nyström layer

    # final softmax classification layer
    # ----------------------------------
    classif = Dense(num_classes, activation="softmax")(input_classifier)

    # finalization of model, compilation and training
    # -----------------------------------------------
    model = Model([input_x] + input_repr_subsample, [classif])
    adam = Adam(lr=.1)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit_generator(datagen_fixed_batch_size(x_train, y_train, batch_size, list_subsample_bases, datagen),
                        steps_per_epoch=int(x_train.shape[0] / batch_size),
                        epochs=epochs)
