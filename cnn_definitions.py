import math
import keras

# This class contains methods to create different CNN models.

def cnn1(data_shape, optimizer):
    """
    Original model from the website https://www.kaggle.com/amro91/planes-classification-with-cnn
    :param data_shape:
    :param optimizer:
    :return:
    """
    kernel_size = 3

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(16, (kernel_size), strides=(1, 1), padding='valid',
                     input_shape=(data_shape[1], data_shape[2], data_shape[3])))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #     model.add(MaxPooling2D((2,2)))

    model.add(keras.layers.Conv2D(32, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #     model.add(MaxPooling2D((2,2)))

    model.add(keras.layers.Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #     model.add(MaxPooling2D((2,2)))

    model.add(keras.layers.Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Flatten())
    #     model.add(Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def step_decay1(epoch):
    """
    Returns current learning rate for the given epoch.
    """
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#    if epoch:
#        lrate = initial_lrate/np.sqrt(epoch)
#    else:
#        return initial_lrate
    return lrate

def cnn2(data_shape, optimizer):
    """
    Modification of the original CNN from the website, i.e. a few commented lines https://www.kaggle.com/amro91/planes-classification-with-cnn
    :param data_shape:
    :param optimizer:
    :return:
    """
    kernel_size = 3

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(16, (kernel_size), strides=(1, 1), padding='valid',
                     input_shape=(data_shape[1], data_shape[2], data_shape[3])))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D((2,2))) # 1

    model.add(keras.layers.Conv2D(32, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D((2,2))) # 2

    model.add(keras.layers.Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D((2,2))) # 3

    model.add(keras.layers.Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D((2,2))) # 4

    model.add(keras.layers.Conv2D(64, (kernel_size), strides=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.MaxPooling2D((2,2))) # 5

    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(64, activation='relu')) # 6
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def cnn3(data_shape, optimizer):
    """
    Custom CNN
    :return:
    """

    kernel_size = 3

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(512, (kernel_size), strides=(1, 1), padding='valid',
                     input_shape=(data_shape[1], data_shape[2], data_shape[3])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
