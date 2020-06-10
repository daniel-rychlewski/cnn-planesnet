# Template of "Building Deep Learning Applications with Keras 2.0" taken from lynda.com, Exercise Files\04\save_trained_model final.py
# Source of template, which I have changed for my purposes: https://www.kaggle.com/amro91/planes-classification-with-cnn

# Create graphs
import matplotlib
import matplotlib.pyplot as plt
# High-level framework für CNN-Erstellung
import keras
import numpy as np
# JSON einlesen
import pandas as pd
# Aufteilung training / testing durchführen
import sklearn.model_selection
import json
import sys

import cnn_definitions
import cnn_utils

training_data_df = pd.read_json("planesnet.json")

# Read JSON parameters
parameter_file = open("parameters.json", "r")
data = json.load(parameter_file)
parameter_file.close()
epochs = int(data["epochs"]["value"])
batch_size = int(data["batch_size"]["value"])
pruning_enabled = int(data["pruning"]["enabled"])
pruning_epochs = int(data["pruning"]["epochs"])
quantization_enabled = int(data["quantization"]["enabled"])

# the file name represents all parameters so that one knows which configuration has been used based on the file name
suffix = "_"+str(epochs)+"epochs_"+str(batch_size)+"batch_size"
sys.stdout = open("output"+suffix, 'w')

X = []
for d in training_data_df['data']: # the actual image
    d = np.array(d)
    X.append(d.reshape(( 3, 20 * 20)).T.reshape( (20,20,3) ))
X = np.array(X)
Y = np.array(training_data_df['labels']) # 1 = plane, 0 = no plane

# splitting the data into training and test sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.20)

# normalizing the data
scalar = sklearn.preprocessing.MinMaxScaler()
scalar.fit(x_train.reshape(x_train.shape[0],-1).astype(float))
x_train = scalar.transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
x_test = scalar.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)

# Create the model. Model definitions are in cnn_definitions.py
optimizer = keras.optimizers.Adam(lr=0.0001)
model = cnn_definitions.cnn2(x_train.shape, optimizer)

# model = load_model('trained_model.h5') # for retraining purposes (transfer learning), a model can be read here
lrate = keras.callbacks.LearningRateScheduler(cnn_definitions.step_decay1)

# Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    validation_data=(x_test, y_test),
    callbacks = [lrate]
)

# plotting the learning curves
fig, ax = plt.subplots(1,2)
fig.set_size_inches((15,5))
ax[0].plot(range(1,epochs+1), history.history['loss'], c='blue', label='Training loss')
ax[0].plot(range(1,epochs+1), history.history['val_loss'], c='red', label='Validation loss')
ax[0].legend()
ax[0].set_xlabel('epochs')

ax[1].plot(range(1,epochs+1), history.history['acc'], c='blue', label='Training accuracy')
ax[1].plot(range(1,epochs+1), history.history['val_acc'], c='red', label='Validation accuracy')
ax[1].legend()
ax[1].set_xlabel('epochs')

matplotlib.pyplot.savefig("my_graph"+suffix+".png")

model_name = "trained_model"+suffix+".h5"
# Save the model to disk
model.save(model_name)
print("Model saved to disk.")

# Parameter Pruning
if pruning_enabled:
    # print(model.summary())
    cnn_utils.prune(model, x_train, x_test, y_train, y_test, batch_size, pruning_epochs)

# Vector Quantization
if quantization_enabled:
    cnn_utils.quantize(x_train, x_test, y_train, y_test, model_name)
