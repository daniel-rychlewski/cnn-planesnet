# Contains useful utility methods for all things CNN.
# The pruning related methods prune_model, get_total_channels, get_model_apoz are taken from kerassurgeon examples.

import keras
import kerassurgeon
import math
import pandas as pd
import numpy as np
import keras.backend as K
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from kerassurgeon.identify import get_apoz
from keras.models import load_model
from tqdm import tqdm

# For vector quantization, use tf-nightly-gpu (around 16 times than the CPU-based tf-nightly)
import tensorflow as tf

def prune_model(model, apoz_df, n_channels_delete):
    # Identify 5% of channels with the highest APoZ in model
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = kerassurgeon.Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'],
                                  dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name),
                        channels=channels)
    # Delete channels
    return surgeon.operate()


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


def get_model_apoz(model, generator):
    # Get APoZ (Average Percentage of Zeros)
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df

def prune(model, x_train, x_test, y_train, y_test, batch_size, epochs):
    """
    Model pruning function
    """

    # Lese Parameter von JSON ein
    parameterFile = open("parameters.json", "r")
    data = json.load(parameterFile)
    parameterFile.close()
    percent_pruning = int(data["pruning"]["percent_per_step"])
    total_percent_pruning = int(data["pruning"]["percent_total_up_to"])

    percent_pruned = 0 # to be edited by the user if necessary, but 0 is also fine and the default value
    # If percent_pruned > 0, continue pruning from previous checkpoint
    # Read model here in case it needs to be read at all so that the number of total channels can be successfully retrieved for the model read as opposed to model passed as a parameter
    if percent_pruned > 0:
        model = load_model("trained_pruned_model_toloadfrom3rdbatch_"+str(percent_pruned)+".h5")

    total_channels = get_total_channels(model)
    # the int type cast below is the reason why small percentages, i.e. below 1%, won't work (because n_channels_delete will be zero, so nothing will be pruned)
    n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))

    # Set up data generators
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size)
    train_steps = train_generator.n // train_generator.batch_size

    test_datagen = ImageDataGenerator()

    validation_generator = test_datagen.flow(
        x_test,
        y_test,
        batch_size=batch_size)
    val_steps = validation_generator.n // validation_generator.batch_size

    # Incrementally prune the network, retraining it each time
    while percent_pruned < total_percent_pruning:
        # Prune the model
        apoz_df = get_model_apoz(model, validation_generator)
        percent_pruned += percent_pruning
        print('pruning up to ', str(percent_pruned),
              '% of the original model weights')
        model = prune_model(model, apoz_df, n_channels_delete)
        print(model.summary())

        # Clean up tensorflow session after pruning and re-load model
        model.save("trained_pruned_model_"+ str(percent_pruned) + ".h5")

        del model
        K.clear_session()
        tf.reset_default_graph()

        model = load_model("trained_pruned_model_" + str(percent_pruned) + '.h5')

        # Re-train the model
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])

        csv_logger = CSVLogger("trained_pruned_model" + str(percent_pruned) + '.csv')
        model.fit_generator(train_generator,
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=val_steps,
                            workers=4,
                            callbacks=[csv_logger])

    # Evaluate the final model performance
    loss = model.evaluate_generator(validation_generator,
                                    validation_generator.n //
                                    validation_generator.batch_size)
    print('pruned model loss: ', loss[0], ', acc: ', loss[1])


def calculate_tflite_parameters(model_path, x_train, x_test, y_train, y_test):
    """
    Reads and evaluates the tflite model.
    """
    K.clear_session()
    tf.reset_default_graph()

    # Source (I have changed the code to cover training and testing accuracy separately): https://danielhunter.io/tensorflow-lite-on-a-raspberry-pi/

    # Load TFLite model and allocate tensors
    interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # for debugging purposes
    # input_shape = input_details[0]['shape']

    # Training
    correct = tflite_evaluate(input_details, interpreter, output_details, x_train, y_train)
    print('Tensorflow Lite Model - Training accuracy:', correct * 1.0 / len(y_train))

    # Testing
    correct = tflite_evaluate(input_details, interpreter, output_details, x_test, y_test)
    print('Tensorflow Lite Model - Test accuracy:', correct * 1.0 / len(y_test))


def tflite_evaluate(input_details, interpreter, output_details, x_test, y_test):
    correct = 0
    for img, label in tqdm(zip(x_test, y_test)):
        interpreter.set_tensor(input_details[0]['index'], np.asarray([img], dtype=np.float32))

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # 0 - 0.5: no plane, i.e. rounded down, everything above that (until 1.0) apparently is a plane and is classified that way
        if int(np.around(output_data[0][0])) == label:
            correct += 1
        # else:
        #     # Show what the mismatches look like. Put a breakpoint below to stop at the image
        #     plt.imshow((img * 255).astype(np.uint8))
        #     plt.show()
    return correct


def quantize(x_train, x_test, y_train, y_test, model_name):
    """
    Quantizes the model, evaluates its accuracies and losses for the given x and y values and saves it to disk
    """

    parameterFile = open("parameters.json", "r")
    data = json.load(parameterFile)
    parameterFile.close()

    if int(data["pruning"]["enabled"]) == 1:
        percent_pruned = 1  # minimum implicitly 1 if not specified in prune()
        percent_pruning = int(data["pruning"]["percent_per_step"])  # increment
        total_percent_pruning = int(data["pruning"]["percent_total_up_to"])  # max

        while percent_pruned < total_percent_pruning:
            K.clear_session()
            tf.reset_default_graph()

            model_name = "trained_pruned_model_" + str(percent_pruned) + ".h5"

            converter = tf.lite.TFLiteConverter.from_keras_model_file("D:/CNNsPlanesNet/" + model_name)
            converter.post_training_quantize = True
            tflite_quantized_model = converter.convert()

            # Write to disk
            open("quantized_pruned_model_"+ str(percent_pruned) +".tflite", "wb").write(tflite_quantized_model)

            calculate_tflite_parameters("quantized_pruned_model_" + str(percent_pruned) + ".tflite", x_train, x_test, y_train, y_test)

            # Next model, i.e. next pruning percentage
            percent_pruned = percent_pruned + percent_pruning

    else:
        # No pruning -> no need to manually determine model_name. Just take the parameter
        # Logic: main.py created only one CNN, so only quantize this one CNN. If main.py is called multiple times, each time the respective model will be quantized
        converter = tf.lite.TFLiteConverter.from_keras_model_file("D:/CNNsPlanesNet/" + model_name)
        converter.post_training_quantize = True
        tflite_quantized_model = converter.convert()

        # Write to disk
        open("quantized_"+ model_name +".tflite", "wb").write(tflite_quantized_model)

        calculate_tflite_parameters("quantized_" + model_name + ".tflite", x_train, x_test, y_train, y_test)