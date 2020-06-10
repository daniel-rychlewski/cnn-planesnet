# Quantizes numerous existing CNN models built in Keras in a batch WITHOUT checking accuracy of the quantized model (not desirable, replaced by a new batch_execute mode called "both").
import tensorflow as tf
# Specify directory
dir = "D:/CNNsPlanesNet/"
# Specify first chunk of model name
model_name = "trained_model_"

epoch_suffix = "10"
batch_size_suffix = "256"
while int(epoch_suffix) <= 100:
    while int(batch_size_suffix) <= 8192:
        converter = tf.lite.TFLiteConverter.from_keras_model_file(dir + model_name + epoch_suffix + "epochs" + "_" + batch_size_suffix + "batch_size" + ".h5")
        converter.post_training_quantize = True
        tflite_quantized_model = converter.convert()
        open("quantized_" + model_name + epoch_suffix + "epochs" + "_" + batch_size_suffix + "batch_size.tflite", "wb").write(tflite_quantized_model)
        batch_size_suffix = str(int(batch_size_suffix) * 2)
    batch_size_suffix = "256"
    epoch_suffix = str(int(epoch_suffix) + 10)