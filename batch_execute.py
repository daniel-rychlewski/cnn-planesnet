# Batch executes the main file with different parameter configurations. Replaces values in parameters.json in doing so
import os
import json

parameterFile = open("parameters.json", "r")
data = json.load(parameterFile)
parameterFile.close()

startCommand = "C:\\Users\\danie\\.conda\\envs\\hsi36\\python.exe D:\\CNNsPlanesNet\\main.py"

mode = "both"

if mode == "epochs":
    epochs_limit = int(data["epochs"]["max"])
    epochs_current = int(data["epochs"]["min"])
    epochs_increment = int(data["epochs"]["increment_by"])
    while epochs_current <= epochs_limit:
        # Write current parameter value to JSON file so that main program runs with correct values
        with open("parameters.json", "w") as outfile:
            data["epochs"]["value"] = str(epochs_current)
            json.dump(data, outfile)
        os.system(startCommand)
        epochs_current = epochs_current + epochs_increment

elif mode == "batch_size":
    batch_limit = int(data["batch_size"]["max"])
    batch_current = int(data["batch_size"]["min"])
    batch_multiply_with = int(data["batch_size"]["multiply_with"])
    while batch_current <= batch_limit:
        # Write current parameter value to JSON file so that main program runs with correct values
        with open("parameters.json", "w") as outfile:
            data["batch_size"]["value"] = str(batch_current)
            json.dump(data, outfile)
        os.system(startCommand)
        batch_current = batch_current * batch_multiply_with

elif mode == "both":
    epochs_limit = int(data["epochs"]["max"])
    epochs_current = int(data["epochs"]["min"])
    epochs_increment = int(data["epochs"]["increment_by"])
    batch_limit = int(data["batch_size"]["max"])
    batch_current = int(data["batch_size"]["min"])
    batch_multiply_with = int(data["batch_size"]["multiply_with"])
    while epochs_current <= epochs_limit:
        while batch_current <= batch_limit:
            # Write current parameter value to JSON file so that main program runs with correct values
            with open("parameters.json", "w") as outfile:
                data["batch_size"]["value"] = str(batch_current)
                json.dump(data, outfile)
            os.system(startCommand)
            batch_current = batch_current * batch_multiply_with

        # Reset batch size for next iteration, but increment epochs
        batch_current = int(data["batch_size"]["min"])
        epochs_current = epochs_current + epochs_increment
        with open("parameters.json", "w") as outfile:
            data["epochs"]["value"] = str(epochs_current)
            json.dump(data, outfile)

