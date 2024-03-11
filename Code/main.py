import os
import tensorflow as tf
import numpy as np
import scipy.io as io
import shutil
import argparse

from Code.model_run import Runner

np.random.seed(1234)
tf_data_type = tf.float64
tf.keras.backend.clear_session()

def main(load_model):
    # Create directories
    current_directory = os.getcwd()
    model_dir = "/Saved_Model"
    save_model_to = current_directory + model_dir

    if not load_model:
        # Remove existing results
        if os.path.exists(save_model_to):
            shutil.rmtree(save_model_to)

        os.makedirs(save_model_to)

    hyperparameters = {
        "net": [2, 8, 8, 8],
        "bs": 50,
        "tsbs": 20,
        "epochs": 200,
        "lr": 0.001,
    }

    states = ["KY", "TN", "AL", "MS", "GA", "SC", "NC", "FL"]

    io.savemat(save_model_to + "/hyperparameters.mat", mdict=hyperparameters)

    # Initialise and run the model
    network = Runner(tf_data_type)
    network.run(hyperparameters, save_model_to, load_model, states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load model from Saved_Model directory")
    args = parser.parse_args()

    if args.load_model:
        load_model = True
    else:
        load_model = False

    main(load_model)