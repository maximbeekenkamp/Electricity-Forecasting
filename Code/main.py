import os
import tensorflow as tf
import numpy as np
import scipy.io as io
import shutil
import argparse

from model_run import Runner

np.random.seed(1234)
tf_data_type = tf.float32
# tf.config.list_physical_devices("GPU")
tf.keras.backend.clear_session()


def main(load_model):
    # Create directories
    # tf.config.list_physical_devices('GPU') # NOTE: if running on Apple Silicon without tensorflow-metal, you have to run on CPU
    current_directory = os.getcwd()
    model_dir = "/Saved_Model"
    save_model_to = current_directory + model_dir

    if not load_model:
        # Remove existing results
        if os.path.exists(save_model_to):
            shutil.rmtree(save_model_to)

        os.makedirs(save_model_to)

    in_dim = 23  # 15 features (see README) + 8 states
    out_dim = 1  # 1 feature (Price)

    hyperparameters = {
        "net": [in_dim, 32, 32, out_dim],
        "bs": 10,
        "tsbs": 32,
        "epochs": 1000,
        "lr": 0.13,
        "lr decay rate": 0.001,
        "dropout_rate": 0.2,

    }

    states = ["KY", "TN", "AL", "MS", "GA", "SC", "NC", "FL"]

    io.savemat(save_model_to + "/hyperparameters.mat", mdict=hyperparameters)

    # Initialise and run the model
    network = Runner(tf_data_type)
    network.run(hyperparameters, save_model_to, load_model, states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", action="store_true", help="Load model from Saved_Model directory"
    )
    args = parser.parse_args()

    if args.load:
        load_model = True
    else:
        load_model = False

    main(load_model)
