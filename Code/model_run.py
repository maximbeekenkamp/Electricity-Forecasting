import tensorflow as tf
import numpy as np
import time
import scipy.io as io
import matplotlib.pyplot as plt

from Code.preprocessing import DataSet
from Code.fnn import FNN


class Runner:
    def __init__(self, tf_data_type):
        self.tf_data_type = tf_data_type

    def run(self, hyperparameters, save_model_to, load_model_bool, states):
        """
        Runs the model.

        Args:
            hyperparameters (dict): Dictionary of hyperparameters.
            save_model_to (str): Directory to save model to.
            load_mode_bool (bool): Boolean for whether the 
            model should be loaded in, or built.
        """

        param = DataSet(states)
        model = FNN(self.tf_data_type)

        if not load_model_bool:
            start_time = time.perf_counter()
            time_step_0 = time.perf_counter()
            self.builder(model, hyperparameters)
            stop_time = time.perf_counter()
            print("Elapsed time (secs): %.3f" % (stop_time - start_time))
        
        loaded_model = io.loadmat(save_model_to + "/Weight_bias.mat")
        self.plot()

        self.forecast(loaded_model, model, param, save_model_to)
        

    def builder(self, hyperparameters, save_model_to, model, param):
        """
        builds the model.
        """
        net = hyperparameters["net"]
        bs = hyperparameters["bs"]
        tsbs = hyperparameters["tsbs"]
        epochs = hyperparameters["epochs"]
        lr = hyperparameters["lr"]

        x_train = param.x_train
        y_train = param.y_train
        x_test = param.x_test
        y_test = param.y_test


        W, b = model.hyper_initial_fnn(net)

        optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

        n = 0

        train_loss = np.zeros((epochs + 1, 1))
        test_loss = np.zeros((epochs + 1, 1))
        while n <= epochs:
            train_dict, W, b = model.nn_train(
                optimiser, x_train, y_train, W, b
            )

            loss = train_dict["loss"]

            if n % 50 == 0:
                y_pred = model.fnn(W, b, x_test) # done to prevent reloading model
                err = np.mean((y_test - y_pred) ** 2 / (y_test**2 + 1e-4))
                err = np.reshape(err, (-1, 1))
                time_step_1000 = time.perf_counter()
                T = time_step_1000 - time_step_0
                print(
                    "Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f"
                    % (n, loss, err, T)
                )
                time_step_0 = time.perf_counter()

            train_loss[n, 0] = loss
            test_loss[n, 0] = err
            n += 1

        y_pred = model.fnn(W, b, x_test)
        err = np.mean((y_test - y_pred) ** 2 / (y_test**2 + 1e-4))
        err = np.reshape(err, (-1, 1))
        np.savetxt(save_model_to + "/err", err, fmt="%e")
        io.savemat(
            save_model_to + "/Out.mat",
            mdict={
                "x_test": x_test,
                "y_test": y_test,
                "y_pred": y_pred,
            },
        )

        ######################
        ### Saving Results ###
        ######################

        W_fnn_save, b_fnn_save = model.save_W_b(W, b)

        W_b_dict_save = {
            "W": W_fnn_save,
            "b": b_fnn_save,
        }

        io.savemat(save_model_to + "/Weight_bias.mat", W_b_dict_save)
        print("Completed storing unpruned weights and biases")


    def plot(self):
        """
        Plots the model.
        """
        pass

    def forecast(self, loaded_model, model, param, save_model_to):
        """
        Applies the now trained model to the 2023-2027 data.

        Args:
            loaded_model (.mat file): The weights and biases of the trained model.
            model (fnn class object): The class containing fnn functions.
            param (dataset class object): The class containing the data.
            save_model_to (str): The directory to save the results to.
        """
        pass