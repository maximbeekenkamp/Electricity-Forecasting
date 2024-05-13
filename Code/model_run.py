import math
import sys
import tensorflow as tf
import numpy as np
import time
import scipy.io as io
import matplotlib.pyplot as plt

from preprocessing import DataSet
from fnn import FNN


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
        bs = hyperparameters["bs"]
        tsbs = hyperparameters["tsbs"]
        dropout_rate = hyperparameters["dropout_rate"]

        param = DataSet(states, bs, tsbs, self.tf_data_type)
        model = FNN(self.tf_data_type, dropout_rate)

        if not load_model_bool:
            # optimal_lr = self.find_optimal_learning_rate(states, hyperparameters)
            # hyperparameters["lr"] = optimal_lr # Disabled for now, as I've already found the optimal learning rate
            # sys.exit()

            start_time = time.perf_counter()
            time_step_0 = time.perf_counter()
            self.builder(hyperparameters, save_model_to, model, param, time_step_0)
            stop_time = time.perf_counter()
            print("Elapsed time (secs): %.3f" % (stop_time - start_time))

        loaded_model = io.loadmat(save_model_to + "/Weight_bias.mat")

        x_pred, y_pred = self.forecast(loaded_model, model, param)

        print("Completed forecasting")
        print("x_pred: ", x_pred)
        print("y_pred: ", y_pred)

    def builder(self, hyperparameters, save_model_to, model, param, time_step_0):
        """
        builds the model.
        """
        net = hyperparameters["net"]
        epochs = hyperparameters["epochs"]
        lr = hyperparameters["lr"]
        decay_rate = hyperparameters["lr decay rate"]

        W, b = model.hyper_initial_fnn(net)

        optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

        n = 0

        train_loss = np.zeros((epochs + 1, 1))
        test_loss = np.zeros((epochs + 1, 1))

        while n <= epochs:
            new_lr = lr / (1 + decay_rate * n / epochs)
            optimiser.learning_rate = new_lr

            x_train, y_train = param.minibatch()
            train_dict, W, b = model.nn_train(optimiser, W, b, x_train, y_train)

            loss = train_dict["loss"]

            if n%10 == 0:
                x_test, y_test = param.testbatch()
                y_pred = model.fnn(W, b, x_test) 
                err = tf.reduce_mean(tf.square(y_test - y_pred))
                err = err.numpy()
                time_step_1000 = time.perf_counter()
                T = time_step_1000 - time_step_0
                print(
                    "Step: %d, Loss: %.4f, Test L2 error: %.4f, Time (secs): %.4f"
                    % (n, loss, err, T)
                )
                time_step_0 = time.perf_counter()

            train_loss[n, 0] = loss
            test_loss[n, 0] = err
            n += 1

        x_test, y_test = param.finalbatch()
        y_pred = model.fnn(W, b, x_test)
        err = tf.reduce_mean(tf.square(y_test - y_pred))
        print("Final L2 error: %.4f" % err)
        err = np.array([err])
        np.savetxt(save_model_to + "/err", err, fmt="%f")
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
        
        self.plotLearning(train_loss, test_loss)

    def find_optimal_learning_rate(
        self, states, hyperparameters, start_lr=0.1, end_lr=0.2
    ):
        """
        Finds the optimal learning rate for the model.
        Also plots the learning rate range test.
        By gradually changing the start_lr and end_lr, we can hone in on the optimal learning rate.

        Args:
            states (list): List of states.
            hyperparameters (dict): Dictionary of hyperparameters.
            start_lr (float, optional): Sets the start point for the lr exploration. Defaults to 0.0015.
            end_lr (float, optional): Sets the end point for the lr exploration. Defaults to 0.0018.

        Returns:
            float: The optimal learning rate.
        """
        net = hyperparameters["net"]
        epochs = hyperparameters["epochs"]
        decay_rate = hyperparameters["lr decay rate"]
        bs = hyperparameters["bs"]
        tsbs = hyperparameters["tsbs"]
        dropout_rate = hyperparameters["dropout_rate"]

        errors = []
        best_lr = None
        lowest_err = float("inf")
        learning_rates = []

        for lr in np.arange(start_lr, end_lr, (end_lr - start_lr) / 10):
            try:
                param = DataSet(states, bs, tsbs, self.tf_data_type)
                model = FNN(self.tf_data_type, dropout_rate)
                W, b = model.hyper_initial_fnn(net)

                print("Testing learning rate: %.10f" % lr)
                learning_rates.append(lr)
                optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

                n = 0

                train_loss = np.zeros((epochs + 1, 1))
                test_loss = np.zeros((epochs + 1, 1))

                while n <= epochs:
                        new_lr = lr / (1 + decay_rate * n / epochs)
                        optimiser.learning_rate = new_lr

                        x_train, y_train = param.minibatch()
                        train_dict, W, b = model.nn_train(optimiser, W, b, x_train, y_train)

                        loss = train_dict["loss"]

                        if n % 100 == 0:
                            x_test, y_test = param.testbatch()
                            y_pred = model.fnn(W, b, x_test)
                            err = tf.reduce_mean(tf.square(y_test - y_pred))
                            err = err.numpy()
                            print("Step: %d, Loss: %.4f, Test L2 error: %.4f" % (n, loss, err))

                        train_loss[n, 0] = loss
                        test_loss[n, 0] = err
                        n += 1
            except KeyboardInterrupt or ValueError as e:
                print("Interrupted")
                break


            x_test, y_test = param.finalbatch()
            y_pred = model.fnn(W, b, x_test)
            err = tf.reduce_mean(tf.square(y_test - y_pred))
            print("Final L2 error: %.4f" % err)
            errors.append(err)
            if err < lowest_err:
                best_lr = lr
                lowest_err = err
            print("Best learning rate so far: %.10f" % best_lr)

        if len(learning_rates) != len(errors):
            learning_rates = learning_rates[:-1]

        print("Best learning rate: %.10f" % best_lr)
        plt.plot(learning_rates, errors)
        plt.xlabel("Learning Rate")
        plt.yscale("log")
        plt.ylabel("Errors")
        plt.title("Learning Rate Range Test")
        plt.show()

        return best_lr

    def plotLearning(self, train_loss, test_loss):
        """
        Plots the loss history.

        Args:
            train_loss (list): List of training loss.
            test_loss (list): List of testing loss.
        """
        plt.rcParams.update({"font.size": 15})
        num_epoch = train_loss.shape[0]
        x = np.linspace(1, num_epoch, num_epoch)
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, train_loss[:, 0], color="blue", label="Training Loss")
        ax.set_yscale("log")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        ax.set_title("Training Loss, Log Scale")

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, test_loss[:, 0], color="red", label="Testing Error")
        ax.set_yscale("log")
        ax.set_ylabel("Error")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        ax.set_title("Testing Error, Log Scale")

        ########## NOT LOG PlOTS
        plt.rcParams.update({"font.size": 15})
        num_epoch = train_loss.shape[0]
        x = np.linspace(1, num_epoch, num_epoch)
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, train_loss[:, 0], color="blue", label="Training Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        ax.set_title("Training Loss, Not Log Scale")

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, test_loss[:, 0], color="red", label="Testing Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        ax.set_title("Testing Loss, Not Log Scale")

        plt.show()


    def forecast(self, loaded_model, model, param):
        """
        Applies the now trained model to the 2023-2027 data.

        Args:
            loaded_model (.mat file): The weights and biases of the trained model.
            model (fnn class object): The class containing fnn functions.
            param (dataset class object): The class containing the data.
        """
        W_tf = [tf.Variable(w, dtype=self.tf_data_type) for w in np.concatenate(loaded_model["W"])]
        b_tf = [tf.Variable(b, dtype=self.tf_data_type) for b in np.concatenate(loaded_model["b"])]

        y_pred = model.fnn(W_tf, b_tf, param.x_pred)

        return param.x_pred, y_pred