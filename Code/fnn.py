import tensorflow as tf
import numpy as np


class FNN:
    def __init__(self, tf_data_type):
        self.tf_data_type = tf_data_type

    def hyper_initial_fnn(self, layers):
        """
        Initialises the weights and biases for a FNN.

        Args:
            layers (list): list description of model architecture.

        Returns:
            Tuple: Initialised weights and biases.
        """
        with tf.device('/CPU:0'): # if running on laptop running Apple Silicon, comment and de-tab otherwise
            L = len(layers)
            W = []
            b = []
            for l in range(1, L):
                in_dim = layers[l - 1]
                out_dim = layers[l]
                std = np.sqrt(2.0 / (in_dim + out_dim))
                weight = tf.Variable(
                    tf.random.normal(shape=[in_dim, out_dim], stddev=std, dtype=self.tf_data_type),
                    dtype=self.tf_data_type
                )
                bias = tf.Variable(
                    tf.zeros(shape=[1, out_dim], dtype=self.tf_data_type),
                    dtype=self.tf_data_type
                )

                W.append(weight)
                b.append(bias)
        return W, b

    def fnn(self, W, b, X):
        """
        Forward pass of the FNN network.

        Args:
            W (Tensor object of ndarray): Weights.
            b (Tensor object of ndarray): Biases.
            X (Tensor object of ndarray): Network inputs.

        Returns:
            Tensor object of ndarray: Output of the dense network.
        """
        L = len(W)
        for i in range(L - 1):
            X = tf.nn.leaky_relu(tf.add(tf.matmul(X, W[i]), b[i]))
        Y = tf.nn.relu(tf.add(tf.matmul(X, W[-1]), b[-1]))

        return Y

    # Saving helper functions
    def save_W_b(self, W, b):
        L = len(W)
        W_out = []
        b_out = []
        for i in range(L):
            W_out.append(W[i].numpy())
            x = b[i]
            b_out.append(np.reshape(x, (-1)))

        return W_out, b_out

    def save_W(self, W):
        L = len(W)
        W_out = []
        for i in range(L):
            W_out.append(W[i].numpy())

        return W_out

    @tf.function(jit_compile=True)
    def nn_train(self, optimizer, W, b, X, Y):
        """
        Backward pass of the DeepONet, using the Adam optimizer.

        Args:
            optimizer (tf function): Adam optimizer.
            W (Tensor object of ndarray): Weights of the network.
            b (Tensor object of ndarray): Biases of the network.
            X (Tensor object of ndarray): Network inputs.

        Returns:
            Tuple: Returns a dictionary containing the loss and the predicted 
            solution, alongside the weights and biases of the model.
        """
        with tf.GradientTape() as tape:
            y_pred = self.fnn(W, b, X) # this is the forward pass
            loss = tf.reduce_mean(tf.square(Y - y_pred) / (tf.square(Y) + 1e-4))

        gradients = tape.gradient(loss, ([W]+[b]))
        optimizer.apply_gradients(zip(gradients, ([W]+[b])))

        loss_dict = {"loss": loss, "Y_pred": y_pred}
        return loss_dict, W, b