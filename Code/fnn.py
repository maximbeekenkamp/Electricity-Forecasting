import tensorflow as tf
import numpy as np


class FNN:
    def __init__(self, tf_data_type, dropout_rate):
        self.tf_data_type = tf_data_type
        self.W, self.b = None, None
        self.dropout_rate = dropout_rate

    def hyper_initial_fnn(self, layers):
        """
        Initialises the weights and biases for a FNN.

        Args:
            layers (list): list description of model architecture.

        Returns:
            Tuple: Initialised weights and biases.
        """
        # with tf.device("/CPU:0"):  # NOTE: if not running on Apple Silicon, comment and de-tab below
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l - 1]
            out_dim = layers[l]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            weight = tf.Variable(
                tf.random.normal(
                    shape=[in_dim, out_dim],
                    stddev=std,
                    dtype=self.tf_data_type,
                ),
                dtype=self.tf_data_type,
                name=f"weight_{l}",
                trainable=True,
            )
            bias = tf.Variable(
                tf.zeros(shape=[out_dim], dtype=self.tf_data_type),
                dtype=self.tf_data_type,
                name=f"bias_{l}",
                trainable=True,
            )

            W.append(weight)
            b.append(bias)
        return W, b

    def fnn(self, W, b, X, training=False):
        """
        Forward pass of the FNN network.

        Args:
            W (ndarray): Weights.
            b (ndarray): Biases.
            X (ndarray): Network inputs.

        Returns:
            Tensor object of ndarray: Output of the dense network.
        """
        L = len(W)
        for i in range(L - 1):
            X = tf.nn.leaky_relu(tf.add(tf.matmul(X, W[i]), b[i]))
            X = self.batch_norm(X)
            if training:
                X = tf.nn.dropout(X, rate=self.dropout_rate)
        Y = tf.nn.leaky_relu(tf.add(tf.matmul(X, W[-1]), b[-1]))

        return Y
    
    def batch_norm(self, X, epsilon=1e-5):
        """
        Batch normalization from scratch.

        Args:
            X (Tensor): Input tensor.
            epsilon (float): Small value to avoid division by zero.

        Returns:
            Tensor: Normalised tensor.
        """
        mean, variance = tf.nn.moments(X, axes=0)
        normalised_X = (X - mean) / tf.sqrt(variance + epsilon)
        return normalised_X

    def save_W_b(self, W, b):
        """
        Save the weights and biases of the network.

        Args:
            W (ndarray): Weights.
            b (ndarray): Biases.

        Returns:
            Tuple of ndarrays: Reformatted weights and biases.
        """
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

    # @tf.function(jit_compile=True) # NOTE:cannot run jit on Apple Silicon
    def nn_train(self, optimizer, W, b, X, Y):
        """
        Backward pass of the DeepONet, using the Adam optimizer.

        Args:
            optimizer (tf function): Adam optimizer.
            W (Tensor object of ndarray): Weights of the network.
            b (Tensor object of ndarray): Biases of the network.
            X (Tensor object of ndarray): Network inputs.
            Y (Tensor object of ndarray): Network outputs.

        Returns:
            Tuple: Returns a dictionary containing the loss and the predicted
            solution, alongside the weights and biases of the model.
        """
        with tf.GradientTape() as tape:
            y_pred = self.fnn(W, b, X, training=True)
            loss = tf.reduce_mean(tf.square(Y - y_pred))

        joint_vars = [val for pair in zip(W, b) for val in pair]

        gradients = tape.gradient(loss, joint_vars)
        # print("gradients: ", gradients)
        optimizer.apply(gradients, joint_vars)

        loss_dict = {"loss": loss, "Y_pred": y_pred}
        return loss_dict, W, b
