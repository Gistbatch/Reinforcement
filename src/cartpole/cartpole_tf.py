import numpy as np
import tensorflow as tf
from tensorflow import keras
import cartpole_rbf


class Regressor:
    def __init__(self, dims, learning_rate=0.01):
        print('TF!')
        self.model = keras.Sequential([keras.layers.Dense(1, input_shape=(dims,))])
        optimizer = keras.optimizers.SGD(learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

    def partial_fit(self, X, Y):
        self.model.fit(X, np.array(Y), verbose=0)

    def predict(self, X):
        values = self.model.predict(X)
        return values.sum()


if __name__ == "__main__":
    cartpole_rbf.Regressor = Regressor
    cartpole_rbf.cart_pole()