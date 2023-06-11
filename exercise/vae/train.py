

from tensorflow.keras.datasets import mnist
from ae import Autoencoder
from numpy.lib.npyio import BagObj 




# load dataset 
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255 # normalize all the sample between 0 and 1 
    x_train = x_train.reshape(x_train.shape + (1,)) # add one more dimension for channel dimension 
    x_test = x_test.astype("float32") / 255 # normalize all the sample between 0 and 1 
    x_test = x_test.reshape(x_test.shape + (1,)) # add one more dimension for channel dimension 

    return x_train, y_train, x_test, y_test


# training function 
def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = Autoencoder(
        input_shape = (28, 28, 1),
        conv_filters = (32, 64, 64, 64),
        conv_kernels = (3, 3, 3, 3),
        conv_strides = (1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder



LEARNING_RATE = 0.0001
BATCH_SIZE = 32 
EPOCHS = 20


if __name__ == "__main__": 

    x_train, _, _, _ = load_mnist()
    autoencoder = train(x_train[:500], LEARNING_RATE, BATCH_SIZE, EPOCHS)
