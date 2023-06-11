

from tensorflow.keras import Model 
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
        Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


# tensorflow implementation 
class Autoencoder: 
    '''
    Autoencoder represents a Deep Convolutional autoencoder architecture with 
    mirrored encoder and decoder components
    '''

    def __init__(self, 
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):

        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8] 
        self.conv_kernels = conv_kernels  # [3, 5, 3] 
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2 for bottleneck


        self.encoder = None 
        self.encoder = None 
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build() 


    def summary(self):
        ''' building a summary in case you forget which layers is what you can check with it'''
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()


    # method that compile them before using them 
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)


    # actual train model integrate all functions needed
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, # here are passing the training data
                       x_train, # here is the target data, in our case is reconstruction loss, so it should be the same as x_train
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)


    # whole pipeline for autoencoder 
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder() 


    # whole function for ae
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input)) # apply the encoded input to decoder
        self.model = Model(model_input, model_output, name="autoencoder")


    # decoder 
    def _build_decoder(self):
        decoder_input = self._add_decoder_input() 
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer) # reshape back to the original 
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        # apply value to tensorflow model API
        self.decoder = Model(decoder_input, decoder_output, name="decoder")


    # add decoder input function 
    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input") 

    
    # dense layer function 
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [4, 4, 32] -> 4*4*32 
                                                             # this is what you input before bottleneck
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer
        

    # reshape layer 
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)


    def _add_conv_transpose_layers(self, x):
        ''' add conv tranpose blocks '''
        # loop through all the conv layers in the reverse order 
        # and stop at the first layer 
        # why? because we want to the mirror what's in the forward order 
        # stop when we reach the first one
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] -> [2, 1, 0] normally, but we need to drop the first layer 
            # [0, 1, 2] -> [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)
        return x 

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,      # [24, 24, 1]
            kernel_size=self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        # actication layer that we use sigmoid
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer


    # building the encoder 
    def _build_encoder(self):
        encoder_input = self._add_encoder_input() 
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")


    # add encoder input 
    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    
    def _add_conv_layers(self, encoder_input):
        ''' creates all Convolutional blocks in encoder '''
        x = encoder_input 
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x 


    def _add_conv_layer(self, layer_index, x):
        '''
        adds a convolutional block to a graph of layers, consisting
        of conv 2d + ReLU + batch normalization. 
        '''

        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"encoder_conv_layers_{layer_number}"
        )

        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x 


    def _add_bottleneck(self, x):
        ''' take data we have, and Flatten data and add to bottleneck (Dense Layer).'''
        
        # first store data, so that we can know the shape of the data
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2, 7, 7, 32]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x 


if __name__ == "__main__": 

    autoencoder = Autoencoder(
        input_shape = (28, 28, 1),
        conv_filters = (32, 64, 64, 64),
        conv_kernels = (3, 3, 3, 3),
        conv_strides = (1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
