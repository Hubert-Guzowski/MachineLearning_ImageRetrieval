import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.layers import Layer
from keras.models import Model

'''
-build(input_shape)
-call(input)
-compute_output_shape(input_shape)

The build method is called when the model containing the layer is built. 
This is where you set up the weights of the layer. 
The input_shape is accepted as an argument to the function.

The call method defines the computations performed on the input. 
The function accepts the input tensor as its argument and returns the output tensor 
after applying the required operations.

Finally, we need to define the compute_output_shape function that is required for Keras 
to infer the shape of the output. 
This allows Keras to do shape inference without actually executing the computation. 
The input_shape is passed as the argument.
'''


class DenseK(Layer):
    def __init__(self, K=3, **kwargs):
        self.K = K
        super(DenseK, self).__init__(**kwargs)

    def build(self, input_shape):
        feat_dims = input_shape[-1]
        assert feat_dims % self.K == 0
        self.kernel = self.add_weight(shape=(1, feat_dims),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(1, feat_dims // self.K),
                                    initializer='zeros',
                                    name='bias')
        self.xk_shape = (-1, feat_dims // self.K, self.K)

    def call(self, x):
        # 1. element-wise product between x and kernel
        xk = x * self.kernel
        # 2. reshape xk, xk.shape = (batch_size, input_feat_dim//3, 3 )
        xk = K.reshape(xk, self.xk_shape)
        # 3. compute y for every K elements in xk
        y = K.sum(xk, axis=-1, keepdims=False) + self.bias
        return y

    def compute_output_shape(self, input_shape):
        batch_size, feat_dims = input_shape
        return batch_size, feat_dims // self.K


x = Input(shape=(3000,))
y = DenseK(K=3, name='3-neuron-dense')(x)
model = Model(inputs=x, outputs=y)
print(model.summary())

a = np.random.randn(2, 3000)
b = np.random.randn(2, 1000)

model.compile('sgd', loss='mse')
model.fit(a, b)
