from einops.layers.tensorflow import Rearrange
import tensorflow as tf
from tensorflow.keras import layers

from net_c.mlp import MLP


class MLPMixer_1d(tf.keras.Model):
    """Original MLP-Mixer, with same API as paper.
    But remove rearrange parts,
    and 2d patch cut changed to 1d sample cut """

    def __init__(
        self,
        num_classes,
        num_blocks,
        sample_size,
        hidden_dim,
        tokens_mlp_dim,
        channels_mlp_dim,
    ):
        super().__init__()
        # s = (sample_size, sample_size)
        self.make_tokens = layers.Conv1D(hidden_dim, sample_size, sample_size)
        # self.rearrange = Rearrange("n h w c -> n (h w) c")
        self.mixers = [
            NdMixerBlock([tokens_mlp_dim, channels_mlp_dim]) for _ in range(num_blocks)
        ]
        self.batchnorm = layers.BatchNormalization()
        self.clf = layers.Dense(num_classes, kernel_initializer="zeros")

    def call(self, inputs):
        x = self.make_tokens(inputs)
        # x = self.rearrange(x)
        for mixer in self.mixers:
            x = mixer(x)
        x = self.batchnorm(x)
        x = tf.reduce_mean(x, axis=1)
        return self.clf(x)


class NdMixerBlock(layers.Layer):
    "N-dimensional MLP-mixer block, same as paper when 2-dimensional."

    def __init__(self, mlp_dims: list = None, activation=tf.nn.gelu):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.activation = activation

    def build(self, input_shape):
        ndim = len(input_shape) - 1
        mlp_dims = self.mlp_dims if self.mlp_dims else [None] * ndim

        self.mlps = [
            MLP(input_shape[i + 1], mlp_dims[i], axis=i + 1, activation=self.activation)
            for i in range(ndim)
        ]

        self.batchnorms = [layers.BatchNormalization() for _ in range(ndim)]

    def call(self, inputs):
        h = inputs
        for mlp, batchnorm in zip(self.mlps, self.batchnorms):
            h = h + mlp(batchnorm(h))
        return h
