import tensorflow_addons as tfa
import tensorflow as tf
from dialogue_bot.model.layers import EmbeddingLayer


class Decoder(tf.keras.layers.Layer):

    def __init__(self, params):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(
            params['vocab_size'],
            params['embedding_size'],
        )
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.output_projector = tf.keras.layers.Dense(params['output_vocab_size'])
        self.decoder_rnncell = tf.keras.layers.GRUCell(params['hidden_size'])
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.decoder_rnncell,
            self.sampler,
            self.output_projector,
        )
