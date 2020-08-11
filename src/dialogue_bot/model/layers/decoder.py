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
        self.beam_width = 3
        self.max_iter = 250

    def embedding_fn(self, inputs):
        embeddings, _ = self.embedding_layer(inputs)
        return embeddings

    def inference_decoder(self, initial_state):
        inference_decoder = tfa.seq2seq.BeamSearchDecoder(
            embedding_fn=self.embedding_fn,
            cell=self.decoder_rnncell,
            maximum_iterations=self.max_iter,
            output_layer=self.output_projector,
            beam_width=self.beam_width,
        )
        batch_size = tf.shape(initial_state)[0]
        start_tokens = tf.fill([batch_size], 2)
        decoder_outputs, _, _ = inference_decoder(
            None,
            start_tokens=start_tokens,
            end_token=1,
            training=False,
            initial_state=tfa.seq2seq.tile_batch(initial_state, self.beam_width)
        )
        return decoder_outputs
