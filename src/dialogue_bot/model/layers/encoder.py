import tensorflow as tf
from typing import Tuple, List


class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            mask_zero=True
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        embeddings = self.embeddings(inputs)
        mask = self.embeddings.compute_mask(inputs)
        return embeddings, mask


class BasicEncoder(tf.keras.layers.Layer):

    def __init__(self, hidden_size):
        super().__init__()
        self.rnn_layer = tf.keras.layers.GRU(
            units=hidden_size,
            return_state=True,
            time_major=False,
        )
        self.rnn_layer = tf.keras.layers.Bidirectional(
            layer=self.rnn_layer
        )

    def call(self,
             inputs: tf.Tensor,
             mask: tf.Tensor = None,
             initial_state: tf.Tensor = None,
             **kwargs
             ) -> [List[tf.Tensor]]:
        x = self.rnn_layer(inputs, mask=mask, initial_state=initial_state)
        last_state = x[1:]
        return last_state


class ContextEncoder(tf.keras.layers.Layer):

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.encoders = [
            BasicEncoder(params['hidden_size']) for _ in range(params['context_size'])
        ]

    def call(self,
             inputs: tf.Tensor,
             mask: tf.Tensor = None,
             **kwargs):
        last_state = None
        for i, encoder in enumerate(self.encoders):
            last_state = encoder(inputs[:, i], mask[:, i], initial_state=last_state)
        return last_state
