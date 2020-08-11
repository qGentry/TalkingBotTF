import tensorflow as tf
from dialogue_bot.model import layers


class Model(tf.Module):

    def __init__(self, params):
        super().__init__()
        self.preprocessor = layers.Preprocessor(params['vocab_path'])
        embedding_params = params['embedding_params']
        self.context_size = params['context_size']
        self.embeddings_layer = layers.EmbeddingLayer(
            embedding_params['vocab_size'],
            embedding_params['embedding_size'],
        )
        self.context_encoder = layers.ContextEncoder(params['encoder_params'])
        self.decoder_projector = tf.keras.layers.Dense(params['decoder_params']['hidden_size'])
        self.decoder = layers.Decoder(params['decoder_params'])

    def call(self, inputs, mode):
        context = inputs['context']

        context_indices = self.preprocessor.process(context)
        context_embeddings, context_mask = self.embeddings_layer(context_indices)
        context_encoded = self.context_encoder(context_embeddings, context_mask)
        context_encoded = tf.concat(context_encoded, axis=-1)
        context_projected = self.decoder_projector(context_encoded)
        if mode == 'train':
            target_indices = self.preprocessor.process(inputs['target'])
            decoder_input = target_indices[:, :-1]
            expected_decoder_output = target_indices[:, 1:]
            decoder_emb_inp, decoder_emb_inp_mask = self.decoder.embedding_layer(decoder_input)
            outputs, _, _ = self.decoder.decoder(decoder_emb_inp,
                                                 initial_state=context_projected,
                                                 )
            return outputs, expected_decoder_output


