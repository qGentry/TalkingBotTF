from dialogue_bot.model import Model
from dialogue_bot.model.layers import Decoder
import tensorflow as tf
import pickle

config = {
    'vocab_path': 'vocabs/vocab.txt',
    'context_size': 2,
    'embedding_params': {
        'vocab_size': 87,
        'embedding_size': 50,
    },
    'encoder_params': {
        'hidden_size': 100,
        'context_size': 2
    },
    'decoder_params': {
        'embedding_size': 64,
        'vocab_size': 87,
        'hidden_size': 77,
        'output_vocab_size': 87,
    }
}

tf.config.experimental_run_functions_eagerly(True)
model = Model(params=config)



