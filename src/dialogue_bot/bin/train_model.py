from dialogue_bot.model.model_wrapper import ModelWrapper
import tensorflow as tf

config = {
    'model_params': {
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
    },
    'wrapper_params': {
        'train': True,
        'batch_size': 100,
        'clip_norm': 2,
        'data_path': 'data/TlkPersonaChatRus/dialogues.tsv',
    }
}

model = ModelWrapper(config)
model.train_model(1)
