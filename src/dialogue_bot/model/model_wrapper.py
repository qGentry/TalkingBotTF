import tensorflow as tf
from dialogue_bot.model.dialogue_model import DialogueModel
import numpy as np
from dialogue_bot.model.trainer import Trainer
from dialogue_bot.dataset.dataset import get_dataset


class ModelWrapper:

    def __init__(self, params):
        self.model = DialogueModel(params['model_params'])
        self.byte_decoder = np.vectorize(lambda x: x.decode())
        self.wrapper_params = params['wrapper_params']
        if self.wrapper_params['train']:
            self.dataset = get_dataset(
                data_path=self.wrapper_params['data_path'],
                context_window_size=params['model_params']['context_size'],
                batch_size=self.wrapper_params['batch_size'],
            )
            self.trainer = Trainer(self.model,
                                   self.wrapper_params["batch_size"],
                                   self.wrapper_params['clip_norm']
                                   )

    def __call__(self, context):
        context = tf.constant(context, dtype=tf.string)
        model_ans = self.model.answer(context)
        return self.byte_decoder(model_ans)

    def train_model(self, num_epochs):
        self.trainer.train(num_epochs, self.dataset)

    def restore_model(self, model_dir='model'):
        latest_checkpoint = tf.train.latest_checkpoint(
            model_dir
        )
        if latest_checkpoint is not None:
            checkpoint = tf.train.Checkpoint(
                model=self.model
            )
            checkpoint.restore(latest_checkpoint)
            print("Successefully restored from last checkpoint")

