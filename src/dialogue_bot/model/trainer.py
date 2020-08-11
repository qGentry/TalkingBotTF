import tensorflow as tf
import tensorflow_addons as tfa
from dialogue_bot.model import DialogueModel


def calc_loss(
        normalization_logits: tf.Tensor,
        expected_normalization_labels: tf.Tensor,
):
    non_pad_mask = tf.not_equal(
        x=expected_normalization_labels,
        y=0,
    )
    weights = tf.cast(non_pad_mask, tf.float32)
    return tfa.seq2seq.sequence_loss(
        logits=normalization_logits,
        targets=expected_normalization_labels,
        weights=weights
    )


class Trainer:

    def __init__(self, model, clip_norm):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_steps_count = tf.Variable(0)
        self.train_step = self.get_train_step(model, self.optimizer, clip_norm)
        checkpoint = tf.train.Checkpoint(
            train_steps_count=self.train_steps_count,
            optimizer=self.optimizer,
            model=self.model
        )
        self.model_dir = "model"
        latest_checkpoint = tf.train.latest_checkpoint(
            self.model_dir
        )
        if latest_checkpoint is not None:
            checkpoint.restore(latest_checkpoint)

        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=self.model_dir,
            max_to_keep=3
        )

    def train(self, num_epochs, dataset):
        for epoch_num in range(num_epochs):
            losses = []
            for i, batch in enumerate(dataset):
                loss = self.train_step(batch)
                self.train_steps_count.assign_add(1)
                losses.append(loss)
                if i % 10 == 0:
                    tf.print("step ", i, " loss is ", loss)
                if i % 100 == 0:
                    self.checkpoint_manager.save()
            tf.print("Epoch ", epoch_num, " is finished, average loss is ", tf.reduce_mean(losses))

    def get_train_step(self,
                       model: DialogueModel,
                       optimizer: tf.keras.optimizers.Optimizer,
                       clip_norm: int = 2,
                       ):
        input_signature = [
            {
                'context': tf.TensorSpec(shape=(None, None), dtype=tf.string),
                'target': tf.TensorSpec(shape=[None], dtype=tf.string),
            }
        ]

        @tf.function(input_signature=input_signature)
        def train_step(batch):
            with tf.GradientTape() as tape:
                logits, expected_output = model.call(batch, mode='train')
                loss = calc_loss(logits.rnn_output, expected_output)
            gradients = tape.gradient(
                target=loss,
                sources=model.trainable_variables
            )
            clipped_gradients = [(tf.clip_by_norm(grad, clip_norm))
                                 for grad in gradients]
            optimizer.apply_gradients(zip(clipped_gradients,
                                          model.trainable_variables))
            return loss

        return train_step
