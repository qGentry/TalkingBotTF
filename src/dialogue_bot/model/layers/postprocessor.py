import tensorflow as tf


class Postprocessor(tf.keras.layers.Layer):

    def __init__(self, vocab_path):
        super().__init__()
        self.indices_table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename=vocab_path,
                key_dtype=tf.int64,
                key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                value_dtype=tf.string,
                value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                name="indices_to_outputs_table_initializer"
            ), "P",
            name="indices_to_outputs_table"
        )

    def process(self, predicted_ids):
        chars_decoded = self.indices_table.lookup(tf.cast(predicted_ids, dtype=tf.int64))
        sentence_decoded = tf.strings.reduce_join(
            chars_decoded,
            separator='',
            axis=1
        )
        return sentence_decoded
