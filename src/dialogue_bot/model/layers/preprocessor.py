import tensorflow as tf


class Preprocessor(tf.Module):

    def __init__(self, vocab_path: str = '../vocabs/vocab.txt'):
        super().__init__()
        self.indices_table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename=vocab_path,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                name="inputs_to_indices_table_initializer"
            ), -1,
            name="inputs_to_indices_table"
        )

    def process(self, batch):
        strings = "S" + batch + "T"
        indices_ragged = tf.strings.unicode_split(strings, "UTF-8")

        indices_ragged = tf.ragged.map_flat_values(
            self.indices_table.lookup,
            indices_ragged
        )
        indices_tensor_dense = indices_ragged.to_tensor(self.indices_table.lookup(tf.constant('P')))
        return indices_tensor_dense
