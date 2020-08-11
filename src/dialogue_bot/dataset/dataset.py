import tensorflow as tf
from dialogue_bot.dataset.data_preprocessor import DataPreprocessor


def get_dataset(
        data_path: str,
        context_window_size: int,
        batch_size: int):

    if context_window_size < 1:
        raise ValueError('Context window size cannot be less than 1')

    preprocessor = DataPreprocessor(
        data_path=data_path,
        context_window_size=context_window_size
    )
    data_list = preprocessor.get_data_list()
    output_types = {
        'context': tf.string,
        'target': tf.string,
    }
    output_shapes = {
        'context': [context_window_size],
        'target': [],
    }
    dataset = tf.data.Dataset.from_generator(
        lambda: data_list,
        output_types=output_types,
        output_shapes=output_shapes
    )
    return dataset.batch(batch_size=batch_size)
