import tensorflow as tf

def create_filename_queue(file_list, shuffle=False):
    '''
    Creates a queue for reading data from file.

    :param file_list: the list of filenames
    :param shuffle: whether to shuffle the file list.
    :return: tf queue with filenames
    '''
    return tf.train.string_input_producer(file_list, shuffle=shuffle)


def load_data_from_filename_queue(filename_queue):
    '''
    Loads (image, label) tuple from filename queue.

    :param filename_queue: tf queue created from create_filename_queue()
    :return: tuple of image (shape [28, 28]), label (shape [1,]) tensors.
    '''
    record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    keys, protos = tf.TFRecordReader(options=record_options).read(filename_queue)
    examples = tf.parse_single_example(protos, features=dict(
        image=tf.FixedLenFeature(shape=[], dtype=tf.string),
        label=tf.FixedLenFeature(shape=[], dtype=tf.int64),
    ))
    image = examples['image']
    label = examples['label']
    image = tf.decode_raw(image, tf.float64)
    image = tf.reshape(image, (40, 173, 1))

    return image, label


def load_data(file_list, shuffle=True, scope='load_data'):
    with tf.name_scope(scope):
        filename_queue = create_filename_queue(file_list, shuffle=False)
        return load_data_from_filename_queue(filename_queue)
