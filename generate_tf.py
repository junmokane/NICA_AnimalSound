'''
Mapping from STL-10 to Custom environement
0 = air_conditioner
1 = car_horn
2 = children_playing
3 = dog_bark
4 = drilling
5 = engine_idling
6 = gun_shot
7 = jackhammer
8 = siren
9 = street_music
[1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000] number of instances
'''


from absl import app
from absl import flags
from absl import logging
import os
import pandas as pd
from tqdm import tqdm
import librosa
from sklearn import preprocessing

import tensorflow as tf
import numpy as np

# info about Custom Environment
ENV_SPECS = dict(none=[0],   # air_conditional
                 prey=[5],   # engine_idling
                 rprey=[2],  # children_playing
                 pred=[3])   # dog_bark


FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', './data/UrbanSound8K/audio/fold', 'Directory for data.')
flags.DEFINE_string('save_dir', './data/UrbanSound8K', 'Directory for saving TFRecord.')
flags.DEFINE_integer('num_none', 1, 'Number of nones.')
flags.DEFINE_integer('num_prey', 1, 'Number of preys.')
flags.DEFINE_integer('num_rprey', 1, 'NUmber of rotten preys.')
flags.DEFINE_integer('num_pred', 1, 'Number of predators.')


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def generate_tfrecord(data_dir, save_dir, num_specs):
    # labels to use for Custom Environment
    none_label = ENV_SPECS['none'][:FLAGS.num_none]
    prey_label = ENV_SPECS['prey'][:FLAGS.num_prey]
    rprey_label = ENV_SPECS['rprey'][:FLAGS.num_rprey]
    pred_label = ENV_SPECS['pred'][:FLAGS.num_pred]
    all_label = none_label + pred_label + rprey_label + prey_label

    # list of each category
    none_list = []  # list of tuples with (image, label)
    prey_list = []  # same
    rprey_list = []
    pred_list = []  # same

    # read UrbanSound8K data
    sounds = []
    labels = []
    data = pd.read_csv("./data/UrbanSound8K/metadata/UrbanSound8K.csv")

    for i in range(len(data)):
        fold_no = str(data.iloc[i]["fold"])
        file = data.iloc[i]["slice_file_name"]
        label = data.iloc[i]["classID"]
        filename = data_dir + fold_no + "/" + file
        if label in all_label:
            y, sr = librosa.load(filename)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
            if S.shape[1] == 173:
                sounds.append(S)
                labels.append(label)

    sounds = np.array(sounds, dtype=np.float64)
    b, y, x = sounds.shape
    assert np.all(sounds == np.reshape(np.reshape(sounds, (b, -1)), (b, y, x)))
    sounds = np.reshape(preprocessing.scale(np.reshape(sounds, (b, -1))), (b, y, x))
    #print(np.max(sounds), np.min(sounds),
    #      np.mean(sounds, axis=0), np.var(sounds, axis=0))

    for i in range(len(sounds)):
        img = sounds[i]
        lab = labels[i]
        if lab in none_label:
            none_list.append((img, 0))
        elif lab in pred_label:
            pred_list.append((img, 1))
        elif lab in prey_label:
            prey_list.append((img, 2))
        elif lab in rprey_label:
            rprey_list.append((img, 3))
        else:
            pass

    print('Length of each list : (none, pred, prey, rprey)',
          len(none_list), len(pred_list), len(prey_list), len(rprey_list))

    # save the result in TFRecord format
    record_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    map_dict = dict(prey=prey_list, rprey=rprey_list, pred=pred_list, none=none_list)

    for key in map_dict.keys():
        logging.info('Processing %s' % key)
        save_path = os.path.join(save_dir, key)
        with tf.python_io.TFRecordWriter(save_path,
                                         options=record_options) as writer:
            for image, label in map_dict[key]:
                binary_image = image.tostring()
                string_set = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(binary_image),
                    'label': _int64_feature(label)
                }))
                writer.write(string_set.SerializeToString())
        writer.close()


def main(argv=()):
    del argv  # unused
    num_specs = [FLAGS.num_prey, FLAGS.num_pred, FLAGS.num_none]
    generate_tfrecord(data_dir=FLAGS.data_dir,
                      save_dir=FLAGS.save_dir,
                      num_specs=num_specs)


if __name__ == "__main__":
    app.run(main)