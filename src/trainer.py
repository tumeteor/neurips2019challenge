import datetime
import os
import random
import sys
import tensorflow as tf
import numpy as np

from src.data_utils.loader import list_filenames, load_data, ORIGINAL_SHAPE

from src.models.ed import Model
from src.utils import reshape_patch, gpu_split

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_length', 12,
                     'encoder hidden states.')
flags.DEFINE_integer('total_length', 15,
                     'total input and output length.')
flags.DEFINE_integer('img_width', ORIGINAL_SHAPE[1],
                     'input image width.')
flags.DEFINE_integer('img_height', ORIGINAL_SHAPE[0],
                     'input image height.')
flags.DEFINE_integer('img_channel', 3,
                     'number of image channel.')


class Generator:
    def __call__(self, file, K, T):
        data = load_data(file, K=K,
                         T=T)
        yield data


class Generator2:
    def __call__(self, data):
        for seq in data:
            seq = np.expand_dims(seq, axis=0)
            ims_reverse = seq[:, :, :, ::-1]
            ims_reverse = reshape_patch(ims_reverse, 1)
            ims_reverse = ims_reverse.squeeze()
            ims_r = reshape_patch(seq, 1)
            ims_r = ims_r.squeeze()

            yield ims_r, ims_reverse


def get_data(filenames, path):
    if "validation" in path:
        filenames = random.sample(filenames, 2)
        num_parallel_calls = 2
    else:
        num_parallel_calls = 3
    cycle_length = len(filenames)
    block_length = 1
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
        Generator(),
        tf.float32,
        (tf.TensorShape([272, FLAGS.total_length, ORIGINAL_SHAPE[0], ORIGINAL_SHAPE[1], 3])),
        args=(path + filename, FLAGS.input_length, FLAGS.total_length - FLAGS.input_length)),
                       cycle_length, block_length, num_parallel_calls=1)

    ds = ds.interleave(lambda x: tf.data.Dataset.from_generator(
        Generator2(),
        (tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([FLAGS.total_length, ORIGINAL_SHAPE[0], ORIGINAL_SHAPE[1], 3]),
                       tf.TensorShape([FLAGS.total_length - FLAGS.input_length - 1, ORIGINAL_SHAPE[0],
                                       ORIGINAL_SHAPE[1], 3]),
                       tf.TensorShape([FLAGS.total_length, ORIGINAL_SHAPE[0], ORIGINAL_SHAPE[1], 3])),
        args=(x,)),
                       cycle_length, block_length, num_parallel_calls=num_parallel_calls)
    return ds


def train_wrapper():
    # load data

    file_names = list_filenames(FLAGS.train_data_paths)
    # shuffle it
    random.shuffle(file_names)

    valid_file_names = list_filenames(FLAGS.valid_data_paths)
    train_dataset = get_data(file_names, path=FLAGS.train_data_paths)
    train_dataset = train_dataset.batch(FLAGS.batch_size * FLAGS.n_gpu, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset.repeat(FLAGS.max_iterations)

    valid_dataset = get_data(valid_file_names, path=FLAGS.valid_data_paths)
    # valid_dataset = valid_dataset.batch(FLAGS.batch_size, drop_remainder=True)

    dataset_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_init_op = dataset_iter.make_initializer(train_dataset)
    valid_init_op = dataset_iter.make_initializer(valid_dataset)

    model = Model(FLAGS, train_init_op, dataset_iter)

    if FLAGS.save_dir and len(os.listdir(FLAGS.save_dir)):
        print("load model from checkpoints..")
        model.load(FLAGS.save_dir)
    counter = 0
    next_element = dataset_iter.get_next()
    try:
        for itr in range(FLAGS.max_iterations):
            ims_r, ims_reverse = model.sess.run(next_element)
            train(model, ims_r, FLAGS, counter, ims_reverse)
            if counter % FLAGS.snapshot_interval == 0:
                model.save(itr)
                # random day for validation
                # model.sess.run(valid_init_op)
                # test(model, dataset_iter, configs=FLAGS, save_name="result")
            if counter % FLAGS.test_interval == 0:
                pass
            counter += 1
    except tf.errors.OutOfRangeError:
        pass


def train(model, ims, real_input_flag, configs, itr, ims_reverse=None):
    ims = ims[:, :configs.total_length]

    ims_list = gpu_split(ims, configs.n_gpu, configs.batch_size)

    cost = model.train(ims_list, configs.lr, real_input_flag)

    flag = 1
    ims_rev = gpu_split(ims_reverse[:, ::-1], configs.n_gpu, configs.batch_size)
    cost += model.train(ims_rev, configs.lr, real_input_flag)
    flag += 1
    ims_rev = gpu_split(ims_reverse[:, ::-1], configs.n_gpu, configs.batch_size)
    cost += model.train(ims_rev, configs.lr, real_input_flag)
    flag += 1

    cost = cost / flag

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print(f'training loss: {cost}')


def main():
    train_wrapper()


if __name__ == '__main__':
    tf.compat.v1.app.run(argv=sys.argv)
