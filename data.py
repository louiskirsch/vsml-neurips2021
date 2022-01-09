import functools
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import lock
from config import configurable, DotDict
from typing import Iterator

AUTOTUNE = tf.data.experimental.AUTOTUNE


@configurable('random_dataset')
def random_dataset(spec, transform, size, target_count, img_size):
    ds = tf.data.Dataset.from_tensor_slices((
        tf.random.normal([size] + img_size),
        tf.random.uniform([size], maxval=target_count, dtype=tf.int32),
    ))
    return transform(ds, target_count)


@configurable('sum_dataset')
def sum_dataset(spec, transform, input_size):
    ds = tf.data.Dataset.from_tensors(0)
    ds = ds.map(lambda _: tf.random.truncated_normal(input_size, mean=tf.random.uniform([], minval=-1., maxval=1.)))
    ds = ds.map(lambda x: (x, tf.cast(tf.reduce_sum(x) >= 0, tf.int64)))
    return transform(ds, num_classes=2)


@configurable('omniglot_fewshot_dataset')
def omniglot_fewshot(spec: DotDict, transform, num_classes: int, resize, rotate: bool, test_last: bool):
    assert test_last
    assert (spec.episode_length - 1) % num_classes == 0
    num_examples = int(np.ceil(spec.episode_length / num_classes))

    builder = tfds.builder('omniglot')
    locked_download('omniglot', builder)
    ds = builder.as_dataset(split=spec.split, as_supervised=True)

    def resize_example(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.image.resize(x, resize)
        return x, y

    def rotate_example(x, y):
        x = tf.image.convert_image_dtype(x, tf.float32)
        x = tf.image.rot90(x, k=tf.random.uniform(shape=[], minval=0,
                                                  maxval=4, dtype=tf.int32))
        return x, y

    # Remove useless color channel
    ds = ds.map(lambda x, y: (x[:, :, :1], y))

    if resize:
        ds = ds.map(resize_example)

    if rotate:
        ds = ds.map(rotate_example)

    ds = ds.shuffle(spec.data_config.shuffle_buffer_size).repeat()
    ds = ds.apply(tf.data.experimental.group_by_window(
        key_func=lambda x, y: y,
        reduce_func=lambda _, window: window,
        window_size=num_examples * spec.batch_size)
    )

    def shuffle_batch(x, y):
        limit = tf.shape(x)[0]
        indices = tf.range(start=0, limit=limit, dtype=tf.int32)
        indices = tf.boolean_mask(indices, indices % num_examples > 0)
        indices = tf.random.shuffle(indices)
        test_idx = tf.random.uniform(shape=[], minval=0, maxval=num_classes, dtype=tf.int32) * num_examples
        indices = tf.concat([indices, [test_idx]], axis=0)
        return tf.gather(x, indices), tf.gather(y, indices)

    def relabel(x, y):
        _, labels = tf.unique(y)
        return x, labels

    ds = ds.batch(spec.batch_size * num_classes * num_examples)
    ds = ds.map(shuffle_batch).map(relabel).unbatch()
    return transform(ds, num_classes, shuffle_repeat=False)


CUSTOM_DATASETS = {
    'random': random_dataset,
    'omniglot_fewshot': omniglot_fewshot,
    'sum': sum_dataset,
}


def pad_to_shape(t, target_shape, **kwargs):
    source_shape = t.shape
    paddings = [[0, t - s] for s, t in zip(source_shape, target_shape)]
    return tf.pad(t, paddings, **kwargs)


def locked_download(name, builder):
    if not tf.io.gfile.exists(builder.data_dir):
        with lock.file_lock(f'{name}.lock'):
            builder.download_and_prepare()


class DataLoader:

    def __init__(self, mpi_rank, mpi_size, training_config, eval_config, data_config):
        self.training_config = training_config
        self.eval_config = eval_config
        self.data_config = data_config
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self._permutation = None
        self._projection = None

    @configurable('data.preprocess')
    def _preprocess(self, name, images, labels, num_classes, dataset_stats,
                    normalization_mode, pad, resize, rand_proj: bool,
                    shuffle: bool):
        images = tf.image.convert_image_dtype(images, tf.float32)
        labels = tf.one_hot(labels, num_classes)

        if resize is not None:
            images = tf.image.resize(images, resize)

        dataset_stats = dataset_stats[name]
        if normalization_mode == 'dataset_standard':
            images = (images - dataset_stats['mean']) / dataset_stats['std']
        elif normalization_mode == 'standard':
            images = tf.image.per_image_standardization(images)
        elif normalization_mode == 'identity':
            pass
        else:
            raise ValueError('Normalization mode must be dataset_standard, standard, '
                             'or identity')

        if pad['input'] is not None:
            images = pad_to_shape(images, pad['input'])
        if pad['output'] is not None:
            labels = pad_to_shape(labels, pad['output'])

        if shuffle and not rand_proj:
            shape = images.shape
            if self._permutation is None:
                self._permutation = np.random.permutation(np.arange(np.prod(shape)))
            images = tf.gather(tf.reshape(images, (-1,)), self._permutation)
            images = tf.reshape(images, shape)

        if rand_proj:
            shape = images.shape
            w_shape = [np.prod(images.shape)] * 2
            if self._projection is None:
                limit = np.sqrt(6 / np.sum(w_shape))
                values = np.random.uniform(-limit, limit, size=w_shape)
                self._projection = values
            images = tf.reshape(images, (1, w_shape[0])) @ self._projection
            images = tf.reshape(images, shape)

        return images, labels

    def build_dataset(self, name: str, split: str, batch_size: int, episode_length: int,
                      population_size: int = 0) -> Iterator:
        def transform(ds: tf.data.Dataset, num_classes: int, batch=True, shuffle_repeat=True):

            if self.data_config.filter_classes is not None:
                filter_classes = self.data_config.filter_classes
                ds = ds.filter(lambda x, y: tf.reduce_any(tf.equal(y, filter_classes)))
                ds = ds.map(lambda x, y: (x, tf.where(tf.equal(y, filter_classes))[0, 0]))
                num_classes = len(filter_classes)

            ds = ds.map(functools.partial(self._preprocess, name,
                                          num_classes=num_classes),
                        num_parallel_calls=AUTOTUNE)

            if shuffle_repeat:
                ds = ds.shuffle(buffer_size=self.data_config.shuffle_buffer_size)
                ds = ds.repeat()

            if batch:
                ds = ds.batch(batch_size, drop_remainder=True)
                ds = ds.batch(episode_length, drop_remainder=True)
                if population_size:
                    ds = ds.batch(population_size // self.mpi_size, drop_remainder=True)

            return ds

        spec = DotDict(split=split,
                       data_config=self.data_config,
                       batch_size=batch_size,
                       episode_length=episode_length)

        # TODO Set tensorflow seed to control randomness
        if name in CUSTOM_DATASETS:
            ds = CUSTOM_DATASETS[name](spec, transform)
        else:
            builder = tfds.builder(name)
            info = builder.info
            locked_download(name, builder)
            ds = builder.as_dataset(split=split, as_supervised=True)
            num_classes = info.features['label'].num_classes
            ds = transform(ds, num_classes)

        return ds

    def build_train_dataset(self):
        ds = self.build_dataset(self.training_config.dataset,
                                'train',
                                self.training_config.batch_size,
                                self.training_config.episode_length,
                                self.training_config.population_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds.as_numpy_iterator()

    def build_eval_dataset(self, dataset_name):
        ds = self.build_dataset(dataset_name,
                                self.eval_config.subset,
                                self.eval_config.batch_size,
                                self.eval_config.episode_length)
        ds = ds.batch(self.eval_config.count)
        ds = ds.prefetch(AUTOTUNE)
        return ds.as_numpy_iterator()
