import re
from abc import ABC

import tensorflow as tf
from pydub import AudioSegment

from feature.feature_extraction import FastFourierTransform


class VoxCeleb2Dataset(tf.data.Dataset, ABC):

    def __new__(cls, DATA_GLOB, batch: int = 16, repeat: int = 1):
        return tf.data.Dataset.from_generator(
            cls._m4a_generator,
            output_types=(tf.dtypes.int64, tf.dtypes.int16),
            output_shapes=(tf.TensorShape([1, None, None, 1]), tf.TensorShape([])),
            args=(DATA_GLOB, batch,),

        ).repeat(repeat).as_numpy_iterator()

    @classmethod
    def _m4a_generator(cls, DATA_GLOB, batch: int = 1):
        filelist = tf.data.Dataset.list_files(DATA_GLOB, shuffle=False)

        for encoded_filename in filelist.repeat().take(batch).as_numpy_iterator():
            filename = encoded_filename.decode('UTF-8')

            signal = AudioSegment.from_file(filename) \
                .set_channels(1) \
                .get_array_of_samples()
            feature = FastFourierTransform.get_fft_spectrum(signal)

            id = int(re.search(r"id(\d+)", filename).group(1))

            yield feature.reshape(1, *feature.shape, 1), id

    @classmethod
    def meta_generator(cls, DATA_GLOB, batch: int = 1):
        filelist = tf.data.Dataset.list_files(DATA_GLOB, shuffle=False).as_numpy_iterator()
        meta_list = []
        for encoded_filename in filelist:
            filename = encoded_filename.decode('UTF-8')

            id = re.search(r"\/(id\d+)\/", filename).group(1)
            base_filename = re.search(r"id\d+\/(\w+.\w+.\w+)", filename).group(1)

            meta_list += [(id, base_filename)]

        return iter(meta_list)