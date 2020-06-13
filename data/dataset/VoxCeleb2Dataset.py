import re
from abc import ABC

from pydub import AudioSegment
import tensorflow as tf

from feature.feature_extraction import FastFourierTransform


class VoxCeleb2Dataset(tf.data.Dataset, ABC):

    def __new__(cls, DATA_GLOB, batch: int = 16):
        cls.filelist = tf.data.Dataset.list_files(DATA_GLOB)  # .as_numpy_iterator()

        return tf.data.Dataset.from_generator(
            cls._m4a_generator,
            output_types=(tf.dtypes.int64, tf.dtypes.int16),
            output_shapes=(tf.TensorShape([1, None, None, 1]), tf.TensorShape([])),
            args=(batch,),

        ).as_numpy_iterator()

    @classmethod
    def _m4a_generator(cls, batch: int = 1):
        for encoded_filename in cls.filelist.take(batch).as_numpy_iterator():
            filename = encoded_filename.decode('UTF-8')

            signal = AudioSegment.from_file(filename) \
                .set_channels(1) \
                .get_array_of_samples()
            feature = FastFourierTransform.get_fft_spectrum(signal)

            id = int(re.search(r"id(\d+)", filename).group(1))

            yield feature.reshape(1, *feature.shape, 1), id