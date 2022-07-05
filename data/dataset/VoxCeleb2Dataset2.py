import re
from abc import ABC

import tensorflow as tf
from pandas import DataFrame
from pydub import AudioSegment

from feature.feature_extraction import FastFourierTransform


class VoxCeleb2Dataset2(tf.data.Dataset, ABC):

    def __new__(cls, filelist: DataFrame, batch: int = 16):
        cls.filelist = filelist

        return tf.data.Dataset.from_generator(
            cls._m4a_generator,
            output_types=(tf.dtypes.int64, tf.dtypes.int16),
            output_shapes=(tf.TensorShape([1, None, None, 1]), tf.TensorShape([])),
            args=(batch,),

        ).as_numpy_iterator()

    @classmethod
    def _m4a_generator(cls, batch: int = 1):
        for row in cls.filelist.itertuples():
            filename = row.filename
            print(filename)

            signal = AudioSegment.from_file(filename) \
                .set_channels(1) \
                .get_array_of_samples()
            feature = FastFourierTransform.get_fft_spectrum(signal)

            id = int(re.search(r"id(\d+)", filename).group(1))

            yield feature.reshape(1, *feature.shape, 1), id

