import glob
import re

import tensorflow as tf
import numpy as np
from scipy.io import wavfile

from pydub import AudioSegment

# from vggvox import constants as config
# import vggvox.model
# from vggvox.feature_extraction import FastFourierTransform
#
# model = vggvox.model.vggvox_model()
# model.load_weights(config.WEIGHTS_FILE)
# model.summary()

TRAIN_DIR = "./data/VoxCeleb2_simple10/aac/"


class SpeakerTemplate(object):
    templates = None  # np.zeros(1024, 3)


def load_data(filename: str):
    # (fs, signal) = wavfile.read(filename)
    signal = AudioSegment.from_file(filename).get_array_of_samples()

    id = re.search("id(\d+)", filename).group(1)
    # yield signal
    return (signal, id)


def simeple_dataset():
    DATA_GLOB = TRAIN_DIR + '*/*/*.m4a'
    print(DATA_GLOB)
    for file in glob.glob(DATA_GLOB):
        print(file)

    list_ds = tf.data.Dataset.list_files(DATA_GLOB)

    for f in list_ds.take(5):
        filename = f.numpy().decode('UTF-8')
        (signal, id) = load_data(filename)
        print(signal[1:5])
        print(id, signal[1:5], type(signal[1:5]))


def enroll_3_reduce_mean(speaker_templates: SpeakerTemplate):
    '''
    Compute a score based on 3 enroll sample and use mean-reduced cosine distance
    score = 1/n *sum{ emb(i); i = 1..n } ; n = 3
    '''

    pass


def test_verification():
    print("Processing enroll samples....")
    # buckets = FastFourierTransform.build_buckets()
    # signal = FastFourierTransform.load_wav('data/wav/enroll/19-227-0000.wav')
    (fs, signal) = wav.read('data/wav/enroll/19-227-0000.wav')
    print(signal.shape, signal[1:100])
    feature = FastFourierTransform.get_fft_spectrum(signal)
    print(feature[1:100])
    print(feature.shape)

    result = model.predict(feature.reshape(1, *feature.shape, 1))
    embeded = np.squeeze(result)
    print(embeded.shape)


if __name__ == '__main__':
    # test_verification()
    simeple_dataset()
