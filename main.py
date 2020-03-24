import collections
import glob
import re
import time

import tensorflow as tf
import numpy as np
from pandas import DataFrame
from scipy.io import wavfile

from pydub import AudioSegment
from pydub.playback import play

from vggvox import constants as config
import vggvox.model
from vggvox.dataset import VoxCeleb2Dataset
from vggvox.feature_extraction import FastFourierTransform

model = vggvox.model.vggvox_model()
model.load_weights(config.WEIGHTS_FILE)
# model.summary()

TRAIN_DIR = "./data/VoxCeleb2_simple10/aac/"
DATA_GLOB = TRAIN_DIR + '*/*/*.m4a'


# DATA_GLOB = TRAIN_DIR + "/id" + "00081/vnEsqFqzC8k/00200.m4a"  # TRAIN_DIR + '*/*/*.m4a'


class SpeakerTemplate(object):
    templates = None  # np.zeros(1024, 3)


def load_m4a(filename: str):
    signal = AudioSegment.from_file(filename) \
        .set_channels(1) \
        .get_array_of_samples()

    id = re.search("id(\d+\/\S+)", filename).group(1)
    # yield signal
    return signal, id


def read_dataset(DATA_GLOB: str):
    print(DATA_GLOB)
    dataset = tf.data.Dataset.list_files(DATA_GLOB)

    for f in dataset.take(1):
        filename = f.numpy().decode('UTF-8')
        (signal, id) = load_m4a(filename)

    return signal, id


def enroll_3_reduce_mean(speaker_templates: SpeakerTemplate):
    '''
    Compute a score based on 3 enroll sample and use mean-reduced cosine distance
    score = 1/n *sum{ emb(i); i = 1..n } ; n = 3
    '''

    pass


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
            print(sample)
    tf.print("Execution time:", time.perf_counter() - start_time)


def test_verification():
    print("Processing enroll samples....")
    # buckets = FastFourierTransform.build_buckets()
    # signal = FastFourierTransform.load_wav('data/wav/enroll/19-227-0000.wav')
    (fs, signal) = wavfile.read('data/wav/enroll/19-227-0000.wav')
    print(signal.shape, signal[1:100])
    feature = FastFourierTransform.get_fft_spectrum(signal)
    print(feature[1:100])
    print(feature.shape)

    result = model.predict(feature.reshape(1, *feature.shape, 1))
    embeded = np.squeeze(result)
    print(embeded.shape)


def test_similarity_verification(spe):
    pass


extract_feature = lambda filename: FastFourierTransform.get_fft_spectrum(
    AudioSegment.from_file(filename)
        .set_channels(1)
        .get_array_of_samples())

extract_id = lambda filename: int(re.search(r"id(\d+)", filename).group(1))

if __name__ == '__main__':
    # test_verification()

    # (samples, id) = read_dataset(DATA_GLOB=DATA_GLOB)
    # feature = FastFourierTransform.get_fft_spectrum(samples)
    # embedding = model.predict(feature.reshape(1, *feature.shape, 1))
    # print(embedding)

    # VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB)
    # benchmark(ArtificialDataset(num_samples=10))
    v = VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB)
    v2 = VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB)
    # res = ArtificialDataset()
    # benchmark(v)
    embs1 = model.predict_generator(v, steps=5)
    embs2 = model.predict_generator(v2, steps=5)
    print(embs1, embs2)

    embs = v.next()
    model.predict(embs[0])

    filelist = DataFrame({
        "id": None,
        "filename": glob.glob(DATA_GLOB)
    })
    filelist.id = filelist.filename.apply(extract_id)
    filelist = filelist.sort_values(by="id").reindex()

    # model.predict_generator(VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB), steps=1)
    # model.predict_generator()
