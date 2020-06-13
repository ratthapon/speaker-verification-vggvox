# %%
import glob
import os
import re
import time

import tensorflow as tf
import numpy as np
from pandas import DataFrame
from scipy.io import wavfile

from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split

from config import constants as config
import vggvox.model
from data.dataset.dataset import VoxCeleb2Dataset, VoxCeleb2Dataset2
from feature.feature_extraction import FastFourierTransform

model = vggvox.model.vggvox_model()
model.load_weights(config.WEIGHTS_FILE)
# model.summary()

HOME = os.getenv("HOME", ".")
TRAIN_DIR = f"{HOME}/dataset/VoxCeleb2_simple10/aac/"
DATA_GLOB = TRAIN_DIR + '*/*/*.m4a'


# DATA_GLOB = TRAIN_DIR + "/id" + "00081/vnEsqFqzC8k/00200.m4a"  # TRAIN_DIR + '*/*/*.m4a'


# %%
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


def average_cosine_score(
        enrolled, test,
        axis=1
):
    '''
    Compute a score based on 3 enroll sample and use mean-reduced cosine distance
    score = 1 - MSE
    '''
    return 1 - tf.math.reduce_sum(tf.math.squared_difference(enrolled, test), axis=axis)


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

# %%
if __name__ == '__main__':
    # test_verification()

    # %% minimal vggvox usage
    # (samples, id) = read_dataset(DATA_GLOB=DATA_GLOB)
    # feature = FastFourierTransform.get_fft_spectrum(samples)
    # embedding = model.predict(feature.reshape(1, *feature.shape, 1))
    # print(embedding)

    # %% applying generator and dataset for large scale
    # VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB)
    # benchmark(ArtificialDataset(num_samples=10))
    v = VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB)
    v2 = VoxCeleb2Dataset(DATA_GLOB=DATA_GLOB)
    # res = ArtificialDataset()
    # benchmark(v)
    embs1 = model.predict_generator(v, steps=5)
    embs2 = model.predict_generator(v2, steps=5)
    print(embs1, embs2)
    scores = average_cosine_score(embs1, embs2[0, :, :, :])
    print("score", scores)
    print("score", np.squeeze(scores))

    embs = v.next()
    model.predict(embs[0])

    # %% Create simple dataset
    filelist = DataFrame({
        "id": None,
        "filename": glob.glob(DATA_GLOB)
    })
    filelist.id = filelist.filename.apply(extract_id)
    filelist = filelist.sort_values(by=["id", "filename"]).reindex()
    N_sample = 20
    samples = filelist.groupby("id").apply(lambda x: x.sample(N_sample, random_state=0)).reset_index(drop=True)

    # %% Extract embedding from dataset
    experiment_set = {}
    speaker_ids = samples.id.unique()
    for speaker_id in speaker_ids:
        enroll, test = train_test_split(
            samples[samples.id == speaker_id],
            train_size=10, random_state=0)

        experiment_set[speaker_id] = {}
        experiment_set[speaker_id]["enroll"] = enroll
        experiment_set[speaker_id]["test"] = test

        experiment_set[speaker_id]["enroll_embs"] = np.squeeze(
            model.predict_generator(
                VoxCeleb2Dataset2(filelist=enroll),
                steps=10
            )
        )
        experiment_set[speaker_id]["test_embs"] = np.squeeze(
            model.predict_generator(
                VoxCeleb2Dataset2(filelist=test),
                steps=10
            )
        )

    # %% Compute baseline score
    for speaker_id in speaker_ids:
        experiment_set[speaker_id]["baseline_scores"] = np.squeeze(
            cosine_distances(
                experiment_set[speaker_id]["enroll_embs"],
                experiment_set[speaker_id]["test_embs"]
            )
        )

    # %% Compute attacking score
    speaker_ids = samples.id.unique()
    for speaker_id in speaker_ids:
        experiment_set[speaker_id]["attacks_score"] = {}
        experiment_set[speaker_id]["avg_score"] = {}

        for attacker in speaker_ids:
            if attacker == speaker_id:
                experiment_set[speaker_id]["avg_score"][f"by_{speaker_id}"] = "{0:.4f}".format(
                    experiment_set[speaker_id]["baseline_scores"].mean()
                )
                continue

            score = np.squeeze(
                cosine_distances(
                    experiment_set[speaker_id]["enroll_embs"],
                    experiment_set[attacker]["test_embs"]
                )
            )

            print(f"Attacker id {attacker} attacked {speaker_id}, avg_damage={score.mean()}")

            experiment_set[speaker_id]["attacks_score"][f"by_{attacker}"] = score

            experiment_set[speaker_id]["avg_score"][f"by_{attacker}"] = "{0:.4f}".format(score.mean())

    #%%
    # clf = tf.reduce_sum(tf.math.squared_difference(model, model), axis=3)
    # tf.keras.utils.plot_model(clf, "combined_pretrained")
