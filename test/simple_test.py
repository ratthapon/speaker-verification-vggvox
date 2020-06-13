import os

import tensorflow as tf
import numpy as np

from config import constants as config
import vggvox.model
from data.dataset.VoxCeleb2Dataset import VoxCeleb2Dataset
from model.VGGVox import VGGVox


def average_cosine_score(
        enrolled, test,
        axis=1
):
    '''
    Compute a score based on 3 enroll sample and use mean-reduced cosine distance
    score = 1 - MSE
    '''
    return 1 - tf.math.reduce_sum(tf.math.squared_difference(enrolled, test), axis=axis)


model = vggvox.model.vggvox_model()
model.load_weights(config.WEIGHTS_FILE)

if __name__ == "__main__":
    HOME = os.getenv("HOME", ".")
    REF_DIR = f"{HOME}/dataset/VoxCeleb2_simple3/ref/"
    REF_GLOB = REF_DIR + '*/*/*.m4a'
    EVAL_DIR = f"{HOME}/dataset/VoxCeleb2_simple3/eval/"
    EVAL_GLOB = EVAL_DIR + '*/*/*.m4a'
    # TODO flatten files in datadir from '*/*/*.m4a' to '*/*.m4a'
    n = 16
    v = VoxCeleb2Dataset(DATA_GLOB=REF_GLOB, batch=n)
    v2 = VoxCeleb2Dataset(DATA_GLOB=EVAL_GLOB, batch=n)
    embs1 = model.predict_generator(v, steps=n)
    embs2 = model.predict_generator(v2, steps=n)
    # print(embs1, embs2, embs1.shape)
    scores = average_cosine_score(embs1, embs2)
    print("score", scores)
    # print("score", np.squeeze(scores))
