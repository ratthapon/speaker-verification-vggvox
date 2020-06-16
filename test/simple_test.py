import itertools
import os

import pandas
import tensorflow as tf
import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from config import constants as config
import vggvox.model
from data.dataset.VoxCeleb2Dataset import VoxCeleb2Dataset

np.set_printoptions(precision=4)
pandas.set_option("display.precision", 4)

## disable to speed up, enable for as_numpy_iterator() execution
## TODO refactor as_numpy_iterator to TF2 with disable eager execution
# tf.compat.v1.disable_eager_execution()



def average_cosine_score(
        enrolled, test,
        axis=1
):
    '''
    Compute a score based on 3 enroll sample and use mean-reduced cosine distance
    score = 1 - MSE
    '''
    scorer = tf.keras.metrics.CosineSimilarity(axis=0)
    _ = scorer.update_state(enrolled, test)
    return scorer.result().numpy()

    # # faster cosine sim using matrix ops
    # normalize_a = tf.nn.l2_normalize(enrolled, 1)
    # normalize_b = tf.nn.l2_normalize(test, 1)
    # similarity = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    # return similarity

    # return 1 - tf.math.reduce_sum(tf.math.squared_difference(enrolled, test), axis=axis)


model = vggvox.model.vggvox_model()
model.load_weights(config.WEIGHTS_FILE)

if __name__ == "__main__":
    HOME = os.getenv("HOME", ".")
    REF_DIR = f"{HOME}/dataset/VoxCeleb2_simple3/ref/"
    REF_GLOB = REF_DIR + '*/*/*.m4a'
    EVAL_DIR = f"{HOME}/dataset/VoxCeleb2_simple3/eval/"
    EVAL_GLOB = EVAL_DIR + '*/*/*.m4a'

    results = DataFrame(columns=["ref_id", "eval_id", "score", "ref_filename", "eval_filename"])
    len_ref = len(list(VoxCeleb2Dataset.meta_generator(REF_GLOB)))
    len_eval = len(list(VoxCeleb2Dataset.meta_generator(EVAL_GLOB)))
    total_matches = len_ref * len_eval
    print(f"len_ref: {len_ref}, len_eval: {len_eval}, total matches: {total_matches}")

    batch_size = 4
    n_batch = 1  # len(dataset) / batch_size
    v = VoxCeleb2Dataset(DATA_GLOB=REF_GLOB, batch=len_ref)
    v2 = VoxCeleb2Dataset(DATA_GLOB=EVAL_GLOB, batch=len_eval)

    ref_metagen = VoxCeleb2Dataset.meta_generator(REF_GLOB)
    eval_metagen = VoxCeleb2Dataset.meta_generator(EVAL_GLOB)

    with tqdm(total=total_matches) as progress_bar:
        # prefetch
        ref_data, _ = v.next()
        eval_data, _ = v2.next()
        ref_id, ref_fname = ref_metagen.__next__()
        eval_id, eval_fname = eval_metagen.__next__()
        progress_bar.update(1)

        for ref_idx in range(0, len_ref):
            ref_template = np.squeeze(model.predict(ref_data, steps=1))

            for eval_idx in range(0, len_eval):
                eval_template = np.squeeze(model.predict(eval_data, steps=1))

                result = DataFrame({
                    "ref_id": [ref_id],
                    "eval_id": [eval_id],
                    "score": [average_cosine_score(ref_template, eval_template)],
                    "ref_filename": [ref_fname],
                    "eval_filename": [eval_fname]
                })
                results = results.append(result)
                progress_bar.update(1)

                try:
                    eval_data, _ = v2.next()
                    eval_id, eval_fname = eval_metagen.__next__()
                except StopIteration:
                    # TODO reset eval_metagen, v2
                    # TODO prefetch sample
                    v2 = VoxCeleb2Dataset(DATA_GLOB=EVAL_GLOB, batch=len_eval)
                    eval_metagen = VoxCeleb2Dataset.meta_generator(EVAL_GLOB)
                    eval_data, _ = v2.next()
                    eval_id, eval_fname = eval_metagen.__next__()

                    results.to_csv("spkr_ver_scores.csv")
                    break

            try:
                ref_data, _ = v.next()
                ref_id, ref_fname = ref_metagen.__next__()
            except StopIteration:
                print("Finished")
                break

        results.to_csv("spkr_ver_scores.csv")
        print(results)
