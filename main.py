import numpy as np

from vggvox import constants as config
import vggvox.model
from vggvox import feature_extraction


model = vggvox.model.vggvox_model()
model.load_weights(config.WEIGHTS_FILE)
model.summary()


def test_verification():
    print("Processing enroll samples....")
    buckets = feature_extraction.build_buckets()
    feature = feature_extraction.get_fft_spectrum('data/wav/enroll/19-227-0000.wav', buckets)
    print(feature.shape)

    result = model.predict(feature.reshape(1, *feature.shape, 1))
    embeded = np.squeeze(result)

# enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC)
# enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
# speakers = enroll_result['speaker']

# print("Processing test samples....")
# test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
# test_embs = np.array([emb.tolist() for emb in test_result['embedding']])
#
# print("Comparing test samples against enroll samples....")
# distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers)
#
# scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
# scores = pd.concat([scores, distances],axis=1)
# scores['result'] = scores[speakers].idxmin(axis=1)
# scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int
#
# print("Writing outputs to [{}]....".format(c.RESULT_FILE))
# result_dir = os.path.dirname(c.RESULT_FILE)
# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)
# with open(c.RESULT_FILE, 'w') as f:
# 	scores.to_csv(f, index=False)


if __name__ == '__main__':
    test_verification()
