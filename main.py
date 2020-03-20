import numpy as np
import scipy.io.wavfile as wav

from vggvox import constants as config
import vggvox.model
from vggvox.feature_extraction import FastFourierTransform


model = vggvox.model.vggvox_model()
model.load_weights(config.WEIGHTS_FILE)
model.summary()


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


if __name__ == '__main__':
    test_verification()
