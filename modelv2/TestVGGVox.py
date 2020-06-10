import tensorflow as tf


def TestVGGVox():
    model = tf.keras.models.load_model(
        "/home/rattaphon/speaker-verification/vggvox/model/tf_model.pb",
        custom_objects=None, compile=True
    )
    assert model is not None


if __name__ == "__main__":
    TestVGGVox()

