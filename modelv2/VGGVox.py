import tensorflow
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Reshape
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
import tensorflow as tf
import vggvox.constants as c


class VGGVox(Model):

    # Block of layers: Conv --> BatchNorm --> ReLU --> Pool
    def set_conv_bn_pool(
            self,
            inp_tensor,
            layer_idx,
            conv_filters,
            conv_kernel_size,
            conv_strides,
            conv_pad,
            pool='',
            pool_size=(2, 2),
            pool_strides=None,
            conv_layer_prefix='conv'
    ):
        setattr(self, f'pad{layer_idx}',
                ZeroPadding2D(
                    padding=conv_pad,
                    name=f'pad{layer_idx}'
                ))

        setattr(self, f'{conv_layer_prefix}{layer_idx}',
                Conv2D(
                    filters=conv_filters,
                    kernel_size=conv_kernel_size,
                    strides=conv_strides,
                    padding='valid',
                    name=f'{conv_layer_prefix}{layer_idx}'
                ))

        setattr(self, f'bn{layer_idx}',
                BatchNormalization(
                    epsilon=1e-5,
                    momentum=1,
                    name=f'bn{layer_idx}'
                ))

        setattr(self, f'relu{layer_idx}',
                Activation(
                    'relu',
                    name=f'relu{layer_idx}'
                ))
        if pool == 'max':
            setattr(self, f'mpool{layer_idx}',
                    MaxPooling2D(
                        pool_size=pool_size,
                        strides=pool_strides,
                        name=f'mpool{layer_idx}'
                    ))
        elif pool == 'avg':
            setattr(self, f'apool{layer_idx}',
                    AveragePooling2D(
                        pool_size=pool_size,
                        strides=pool_strides,
                        name=f'apool{layer_idx}'
                    ))

    # Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
    def set_conv_bn_dynamic_apool(
            self,
            inp_tensor,
            layer_idx,
            conv_filters,
            conv_kernel_size,
            conv_strides,
            conv_pad,
            conv_layer_prefix='conv'
    ):
        setattr(self, f'pad{layer_idx}',
                ZeroPadding2D(
                    padding=conv_pad,
                    name=f'pad{layer_idx}'
                ))

        setattr(self, f'{conv_layer_prefix}{layer_idx}',
                Conv2D(
                    filters=conv_filters,
                    kernel_size=conv_kernel_size,
                    strides=conv_strides,
                    padding='valid',
                    name=f'{conv_layer_prefix}{layer_idx}'
                ))

        setattr(self, f'bn{layer_idx}',
                BatchNormalization(
                    epsilon=1e-5,
                    momentum=1,
                    name=f'bn{layer_idx}'
                ))

        setattr(self, f'relu{layer_idx}',
                Activation(
                    'relu',
                    name=f'relu{layer_idx}'
                ))

        setattr(self, f'gapool{layer_idx}',
                GlobalAveragePooling2D(
                    name=f'gapool{layer_idx}'
                ))

        setattr(self, f'reshape{layer_idx}',
                Reshape(
                    (1, 1, conv_filters),
                    name=f'reshape{layer_idx}'
                ))

    # VGGVox verification model
    def __init__(self):
        super(VGGVox, self).__init__()

        self.set_conv_bn_pool(
            None, layer_idx=1,
            conv_filters=96,
            conv_kernel_size=(7, 7),
            conv_strides=(2, 2),
            conv_pad=(1, 1),
            pool='max',
            pool_size=(3, 3),
            pool_strides=(2, 2)
        )
        self.set_conv_bn_pool(
            None, layer_idx=2,
            conv_filters=256,
            conv_kernel_size=(5, 5),
            conv_strides=(2, 2),
            conv_pad=(1, 1),
            pool='max',
            pool_size=(3, 3),
            pool_strides=(2, 2)
        )
        self.set_conv_bn_pool(
            None, layer_idx=3,
            conv_filters=384,
            conv_kernel_size=(3, 3),
            conv_strides=(1, 1),
            conv_pad=(1, 1)
        )
        self.set_conv_bn_pool(
            None, layer_idx=4,
            conv_filters=256,
            conv_kernel_size=(3, 3),
            conv_strides=(1, 1),
            conv_pad=(1, 1)
        )
        self.set_conv_bn_pool(
            None, layer_idx=5,
            conv_filters=256,
            conv_kernel_size=(3, 3),
            conv_strides=(1, 1),
            conv_pad=(1, 1),
            pool='max',
            pool_size=(5, 3),
            pool_strides=(3, 2)
        )

        self.set_conv_bn_dynamic_apool(
            None, layer_idx=6,
            conv_filters=4096,
            conv_kernel_size=(9, 1),
            conv_strides=(1, 1),
            conv_pad=(0, 0),
            conv_layer_prefix='fc'
        )
        self.set_conv_bn_pool(
            None, layer_idx=7,
            conv_filters=1024,
            conv_kernel_size=(1, 1),
            conv_strides=(1, 1),
            conv_pad=(0, 0),
            conv_layer_prefix='fc'
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mpool1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mpool2(x)

        x = self.pad3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.pad4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pad5(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.mpool5(x)

        x = self.pad6(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.gapool6(x)
        x = self.reshape6(x)

        x = self.pad7(x)
        x = self.fc7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.norm(x)
        x = self.fc8(x)

        return x

    def load_weights(self, WEIGHTS_FILE):
        self.model.load_weights(c.WEIGHTS_FILE)

    def summary(self):
        tf.keras.utils.plot_model(
            self, to_file='model.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )

        writer = tf.summary.create_file_writer("/tmp/mylogs")
        with writer.as_default():
            for step in range(100):
                # other model code would go here
                tf.summary.scalar("my_metric", 0.5, step=step)
                writer.flush()


def test():
    model = VGGVox()
    # model.load_weights(c.WEIGHTS_FILE)
    model.summary()
    print(model)


if __name__ == '__main__':
    test()
