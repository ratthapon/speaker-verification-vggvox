import keras.backend as K
import tensorflow.compat.v1 as tf
from keras.layers import Input, GlobalAveragePooling2D, Reshape
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers import Lambda, Activation
from keras.layers import BatchNormalization
from keras.models import Model
import vggvox.constants as c


class VGGVox:

    # Block of layers: Conv --> BatchNorm --> ReLU --> Pool
    def conv_bn_pool(
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

        x = ZeroPadding2D(
            padding=conv_pad,
            name='pad{}'.format(layer_idx)
        )(inp_tensor)
        print(inp_tensor)
        print(x)
        x = Conv2D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            strides=conv_strides,
            padding='valid',
            name='{}{}'.format(conv_layer_prefix, layer_idx)
        )(x)
        print(x)
        x = BatchNormalization(
            epsilon=1e-5,
            momentum=1,
            name='bn{}'.format(layer_idx)
        )(x)
        x = Activation(
            'relu',
            name='relu{}'.format(layer_idx)
        )(x)
        if pool == 'max':
            x = MaxPooling2D(
                pool_size=pool_size,
                strides=pool_strides,
                name='mpool{}'.format(layer_idx)
            )(x)
        elif pool == 'avg':
            x = AveragePooling2D(
                pool_size=pool_size,
                strides=pool_strides,
                name='apool{}'.format(layer_idx)
            )(x)
        return x

    # Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
    def conv_bn_dynamic_apool(
            self,
            inp_tensor,
            layer_idx,
            conv_filters,
            conv_kernel_size,
            conv_strides,
            conv_pad,
            conv_layer_prefix='conv'
    ):
        x = ZeroPadding2D(
            padding=conv_pad,
            name='pad{}'.format(layer_idx)
        )(inp_tensor)
        x = Conv2D(
            filters=conv_filters,
            kernel_size=conv_kernel_size,
            strides=conv_strides,
            padding='valid',
            name='{}{}'.format(conv_layer_prefix, layer_idx)
        )(x)
        x = BatchNormalization(
            epsilon=1e-5,
            momentum=1,
            name='bn{}'.format(layer_idx)
        )(x)
        x = Activation(
            'relu',
            name='relu{}'.format(layer_idx)
        )(x)
        x = GlobalAveragePooling2D(
            name='gapool{}'.format(layer_idx)
        )(x)
        x = Reshape(
            (1, 1, conv_filters),
            name='reshape{}'.format(layer_idx)
        )(x)
        return x

    # VGGVox verification model
    def __init__(self):
        inp = Input(
            c.INPUT_SHAPE,
            name='input')
        x = self.conv_bn_pool(
            inp, layer_idx=1,
            conv_filters=96,
            conv_kernel_size=(7, 7),
            conv_strides=(2, 2),
            conv_pad=(1, 1),
            pool='max',
            pool_size=(3, 3),
            pool_strides=(2, 2)
        )
        x = self.conv_bn_pool(
            x, layer_idx=2,
            conv_filters=256,
            conv_kernel_size=(5, 5),
            conv_strides=(2, 2),
            conv_pad=(1, 1),
            pool='max',
            pool_size=(3, 3),
            pool_strides=(2, 2)
        )
        x = self.conv_bn_pool(
            x, layer_idx=3,
            conv_filters=384,
            conv_kernel_size=(3, 3),
            conv_strides=(1, 1),
            conv_pad=(1, 1)
        )
        x = self.conv_bn_pool(
            x, layer_idx=4,
            conv_filters=256,
            conv_kernel_size=(3, 3),
            conv_strides=(1, 1),
            conv_pad=(1, 1)
        )
        x = self.conv_bn_pool(
            x, layer_idx=5,
            conv_filters=256,
            conv_kernel_size=(3, 3),
            conv_strides=(1, 1),
            conv_pad=(1, 1),
            pool='max',
            pool_size=(5, 3),
            pool_strides=(3, 2)
        )
        x = self.conv_bn_dynamic_apool(
            x, layer_idx=6,
            conv_filters=4096,
            conv_kernel_size=(9, 1),
            conv_strides=(1, 1),
            conv_pad=(0, 0),
            conv_layer_prefix='fc'
        )
        x = self.conv_bn_pool(
            x, layer_idx=7,
            conv_filters=1024,
            conv_kernel_size=(1, 1),
            conv_strides=(1, 1),
            conv_pad=(0, 0),
            conv_layer_prefix='fc'
        )
        x = Lambda(
            lambda y: K.l2_normalize(y, axis=3),
            name='norm'
        )(x)
        x = Conv2D(
            filters=1024,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            name='fc8'
        )(x)
        self.model = Model(inp, x, name='VGGVox')

    def load_weights(self, WEIGHTS_FILE):
        self.model.load_weights(c.WEIGHTS_FILE)

    def summary(self):
        return self.model.summary()


def test():
    model = VGGVox()
    model.load_weights(c.WEIGHTS_FILE)
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.model.outputs])
    # Save to ./model/tf_model.pb
    tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)
    # model.load_weights(c.WEIGHTS_FILE)
    model.summary()


    # with open("../models/model_arch.json", "w") as model_arch_file:
    # json.dump(json.loads(model.to_json()), model_arch_file, indent=2)
    # json.dump(model.to_json(), model_arch_file)
    # json.dump(json.loads(model.to_json()), model_arch_file)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


if __name__ == '__main__':
    test()
