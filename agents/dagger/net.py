import tensorflow as tf
import tensorflow_hub as hub

from agents.dagger.layers import conv2d, max_pool_2x2, linear, lrn
import config as c


ALEXNET_NAME = 'AlexNet'
ALEXNET_FC7 = 4096

MOBILENET_V2_NAME = 'MobileNetV2'
MOBILENET_V2_TFHUB_MODULE = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/1'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


class Net(object):
    def __init__(self, x, num_targets=c.NUM_TARGETS, is_training=True, num_last_hidden=ALEXNET_FC7, net_name=ALEXNET_NAME):
        self.input = x
        self.num_targets = num_targets
        self.is_training = is_training
        self.num_last_hidden = num_last_hidden
        self.net_name = net_name
        if net_name == ALEXNET_NAME:
            init_fn = self.init_alexnet
        elif net_name == MOBILENET_V2_NAME:
            init_fn = self.init_mobilenet_v2
        else:
            raise NotImplementedError('net_name %r not recognized' % net_name)

        self.last_hidden_layer_activations, self.out = init_fn()

    def init_alexnet(self):
        """AlexNet architecture with modified final fully-connected layers regressed on driving control outputs (steering, throttle, etc...)"""

        # phase = tf.placeholder(tf.bool, name='phase')  # Used for batch norm

        conv1 = tf.nn.relu(conv2d(self.input, "conv1", 96, 11, 4, 1))
        lrn1 = lrn(conv1)
        maxpool1 = max_pool_2x2(lrn1)
        conv2 = tf.nn.relu(conv2d(maxpool1, "conv2", 256, 5, 1, 2))
        lrn2 = lrn(conv2)
        maxpool2 = max_pool_2x2(lrn2)
        conv3 = tf.nn.relu(conv2d(maxpool2, "conv3", 384, 3, 1,
                                  1))  # Not sure why this isn't 2 groups, but pretrained net was trained like this so we're going with it.

        # Avoid diverging from pretrained weights with things like batch norm for now.
        # Perhaps try a modern small net like Inception V1, ResNet 18, or Resnet 50
        # conv3 = tf.contrib.layers.batch_norm(conv3, scope='batchnorm3', is_training=phase,
        #     # fused=True,
        #     # data_format='NCHW',
        #     # renorm=True
        # )

        conv4 = tf.nn.relu(conv2d(conv3, "conv4", 384, 3, 1, 2))
        conv5 = tf.nn.relu(conv2d(conv4, "conv5", 256, 3, 1, 2))
        maxpool5 = max_pool_2x2(conv5)
        fc6 = tf.nn.relu(linear(maxpool5, "fc6", 4096))
        if self.is_training:
            fc6 = tf.nn.dropout(fc6, 0.5)
        else:
            fc6 = tf.nn.dropout(fc6, 1.0)

        fc7 = tf.nn.relu(linear(fc6, "fc7", self.num_last_hidden))
        # fc7 = tf.contrib.layers.batch_norm(fc7, scope='batchnorm7', is_training=phase)
        if self.is_training:
            fc7 = tf.nn.dropout(fc7, 0.95)
        else:
            fc7 = tf.nn.dropout(fc7, 1.0)

        fc8 = linear(fc7, "fc8", self.num_targets)

        return fc7, fc8

    def init_mobilenet_v2(self):
        module_spec = hub.load_module_spec(MOBILENET_V2_TFHUB_MODULE)
        height, width = hub.get_expected_image_size(module_spec)
        graph = tf.get_default_graph()
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        hub_module = hub.Module(module_spec, trainable=False)
        pretrained_output_tensor = hub_module(resized_input_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS
                                 for node in graph.as_graph_def().node)

        self.input = resized_input_tensor
        return self.add_tfhub_retrain_ops(pretrained_output_tensor)

    def add_tfhub_retrain_ops(self, last_hidden_layer_activations):
        batch_size, input_tensor_size = last_hidden_layer_activations.get_shape().as_list()
        assert batch_size is None, 'We want to work with arbitrary batch size.'
        with tf.name_scope('input'):
            fc_input_placeholder = tf.placeholder_with_default(
                last_hidden_layer_activations,
                shape=[batch_size, input_tensor_size],
                name='TfHubInputPlaceHolder')

            variable_summaries(fc_input_placeholder)

            # ground_truth_input = tf.placeholder(
            #     tf.int64, [batch_size], name='GroundTruthInput')
        # Organizing the following ops so they are easier to see in TensorBoard.
        layer_name = 'final_retrain_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value1 = tf.truncated_normal(
                    [input_tensor_size, input_tensor_size], stddev=0.001)
                initial_value2 = tf.truncated_normal(
                    [input_tensor_size, c.NUM_TARGETS], stddev=0.001)
                fc_weights_1 = tf.Variable(initial_value1, name='fc_1')
                fc_weights_2 = tf.Variable(initial_value2, name='fc_2')
                variable_summaries(fc_weights_1)
                variable_summaries(fc_weights_2)

            with tf.name_scope('biases'):
                fc_biases_1 = tf.Variable(tf.zeros([input_tensor_size]), name='fc_biases_1')
                fc_biases_2 = tf.Variable(tf.zeros([c.NUM_TARGETS]), name='fc_biases_2')
                variable_summaries(fc_biases_1)
                variable_summaries(fc_biases_2)

            with tf.name_scope('Wx_plus_b'):
                fc_1_activations = tf.nn.relu(tf.matmul(fc_input_placeholder, fc_weights_1) + fc_biases_1)
                fc_2_activations = tf.matmul(fc_1_activations, fc_weights_2) + fc_biases_2
                tf.summary.histogram('fc_1_activations', fc_1_activations)
                tf.summary.histogram('fc_2_activations', fc_2_activations)

            return fc_1_activations, fc_2_activations


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
