import tensorflow as tf
import tensorflow_hub as hub

from agents.dagger.layers import conv2d, max_pool_2x2, linear, lrn
import config as c
import logs
from utils import download, has_stuff

log = logs.get_log(__name__)


ALEXNET_NAME = 'AlexNet'
ALEXNET_FC7 = 4096
ALEXNET_IMAGE_SHAPE = (227, 227, 3)

MOBILENET_V2_NAME = 'MobileNetV2'
MOBILENET_V2_IMAGE_SHAPE = (192, 192, 3)
MOBILENET_V2_TFHUB_MODULE = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/1'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


class Net(object):
    def __init__(self, global_step, num_targets=c.NUM_TARGETS, is_training=True):
        self.num_targets = num_targets
        self.is_training = is_training
        self.global_step = global_step
        self.input, self.last_hidden, self.out, self.eval_out = self._init_net()
        self.input_image_shape = (self.input.shape[1].value, self.input.shape[2].value, self.input.shape[3].value)
        self.num_last_hidden = self.last_hidden.shape[1].value

    def get_tf_init_fn(self, init_op):
        raise NotImplementedError('get_tf_init_fn not implemented')

    def _init_net(self):
        raise NotImplementedError('init_net not implemented')


class MobileNetV2(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starter_learning_rate = 1.0
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step=self.global_step,
                                                        decay_steps=255, decay_rate=0.5, staircase=True)

    def get_tf_init_fn(self, init_op):
        def ret(ses):
            log.info('Initializing parameters.')
            ses.run(init_op)

        return ret

    def _init_net(self):
        module_spec = hub.load_module_spec(MOBILENET_V2_TFHUB_MODULE)
        height, width = hub.get_expected_image_size(module_spec)
        graph = tf.get_default_graph()
        in_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        hub_module = hub.Module(module_spec, trainable=False)
        pretrained_output_tensor = hub_module(in_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS
                                 for node in graph.as_graph_def().node)

        last_hidden, out = self._add_tfhub_retrain_ops(pretrained_output_tensor)

        # TODO: Make this use an eval graph, to avoid quantization
        # moving averages being updated by the validation set, though in
        # practice this makes a negligible difference.
        eval_out = out

        return in_tensor, last_hidden, out, eval_out

    @staticmethod
    def _add_tfhub_retrain_ops(last_hidden_layer_activations):
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


class AlexNet(Net):
    def __init__(self, *args, **kwargs):
        self.input_image_shape = ALEXNET_IMAGE_SHAPE
        self.num_last_hidden = ALEXNET_FC7
        self.net_name = ALEXNET_NAME
        self.starter_learning_rate = 2e-6
        super().__init__(*args, **kwargs)

        # TODO: add polyak averaging.
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step=self.global_step,
                                                        decay_steps=73000, decay_rate=0.5, staircase=True)

    def _init_net(self):
        in_tensor = tf.placeholder(tf.float32, (None,) + self.input_image_shape)

        with tf.variable_scope("model") as variable_scope:
            last_hidden, out = self._init_alexnet(in_tensor, self.is_training, self.num_targets,
                                                  self.num_last_hidden)
            if self.is_training:
                variable_scope.reuse_variables()
                _, eval_out = self._init_alexnet(in_tensor, False, self.num_targets, self.num_last_hidden)
            else:
                eval_out = None

        return in_tensor, last_hidden, out, eval_out

    def get_tf_init_fn(self, init_op_):
        return self._load_alexnet_pretrained(init_op_)

    @staticmethod
    def _load_alexnet_pretrained(init_op):
        pretrained_var_map = {}
        for v in tf.trainable_variables():
            found = False
            for bad_layer in ["fc6", "fc7", "fc8"]:
                if bad_layer in v.name:
                    found = True
            if found:
                continue

            pretrained_var_map[v.op.name[6:]] = v
        alexnet_saver = tf.train.Saver(pretrained_var_map)

        def init_fn(ses):
            log.info('Initializing parameters.')
            if not has_stuff(c.BVLC_CKPT_PATH):
                print('\n--------- ImageNet checkpoint not found, downloading ----------')
                download(c.BVLC_CKPT_URL, c.WEIGHTS_DIR, warn_existing=False, overwrite=True)
            ses.run(init_op)
            alexnet_saver.restore(ses, c.BVLC_CKPT_PATH)

        return init_fn

    @staticmethod
    def _init_alexnet(in_tensor, is_training, num_targets, num_last_hidden):
        """AlexNet architecture with modified final fully-connected layers regressed on driving control outputs (steering, throttle, etc...)"""
        # phase = tf.placeholder(tf.bool, name='phase')  # Used for batch norm

        conv1 = tf.nn.relu(conv2d(in_tensor, "conv1", 96, 11, 4, 1))
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
        if is_training:
            fc6 = tf.nn.dropout(fc6, 0.5)
        else:
            fc6 = tf.nn.dropout(fc6, 1.0)

        fc7 = tf.nn.relu(linear(fc6, "fc7", num_last_hidden))
        # fc7 = tf.contrib.layers.batch_norm(fc7, scope='batchnorm7', is_training=phase)
        if is_training:
            fc7 = tf.nn.dropout(fc7, 0.95)
        else:
            fc7 = tf.nn.dropout(fc7, 1.0)

        fc8 = linear(fc7, "fc8", num_targets)

        return fc7, fc8


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







