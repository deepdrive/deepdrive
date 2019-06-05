import tensorflow as tf

from agents.dagger.layers import conv2d, max_pool_2x2, linear, lrn
import config as c
import logs
from config import ALEXNET_NAME, ALEXNET_FC7, ALEXNET_IMAGE_SHAPE, \
    MOBILENET_V2_SLIM_NAME, MOBILENET_V2_IMAGE_SHAPE
from util.download import download, has_stuff
from vendor.tensorflow.models.research.slim.nets import nets_factory
from vendor.tensorflow.models.research.slim.preprocessing import preprocessing_factory

log = logs.get_log(__name__)

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


class Net(object):
    def __init__(self, global_step=None, num_targets=c.NUM_TARGETS, is_training=True,
                 freeze_pretrained=False, overfit=False):
        self.num_targets = num_targets
        self.is_training = is_training
        self.global_step = global_step
        self.freeze_pretrained = freeze_pretrained  # TODO: Implement for AlexNet
        self.overfit = overfit
        self.batch_size = None
        self.learning_rate = None
        self.starter_learning_rate = None
        self.weight_decay = None
        self.input, self.last_hidden, self.out, self.eval_out = self._init_net()
        self.num_last_hidden = self.last_hidden.shape[-1]

    def get_tf_init_fn(self, init_op):
        raise NotImplementedError('get_tf_init_fn not implemented')

    def preprocess(self, image):
        return image

    def _init_net(self):
        raise NotImplementedError('init_net not implemented')


class MobileNetV2(Net):
    def __init__(self, *args, **kwargs):
        self.input_image_shape = MOBILENET_V2_IMAGE_SHAPE
        self.image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            MOBILENET_V2_SLIM_NAME, is_training=False)
        super(MobileNetV2, self).__init__(*args, **kwargs)
        if self.is_training:
            if self.freeze_pretrained:
                self.batch_size = 48
                self.starter_learning_rate = 1e-3
            else:
                self.batch_size = 32
                self.starter_learning_rate = 1e-3
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step=self.global_step,
                                                            decay_steps=55000, decay_rate=0.5, staircase=True)
            if self.overfit:
                self.weight_decay = 0.
            else:
                self.weight_decay = 0.00004  # same as mobilenet2 paper https://arxiv.org/pdf/1801.04381.pdf
            self.mute_spurious_targets = True

    def get_tf_init_fn(self, init_op):
        def ret(ses):
            log.info('Initializing parameters.')
            ses.run(init_op)
        return ret

    def preprocess(self, image):
        image = self.image_preprocessing_fn(image, self.input_image_shape[0], self.input_image_shape[1])
        image = tf.expand_dims(image, 0)
        return image

    def _init_net(self):
        in_tensor = tf.placeholder(
            tf.uint8, [None] + list(MOBILENET_V2_IMAGE_SHAPE))
        network_fn = nets_factory.get_network_fn(
            MOBILENET_V2_SLIM_NAME,
            num_classes=None,
            num_targets=6,
            is_training=False, )
        log.info('Loading mobilenet v2')
        image = self.preprocess(in_tensor)
        out, endpoints = network_fn(image)
        last_hidden = endpoints['global_pool']
        eval_out = out

        return in_tensor, last_hidden, out, eval_out

class AlexNet(Net):
    def __init__(self, *args, **kwargs):
        self.input_image_shape = ALEXNET_IMAGE_SHAPE
        self.num_last_hidden = ALEXNET_FC7
        self.net_name = ALEXNET_NAME
        self.starter_learning_rate = 2e-6
        super(AlexNet, self).__init__(*args, **kwargs)

        if self.is_training:
            # Decrease this to fit in your GPU's memory
            # If you increase, remember that it decreases accuracy https://arxiv.org/abs/1711.00489
            self.batch_size = 32

            # TODO: add polyak averaging.
            self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step=self.global_step,
                                                            decay_steps=73000, decay_rate=0.5, staircase=True)

            if self.overfit:
                self.weight_decay = 0.
            else:
                self.weight_decay = 0.0005

            self.mute_spurious_targets = False

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
            if not has_stuff(c.ALEXNET_PRETRAINED_PATH):
                print('\n--------- ImageNet checkpoint not found, downloading ----------')
                download(c.ALEXNET_PRETRAINED_URL, c.WEIGHTS_DIR, warn_existing=False, overwrite=True)
            ses.run(init_op)
            alexnet_saver.restore(ses, c.ALEXNET_PRETRAINED_PATH)

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







