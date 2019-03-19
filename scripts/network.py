from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
#from keras_gcnn.layers import GBatchNorm, GConv2D
from keras_gcnn.layers.pooling import GroupPool
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'
MOVING_AVERAGE_DECAY = 0.98
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.0002
CONV_WEIGHT_STDDEV = 0.1
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops' # must be grouped with training op

BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, is_training, trainable=True, num_classes=21):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.update_bn = tf.placeholder_with_default(tf.constant(False),
                                                       shape=[],
                                                       name='update_bn')

        self.is_training = is_training

        self.g_bn_list = []

        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='latin1').item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        if 'bn' in op_name:
                            param_name = BN_param_map[param_name]
                            data = np.squeeze(data)

                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, trainable_param=True, initializer_param=tf.keras.initializers.he_normal()):
        '''Creates a new TensorFlow variable.'''
        #return tf.get_variable(name, shape, trainable=self.trainable, initializer=tf.contrib.layers.xavier_initializer())
        return tf.get_variable(name, shape, trainable=trainable_param, initializer=initializer_param)

    def get_layer_name(self):
        return layer_name

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def zero_padding(self, input, paddings, name):
        pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
        return tf.pad(input, paddings=pad_mat, name=name)

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding,data_format=DEFAULT_DATAFORMAT)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

#    @layer
#    def trans_conv(self, input, k=3, s_h=2, s_w=2, name, relu=True, padding='SAME', group=1, biased=True):

#        self.validate_padding(padding)

#        c_i = input.get_shape()[-1]

#        if relu:
#            return tf.layers.conv2d_transpose(i, c_i, k, strides=(s_h, s_w), padding=padding, use_bias=biased, kernel_initializer=tf.truncated_normal_initializer, name=name, activation = tf.nn.relu)
#        else:
#            return tf.layers.conv2d_transpose(i, c_i, k, strides=(s_h, s_w),padding=padding, use_bias=biased, kernel_initializer=tf.truncated_normal_initializer, name=name)

    @layer
    def upsampling(self,
            input,
            name):

        up_size = tf.convert_to_tensor(input.shape[1:3])
        up_size = up_size * 2

        output = tf.image.resize_nearest_neighbor(input, up_size, align_corners=True, name=name)

        return output


    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]

        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
            output = convolve(input, kernel)

            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        output = tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name,
                              data_format=DEFAULT_DATAFORMAT)
        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        output = tf.layers.batch_normalization(
            input,
            momentum=0.95,
            epsilon=1e-5,
            training=self.is_training,
            name=name
        )

        if relu:
            output = tf.nn.relu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, is_training, name):
        if is_training:
            keep = keep_prob
        else:
            keep = 1
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        return tf.image.resize_bilinear(input, size=size, align_corners=True, name=name)

    @layer
    def selu(self, input, name):
        return tf.nn.selu(input, name=name)


    @layer
    def residual_block(self, input, name, use_dropout=False, padding='SAME'):
        c_i = input.get_shape()[-1]

        with tf.variable_scope(name) as scope:
            output = tf.layers.batch_normalization(input, momentum=0.95, epsilon=1e-5,training=self.is_training)

            kernel1 = self.make_var('weights1', shape=[3, 3, c_i, c_i])
            output = tf.nn.conv2d(output, kernel1, [1, 1, 1, 1], padding=padding,data_format=DEFAULT_DATAFORMAT)
            bias1 = self.make_var('bias1', [c_i])
            output = tf.nn.bias_add(output, bias1)

            output = tf.layers.batch_normalization(output, momentum=0.95, epsilon=1e-5,training=self.is_training)

            output = tf.nn.relu(output, name='relu1')

            if use_dropout:
                keep_prob = 0.8
                if not self.is_training:
                    keep_prob = 1.0
                output = tf.nn.dropout(output, keep_prob, name='dropout')

            kernel2 = self.make_var('weights2', shape=[3, 3, c_i, c_i])
            output = tf.nn.conv2d(output, kernel2, [1, 1, 1, 1], padding=padding, data_format=DEFAULT_DATAFORMAT)
            bias2 = self.make_var('bias2', [c_i])
            output = tf.nn.bias_add(output, bias2)

            output = tf.layers.batch_normalization(output, momentum=0.95, epsilon=1e-5, training=self.is_training)

            output = tf.add(output, input)

            return output


    @layer
    def g_conv(self,
             input,
             input_type,
             output_type,
             kernel_size,
             c_o,
             s_h,
             s_w,
             name,
             padding=DEFAULT_PADDING,
             group=1):
        # Verify that the padding is acceptable
        self.validate_padding(padding)

        c_i = np.int32(input.get_shape()[-1])

        if input_type == 'Z2':
            c_i = c_i
        elif input_type == 'C4':
            c_i = c_i // 4
        else:
            c_i = c_i // 8

        #output = GConv2D(c_o, (kernel_size, kernel_size), kernel_initializer='he_normal', padding=padding, strides=(s_h, s_w), use_bias=False, h_input=input_type, h_output=output_type, name=name)(input)

        with tf.variable_scope(name) as scope:

            gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
                h_input=input_type, h_output=output_type, in_channels=c_i, out_channels=c_o, ksize=kernel_size)
            #w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
            w = self.make_var(name='weight', shape=w_shape)
            output = gconv2d(input=input, filter=w, strides=[1, 1, 1, 1], padding='SAME', gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)

            return output

    def g_conv_no_decorator(self,
               input,
               input_type,
               output_type,
               kernel_size,
               c_o,
               s_h,
               s_w,
               name,
               padding=DEFAULT_PADDING,
               group=1):
        # Verify that the padding is acceptable
        self.validate_padding(padding)

        c_i = np.int32(input.get_shape()[-1])

        if input_type == 'Z2':
            c_i = c_i
        elif input_type == 'C4':
            c_i = c_i // 4
        else:
            c_i = c_i // 8

        # output = GConv2D(c_o, (kernel_size, kernel_size), kernel_initializer='he_normal', padding=padding, strides=(s_h, s_w), use_bias=False, h_input=input_type, h_output=output_type, name=name)(input)

        with tf.variable_scope(name) as scope:

            gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
                h_input=input_type, h_output=output_type, in_channels=c_i, out_channels=c_o, ksize=kernel_size)
            # w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))
            w = self.make_var(name='weight', shape=w_shape)
            output = gconv2d(input=input, filter=w, strides=[1, 1, 1, 1], padding='SAME', gconv_indices=gconv_indices,
                             gconv_shape_info=gconv_shape_info)

            return output

    # @layer
    # def g_batch_normalization(self, input, name, input_type, scale_offset=True):
    #     with tf.variable_scope(name) as scope:
    #         output = GBatchNorm(input_type)(input,training=self.is_training)
    #         return output

    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        with tf.name_scope(name) as scope_name:
            with tf.variable_scope(scope_name) as scope:
                try:
                    return tf.get_variable(name,
                                           shape=shape,
                                           initializer=initializer,
                                           regularizer=regularizer,
                                           trainable=trainable)
                except:
                    scope.reuse_variables()
                    return tf.get_variable(name,
                                           shape=shape,
                                           initializer=initializer,
                                           regularizer=regularizer,
                                           trainable=trainable)

    @layer
    def g_batch_norm_tensorflow(self, input, name, input_type):

        with tf.variable_scope(name):
            with tf.variable_scope('BatchNorm') as scope:
                x_shape = input.get_shape()
                params_shape = x_shape[-1:]

                axis = list(range(len(x_shape) - 1))

                one_init = tf.constant_initializer(value=1.0)
                zero_init = tf.constant_initializer(value=0.0)

                beta = self.make_var('beta', params_shape, initializer_param=zero_init)
                gamma = self.make_var('gamma', params_shape, initializer_param=one_init)
                moving_mean = self.make_var('moving_mean', params_shape, trainable_param=False, initializer_param=zero_init)
                moving_variance = self.make_var('moving_variance', params_shape, trainable_param=False, initializer_param=one_init)

                control_inputs = []

                if self.is_training:

                    mean, variance = tf.nn.moments(input, axis)

                    if input_type != 'Z2':

                        if input_type == 'C4':
                            num_repeat = 4
                        else:
                            num_repeat = 8

                        mean = tf.reshape(mean, [-1, num_repeat])
                        mean = tf.reduce_mean(mean, 1, keep_dims=False)
                        mean = tf.reshape(tf.tile(tf.expand_dims(mean, -1), [1, num_repeat]), [-1])
                        variance = tf.reshape(variance, [-1, num_repeat])
                        variance = tf.reduce_mean(variance, 1, keep_dims=False)
                        variance = tf.reshape(tf.tile(tf.expand_dims(variance, -1), [1, num_repeat]), [-1])

                    # update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                    #                                                            mean, BN_DECAY)
                    # update_moving_variance = moving_averages.assign_moving_average(
                    #     moving_variance, variance, BN_DECAY)
                    # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
                    # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

                    train_mean = tf.assign(moving_mean,
                                           moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
                    train_var = tf.assign(moving_variance,
                                          moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

                    with tf.control_dependencies([train_mean, train_var]):
                        return tf.nn.batch_normalization(input,
                                                         mean, variance, beta, gamma, BN_EPSILON)
                else:
                    return tf.nn.batch_normalization(input, moving_mean, moving_variance, beta, gamma, BN_EPSILON)

    def g_batch_norm_tensorflow_no_decorator(self, input, name, input_type):

        with tf.variable_scope(name):
            with tf.variable_scope('BatchNorm') as scope:
                x_shape = input.get_shape()
                params_shape = x_shape[-1:]

                axis = list(range(len(x_shape) - 1))

                one_init = tf.constant_initializer(value=1.0)
                zero_init = tf.constant_initializer(value=0.0)

                beta = self.make_var('beta', params_shape, initializer_param=zero_init)
                gamma = self.make_var('gamma', params_shape, initializer_param=one_init)
                moving_mean = self.make_var('moving_mean', params_shape, trainable_param=False,
                                            initializer_param=zero_init)
                moving_variance = self.make_var('moving_variance', params_shape, trainable_param=False,
                                                initializer_param=one_init)

                control_inputs = []

                if self.is_training:

                    mean, variance = tf.nn.moments(input, axis)

                    if input_type != 'Z2':

                        if input_type == 'C4':
                            num_repeat = 4
                        else:
                            num_repeat = 8

                        mean = tf.reshape(mean, [-1, num_repeat])
                        mean = tf.reduce_mean(mean, 1, keep_dims=False)
                        mean = tf.reshape(tf.tile(tf.expand_dims(mean, -1), [1, num_repeat]), [-1])
                        variance = tf.reshape(variance, [-1, num_repeat])
                        variance = tf.reduce_mean(variance, 1, keep_dims=False)
                        variance = tf.reshape(tf.tile(tf.expand_dims(variance, -1), [1, num_repeat]), [-1])

                    # update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                    #                                                            mean, BN_DECAY)
                    # update_moving_variance = moving_averages.assign_moving_average(
                    #     moving_variance, variance, BN_DECAY)
                    # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
                    # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

                    train_mean = tf.assign(moving_mean,
                                           moving_mean * BN_DECAY + mean * (1 - BN_DECAY))
                    train_var = tf.assign(moving_variance,
                                          moving_variance * BN_DECAY + variance * (1 - BN_DECAY))

                    with tf.control_dependencies([train_mean, train_var]):
                        return tf.nn.batch_normalization(input,
                                                         mean, variance, beta, gamma, BN_EPSILON)
                else:
                    return tf.nn.batch_normalization(input, moving_mean, moving_variance, beta, gamma, BN_EPSILON)


    @layer
    def g_residual_block(self, input, name, use_dropout=False, padding='SAME'):
        c_o = np.int32(input.get_shape()[-1] // 4)
        kernel_size = 3
        s_h = 1
        s_w = 1

        with tf.variable_scope(name) as scope:
            #output = GBatchNorm('C4')(input,training=self.is_training)
            output = self.g_batch_norm_tensorflow_no_decorator(input, name='bn1', input_type='C4')

            #output = GConv2D(c_o, (kernel_size, kernel_size), kernel_initializer='he_normal', padding=padding, strides=(s_h, s_w), use_bias=False, h_input='C4', h_output='C4', name=name+'c1')(output)
            output = self.g_conv_no_decorator(output, 'C4', 'C4', kernel_size, c_o, 1, 1, padding='SAME', name=name+'c1')

            #output = GBatchNorm('C4')(output,training=self.is_training)
            output = self.g_batch_norm_tensorflow_no_decorator(output, name='bn2', input_type='C4')

            output = tf.nn.relu(output, 'rl1')

            if use_dropout:
                keep_prob = 0.8
                if not self.is_training:
                    keep_prob = 1.0
                output = tf.nn.dropout(output, keep_prob, name='dr1')

            #output = GConv2D(c_o, (kernel_size, kernel_size), kernel_initializer='he_normal', padding=padding, strides=(s_h, s_w), use_bias=False, h_input='C4', h_output='C4', name=name+'c2')(output)
            output = self.g_conv_no_decorator(output, 'C4', 'C4', kernel_size, c_o, 1, 1, padding='SAME',
                                              name=name + 'c2')

            #output = GBatchNorm('C4')(output,training=self.is_training)
            output = self.g_batch_norm_tensorflow_no_decorator(output, name='bn3', input_type='C4')

            output = tf.add(output, input)

            return output


    @layer
    def g_concat(self, inputs, name):
        return tf.concat(axis=-1, values=inputs, name=name)
        #return tf.add(inputs[0], inputs[1], name=name)

    @layer
    def g_avg_global_pool(self, input, name):
        with tf.variable_scope(name) as scope:
            output = GroupPool('C4')(input)
            return output



