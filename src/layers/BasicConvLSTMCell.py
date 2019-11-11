import math

import tensorflow as tf


class BasicConvLSTMCell:
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, tln=False, initializer=None):

        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.batch = seq_shape[0]
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self.layer_norm = tln
        self._forget_bias = 1.0

        def w_initializer(dim_in, dim_out):
            random_range = math.sqrt(6.0 / (dim_in + dim_out))
            return tf.random_uniform_initializer(-random_range, random_range)

        if initializer is None or initializer == -1:
            self.initializer = w_initializer
        else:
            self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self):
        return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
                        dtype=tf.float32)

    def __call__(self, x, h, c, m):
        for x in (h, c, m):
            x = self.init_state() if x is None else x

        with tf.compat.v1.variable_scope(self.layer_name):
            t_cc = tf.keras.layers.Conv2D(
                self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden * 4),
                name='time_state_to_state')(h)
            s_cc = tf.keras.layers.Conv2D(
                self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(self.num_hidden_in, self.num_hidden * 4),
                name='spatio_state_to_state')(m)
            x_shape_in = x.get_shape().as_list()[-1]
            x_cc = tf.keras.layers.Conv2D(
                self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer(x_shape_in, self.num_hidden * 4),
                name='input_to_state')(x)

            i_s, g_s, f_s, o_s = tf.split(s_cc, 4, 3)
            i_t, g_t, f_t, o_t = tf.split(t_cc, 4, 3)
            i_x, g_x, f_x, o_x = tf.split(x_cc, 4, 3)

            i = tf.nn.sigmoid(i_x + i_t)
            i_ = tf.nn.sigmoid(i_x + i_s)
            g = tf.nn.tanh(g_x + g_t)
            g_ = tf.nn.tanh(g_x + g_s)
            f = tf.nn.sigmoid(f_x + f_t + self._forget_bias)
            f_ = tf.nn.sigmoid(f_x + f_s + self._forget_bias)
            o = tf.nn.sigmoid(o_x + o_t + o_s)
            new_m = f_ * m + i_ * g_
            new_c = f * c + i * g
            cell = tf.concat([new_c, new_m], 3)
            cell = tf.keras.layers.Conv2D(self.num_hidden, 1, 1, padding='same',
                                          kernel_initializer=self.initializer(self.num_hidden * 2, self.num_hidden),
                                          name='cell_reduce')(cell)
            new_h = o * tf.nn.tanh(cell)

            return new_h, new_c, new_m, o
