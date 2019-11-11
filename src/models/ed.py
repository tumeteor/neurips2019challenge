import os
import tensorflow as tf
from src.data_utils.loader import TARGET_SHAPE
from src.layers.BasicConvLSTMCell import BasicConvLSTMCell as clstm
from src.layers.MIMBlock import MIMBlock as mimblock
from src.layers.MIMN import MIMN as mimn
import math

from src.models.attn import CrossFrameAttention


class Model:
    def __init__(self, configs, train_init_op, itr):
        self.configs = configs
        # inputs

        self.x, x_rev = itr.get_next()
        grads = []
        loss_train = []
        self.pred_seq = []

        for i in range(self.configs.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                                 reuse=True if i > 0 else None):
                    output_list = self.forward(self.x[i],
                                               num_layers=3,
                                               num_hidden=[32, 32, 32],
                                               filter_size=5,
                                               stride=1,
                                               total_length=20,
                                               input_length=10,
                                               tln=True,
                                               weekday=None)
                    gen_ims = output_list[0]
                    loss = output_list[1]
                    if len(output_list) > 2:
                        self.debug = output_list[2]
                    else:
                        self.debug = []
                    pred_ims = gen_ims[:, self.configs.input_length - self.configs.total_length:]
                    loss_train.append(loss / self.configs.batch_size)

                    # gradients
                    _grads = tf.gradients(loss, tf.compat.v1.trainable_variables())
                    _grads = [grad if grad is not None else tf.zeros_like(var)
                              for var, grad in zip(tf.compat.v1.trainable_variables(), _grads)]
                    grads.append(_grads)
                    self.pred_seq.append(pred_ims)

        # add losses and gradients together and get training updates
        with tf.device('/gpu:0'):
            for i in range(1, self.configs.n_gpu):
                loss_train[0] += loss_train[i]
                for j in range(len(grads[0])):
                    grads[0][j] += grads[i][j]

        opt = tf.keras.optimizers.Adam(lr=self.configs.lr, decay=self.configs.decay_rate)
        opt.apply_gradients(zip(grads[0], tf.compat.v1.trainable_variables()))

        self.saver = tf.train.Saver(tf.compat.v1.global_variables())
        init = tf.compat.v1.global_variables_initializer()
        configProt = tf.compat.v1.ConfigProto()
        configProt.gpu_options.allow_growth = configs.allow_gpu_growth
        configProt.allow_soft_placement = True
        configProt.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.compat.v1.Session(config=configProt)
        self.sess.run(init)
        self.sess.run(train_init_op)

    def w_initializer(self, dim_in, dim_out):
        random_range = math.sqrt(6.0 / (dim_in + dim_out))
        return tf.random_uniform_initializer(-random_range, random_range)

    def forward(self, images, num_layers, num_hidden, filter_size,
                stride=1, total_length=20, input_length=10, tln=True, weekday=None):
        gen_images = []
        clstm_layer = []
        stlstm_layer_diff = []
        cell_state = []
        hidden_state = []
        cell_state_diff = []
        hidden_state_diff = []
        shape = (images.get_shape().as_list()[0], total_length, TARGET_SHAPE[0], TARGET_SHAPE[1], 3)
        output_channels = shape[-1]

        # make a base image (map):
        input_images, output_images = tf.split(images, [input_length, total_length - input_length], axis=1)
        base_img = tf.reduce_mean(input_images, axis=1)
        base_img = tf.expand_dims(base_img, 1)

        total_length += 1
        input_length += 1
        images = tf.concat((base_img, images), axis=1)

        for i in range(num_layers):
            if i == 0:
                num_hidden_in = num_hidden[num_layers - 1]
            else:
                num_hidden_in = num_hidden[i - 1]
            if i < 1:
                clstm_layer_new = clstm('stlstm_' + str(i + 1),
                                        filter_size,
                                        num_hidden_in,
                                        num_hidden[i],
                                        shape,
                                        tln=tln)
            else:
                clstm_layer_new = mimblock('stlstm_' + str(i + 1),
                                           filter_size,
                                           num_hidden_in,
                                           num_hidden[i],
                                           shape,
                                           tln=tln)
            clstm_layer.append(clstm_layer_new)
            cell_state.append(None)
            hidden_state.append(None)

        for i in range(num_layers - 1):
            clstm_layer_new = mimn('stlstm_diff' + str(i + 1),
                                   filter_size,
                                   num_hidden[i + 1],
                                   shape,
                                   tln=tln)
            stlstm_layer_diff.append(clstm_layer_new)
            cell_state_diff.append(None)
            hidden_state_diff.append(None)

        st_memory = None
        # encoder
        reuse = False
        for time_step in range(input_length -1):
            with tf.compat.v1.variable_scope('predin', reuse=reuse):
                x_gen = images[:, time_step]

                hidden_state[0], cell_state[0], st_memory, o = clstm_layer[0](
                    x_gen, hidden_state[0], cell_state[0], st_memory)
                preh = hidden_state[0]

                for i in range(1, num_layers):
                    if time_step > 0:
                        if i == 1:
                            hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                                hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                        else:
                            hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                                hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        stlstm_layer_diff[i - 1](tf.zeros_like(hidden_state[i - 1]), None, None)
                    preh = hidden_state[i]
                    hidden_state[i], cell_state[i], st_memory, o = clstm_layer[i](
                        hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i], st_memory)
            reuse = True
        x_gen_flat = tf.keras.layers.Flatten()(x_gen)
        time_layer = tf.keras.layers.Dense(64, activation='relu')(x_gen_flat)
        pred_weekday = tf.keras.layers.Dense(1, activation='sigmoid')(time_layer)
        time_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=weekday, logits=pred_weekday)

        # decoder
        reuse = False
        for time_step in range(input_length - 1, total_length - 1):
            with tf.compat.v1.variable_scope('predout', reuse=reuse):

                x_gen = images[:, time_step]

                # resize here
                # x_gen = tf.image.resize_images(
                #     x_gen,
                #     TARGET_SHAPE  # height, width
                # )

                attn_layer = CrossFrameAttention(64)
                attn_c, attention_w = attn_layer(hidden_state[num_layers - 1], o)

                preh = attn_c
                hidden_state[0], cell_state[0], st_memory, o = clstm_layer[0](
                    x_gen, hidden_state[0], cell_state[0], st_memory)
                for i in range(1, num_layers):
                    if time_step > 0:
                        if i == 1:
                            hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                                hidden_state[i - 1] - preh, hidden_state_diff[i - 1], cell_state_diff[i - 1])
                        else:
                            hidden_state_diff[i - 1], cell_state_diff[i - 1] = stlstm_layer_diff[i - 1](
                                hidden_state_diff[i - 2], hidden_state_diff[i - 1], cell_state_diff[i - 1])
                    else:
                        stlstm_layer_diff[i - 1](tf.zeros_like(hidden_state[i - 1]), None, None)
                    preh = hidden_state[i]
                    hidden_state[i], cell_state[i], st_memory, o = clstm_layer[i](
                        hidden_state[i - 1], hidden_state_diff[i - 1], hidden_state[i], cell_state[i], st_memory)

                x_gen = tf.keras.layers.Conv2DTranspose(
                    filters=output_channels,
                    kernel_size=1,
                    strides=stride,
                    padding='same',
                    kernel_initializer=self.w_initializer(num_hidden[num_layers - 1],
                                                          output_channels),
                    name="back_to_pixel")(hidden_state[num_layers - 1])

            gen_images.append(x_gen)
            reuse = True

        gen_images = tf.stack(gen_images, axis=1)

        # apply masking
        # mask = tf.greater(base_img, 0)
        # mask = tf.tile(mask, multiples=[1, 3, 1, 1, 1])

        # masked_gen_images = tf.boolean_mask(gen_images, mask)
        # masked_output_images = tf.boolean_mask(output_images, mask)
        # gen_images = tf.multiply(gen_images, tf.cast(mask, tf.float32))
        loss = tf.nn.l2_loss(gen_images - images[:, -5]) + time_loss

        return [gen_images, loss]

    def save(self, itr):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + self.configs.save_dir)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        # checkpoint_path = os.path.join(checkpoint_path, 'model.ckpt')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))

    def train(self, inputs):
        self.x = inputs
        loss, _, debug = self.sess.run((self.loss_train, self.train_op))
        return loss

    def test(self, inputs):
        self.x = inputs
        gen_ims, debug = self.sess.run(self.pred_seq)
        return gen_ims, debug
