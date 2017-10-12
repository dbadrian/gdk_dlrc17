__author__ = "Karoline Stosio"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Karoline Stosio, Lucia Seitz"]
__license__ = "MIT"
__maintainer__ = "Karoline Stosio"

import glob
import sys
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
from random import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.manifold import TSNE
import scipy as sc
import edward as ed
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import Progbar
from scipy.misc import imsave
from tensorflow.contrib import slim
from skimage import exposure


class Engine(object):
    def __init__(self):
        self.M = 1  # batch size during testing
        self.d = 128  # latent dimension
        self.CHECKPOINT = '/home/dlrc/Documents/Ach7/VAE_checkpoints/vae_854.ckpt'
        # MODEL
        self.z = Normal(loc=tf.zeros([self.M, self.d]), scale=tf.ones([self.M, self.d]))
        self.loc1, self.scale1 = self.generative_network(self.z)
        self.x = Normal(loc=self.loc1, scale=self.scale1)

        # INFERENCE
        self.x_ph = tf.placeholder(tf.float32, [None, 64 * 64 * 3])
        self.loc, self.scale = self.inference_network(tf.cast(self.x_ph, tf.float32))
        self.qz = Normal(loc=self.loc, scale=self.scale)

        # Read in
        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())

        variables = slim.get_model_variables()
        # for v in variables:
        # print(v.name)
        good_vars = [v for v in variables if
                     v.name.split('/')[0] == 'inference' or v.name.split('/')[0] == 'generative']
        # for v in good_vars:
        # print(v.name)
        restorer = tf.train.Saver(var_list=good_vars)
        generated_image = []
        sample_from_latent_space = []
        self.sess = tf.Session()
        self.sess.run(init_op)
        print("Restoring session!")
        restorer.restore(self.sess, self.CHECKPOINT)
        print("Session restored!")

    def read_im(self, img, preproc=True):
        if preproc:
            return self.im_preproc(img)
        else:
            return img.flatten()

    def im_preproc(self, im):
        variance = 0.03
        X = im.reshape((64, 64, 3))
        X = X.astype(float) / 255
        X[:, :, 0] = (X[:, :, 0] - variance) / (1 + 2 * variance)
        X[:, :, 1] = (X[:, :, 1] - variance) / (1 + 2 * variance)
        X[:, :, 2] = (X[:, :, 2] - variance) / (1 + 2 * variance)
        X = exposure.equalize_hist(X)
        return X.flatten()

    def generative_network(self, z, reuse=False):
        """Generative network to parameterize generative model. It takes
        latent variables as input and outputs the likelihood parameters.
        logits = neural_network(z)
        """
        with tf.variable_scope('generative') as sc:
            with slim.arg_scope([slim.conv2d_transpose],
                                activation_fn=tf.nn.elu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'scale': True}):
                if reuse:
                    sc.reuse_variables()
                print('generative')
                print(z)
                net = tf.reshape(z, [self.M, 1, 1, self.d])
                print(net)
                net = slim.conv2d_transpose(net, 128, 4, stride=4, scope='t_conv1')
                print(net)
                net = slim.conv2d_transpose(net, 64, 5, stride=2, scope='t_conv2')
                print(net)
                net = slim.conv2d_transpose(net, 32, 5, stride=2, scope='t_conv3')
                print(net)
                net = slim.conv2d_transpose(net, 16, 5, stride=2, scope='t_conv4')
                print(net)
                net = slim.conv2d_transpose(net, 3, 5, stride=2, activation_fn=None, scope='t_conv5')
                print(net)
                # net1 = slim.flatten(net[:,:,:,:3])
                # net2 = slim.flatten(net[:,:,:,3:])
                # net = slim.flatten(net)
                print(net)
                # print(net1)
                # print(net2)net
                loc = tf.nn.sigmoid(slim.flatten(net))
                scale = tf.ones_like(loc) * tf.nn.softplus(tf.Variable(0.001))
                print('generative part completed')

        # return net1, net2
        return loc, scale

    def inference_network(self, x, reuse=False):
        """Inference network to parameterize variational model. It takes
        data as input and outputs the variational parameters.
        loc, scale = neural_network(x)
        """
        with tf.variable_scope('inference') as sc:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=tf.nn.elu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'scale': True}):
                if reuse:
                    sc.reuse_variables()
                print('inference')
                print(self.x)
                # net = tf.reshape(x, [M, 28, 28, 1])
                net = tf.reshape(self.x, [self.M, 64, 64, 3])
                print(net)
                net = slim.conv2d(net, 8, 5, stride=2, scope='conv1')
                print(net)
                net = slim.conv2d(net, 16, 5, stride=3, scope='conv2')
                print(net)
                net = slim.conv2d(net, 32, 5, stride=3, scope='conv3')
                print(net)
                net = slim.conv2d(net, 64, 5, stride=2, scope='conv4')
                print(net)
                net = slim.dropout(net, 0.9)
                print(net)
                net = slim.flatten(net)
                print(net)
                params = slim.fully_connected(net, self.d * 2, activation_fn=None, scope='fc1')
                print('params', net)

            loc = params[:, :self.d]
            scale = tf.nn.softplus(params[:, self.d:])
        print('loc', loc)
        print('scale', scale)
        print('inference part completed')
        return loc, scale

    def get_latent(self, img):
        # init_op = tf.group(
        #     tf.local_variables_initializer(),
        #     tf.global_variables_initializer())
        #
        # variables = slim.get_model_variables()
        # # for v in variables:
        #     # print(v.name)
        # good_vars = [v for v in variables if
        #              v.name.split('/')[0] == 'inference' or v.name.split('/')[0] == 'generative']
        # # for v in good_vars:
        #     # print(v.name)
        # restorer = tf.train.Saver(var_list=good_vars)
        # generated_image = []
        # sample_from_latent_space = []
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #     print("Restoring session!")
        #     restorer.restore(sess, self.CHECKPOINT)
        #     print("Session restored!")
        #     input_image = self.read_im(img)[None, :]
        #     print("Image read!")
        #     generated_image, sample_from_latent_space = sess.run([self.loc1, self.loc],
        #                                                          feed_dict={self.x_ph: input_image})
        # # return generated_image, sample_from_latent_space


        input_image = self.read_im(img)[None, :]
        generated_image, sample_from_latent_space = self.sess.run([self.loc1, self.loc],
                                                             feed_dict={self.x_ph: input_image})
        return sample_from_latent_space