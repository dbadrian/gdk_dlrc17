import glob
import time
import numpy
import scipy.misc
import os

import tensorflow as tf
from tensorflow.contrib.framework import list_variables

import Constants
import Measures
from Log import log
from Network import Network
from Trainer import Trainer
from Utilf import load_wider_or_deeper_mxnet_model
from datasets.Forward import forward, oneshot_forward, online_forward
from datasets.Loader import load_dataset
from Forwarding.CMC_Validator import do_cmc_validation, view_latent_space
from tensorflow.contrib import slim
from datasets.Util.Util import username


class Engine(object):
  def __init__(self, config,latent = False,notlatent=False,small_net=False):
    self.config = config
    self.dataset = config.unicode("dataset").lower()
    self.load_init = config.unicode("load_init", "")
    self.load = config.unicode("load", "")
    self.task = config.unicode("task", "train")
    self.use_partialflow = config.bool("use_partialflow", False)
    self.do_oneshot_or_online_or_offline = self.task in ("oneshot_forward", "oneshot", "online", "offline")
    if self.do_oneshot_or_online_or_offline:
      assert config.int("batch_size_eval", 1) == 1
    self.need_train = self.task == "train" or self.do_oneshot_or_online_or_offline or self.task == "forward_train"

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    self.session = tf.InteractiveSession(config=sess_config)

    # self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    self.coordinator = tf.train.Coordinator()
    self.valid_data = load_dataset(config, "valid", self.session, self.coordinator)
    if self.need_train:
      self.train_data = load_dataset(config, "train", self.session, self.coordinator)

    self.num_epochs = config.int("num_epochs", 1000)
    self.model = config.unicode("model")
    self.model_base_dir = config.dir("model_dir", "models")
    self.model_dir = self.model_base_dir + self.model + "/"
    self.save = config.bool("save", True)
    if latent:
      with tf.variable_scope('latent'):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
    else:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.start_epoch = 0
    reuse_variables = None
    if self.need_train:
      freeze_batchnorm = config.bool("freeze_batchnorm", False)
      self.train_network = Network(config, self.train_data, self.global_step, training=True,
                                   use_partialflow=self.use_partialflow,
                                   do_oneshot=self.do_oneshot_or_online_or_offline,
                                   freeze_batchnorm=freeze_batchnorm, name="trainnet")
      reuse_variables = True
    else:
      self.train_network = None
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      print(tf.get_variable_scope())
      self.test_network = Network(config, self.valid_data, self.global_step, training=False,
                                  do_oneshot=self.do_oneshot_or_online_or_offline, use_partialflow=False,
                                  freeze_batchnorm=True, name="testnet",latent=latent)
    print ("number of parameters:", "{:,}".format(self.test_network.n_params))
    self.trainer = Trainer(config, self.train_network, self.test_network, self.global_step, self.session)
    max_saves_to_keep = config.int("max_saves_to_keep", 0)
    self.max_to_keep = max_saves_to_keep
    self.saver = tf.train.Saver(max_to_keep=max_saves_to_keep, pad_step_number=True)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.train.start_queue_runners(self.session)
    self.load_init_saver = self._create_load_init_saver()
    # vars = slim.get_variables()
    # for v in vars:
    #   print(v.name)
    if not self.do_oneshot_or_online_or_offline:
      self.try_load_weights(latent=latent, notlatent=notlatent,small_net=small_net)
    #put this in again later
    #self.session.graph.finalize()

  def _create_load_init_saver(self):
    if self.load_init != "" and not self.load_init.endswith(".pickle"):
      vars_file = [x[0] for x in list_variables(self.load_init)]
      vars_model = tf.global_variables()
      assert all([x.name.endswith(":0") for x in vars_model])
      vars_intersection = [x for x in vars_model if x.name[:-2] in vars_file]
      vars_missing = [x for x in vars_model if x.name[:-2] not in vars_file]
      if len(vars_missing) > 0:
        print("the following variables will not be initialized since they are not present in the " \
                         "initialization model", [v.name for v in vars_missing])
      return tf.train.Saver(var_list=vars_intersection)
    else:
      return None

  def try_load_weights(self, latent=False, notlatent=False,small_net=False ):
    fn = None
    if self.load != "":
      fn = self.load.replace(".index", "")
    else:
      files = sorted(glob.glob(self.model_dir + self.model + "-*.index"))
      if len(files) > 0:
        fn = files[-1].replace(".index", "")

    small_net = self.config.int('small_net',0)

    if fn is not None:
      print ("loading model from", fn)
      vars = slim.get_variables()
      if small_net:
        varlist = [var for var in vars if var.name.split('/')[0] == 'conv0']+[
          var for var in vars if var.name.split('/')[0] == 'res0']+[
          var for var in vars if var.name.split('/')[0] == 'res1']+[
          var for var in vars if var.name.split('/')[0] == 'res2']#+['global_step']# +[
          #var for var in vars if var.name.split('/')[0] == 'output']
        saver = tf.train.Saver(max_to_keep=self.max_to_keep, pad_step_number=True, var_list=varlist)
        saver.restore(self.session, fn)
        # varlist = [var for var in vars if var.name.split('/')[0] == 'conv0'] + [
        #            var for var in vars if var.name.split('/')[0] == 'res0'] + [
        #            var for var in vars if var.name.split('/')[0] == 'res1'] + [
        #            var for var in vars if var.name.split('/')[0] == 'res2'] + [
        #            var for var in vars if var.name.split('/')[0] == 'output']
        # self.saver = tf.train.Saver(max_to_keep=self.max_to_keep, pad_step_number=True, var_list=varlist)
      if latent:
        varlist = [var for var in vars if var.name.split('/')[0] == 'latent']
        self.saver =  tf.train.Saver(max_to_keep=self.max_to_keep, pad_step_number=True,var_list=varlist)
        self.saver.restore(self.session, fn)
      if notlatent:
        varlist = [var for var in vars if var.name.split('/')[0]!='latent']
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep, pad_step_number=True,var_list=varlist)
        self.saver.restore(self.session, fn)
      if not(latent or notlatent or small_net):
        self.saver.restore(self.session, fn)
      if self.model == fn.split("/")[-2]:
        self.start_epoch = int(fn.split("-")[-1])
        print ( "starting from epoch", self.start_epoch + 1)
    elif self.load_init != "":
      if self.load_init.endswith(".pickle"):
        print ( "trying to initialize model from wider-or-deeper mxnet model", self.load_init)
        load_wider_or_deeper_mxnet_model(self.load_init, self.session)
      else:
        fn = self.load_init
        print ( "initializing model from", fn)
        assert self.load_init_saver is not None
        self.load_init_saver.restore(self.session, fn)

  def reset_optimizer(self):
    self.trainer.reset_optimizer()

  @staticmethod
  def run_epoch(step_fn, data, epoch):
    loss_total = 0.0
    n_imgs_per_epoch = data.num_examples_per_epoch()
    measures_accumulated = {}
    n_imgs_processed = 0
    while n_imgs_processed < n_imgs_per_epoch:
      start = time.time()
      loss_summed, measures, n_imgs = step_fn(epoch)
      loss_total += loss_summed

      measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measures)

      n_imgs_processed += n_imgs

      loss_avg = loss_summed / n_imgs
      measures_avg = Measures.calc_measures_avg(measures, n_imgs, data.ignore_classes)
      end = time.time()
      elapsed = end - start

      #TODO: Print proper averages for the measures
      print ( n_imgs_processed, '/', n_imgs_per_epoch, loss_avg, measures_avg, "elapsed", elapsed)
    loss_total /= n_imgs_processed
    measures_accumulated = Measures.calc_measures_avg(measures_accumulated, n_imgs_processed, data.ignore_classes)
    return loss_total, measures_accumulated

  def train(self):
    assert self.need_train
    print ("starting training")
    for epoch in range(self.start_epoch, self.num_epochs):
      start = time.time()
      train_loss, train_measures = self.run_epoch(self.trainer.train_step, self.train_data, epoch)
      # train_loss = 0
      # train_measures = {}

      # valid_loss, valid_measures = self.run_epoch(self.trainer.validation_step, self.valid_data, epoch)
      # valid_loss, valid_measures = do_cmc_validation(self, self.test_network, self.valid_data)
      # valid_loss, valid_measures = view_latent_space(self, self.test_network, self.valid_data,epoch)
      valid_loss = 0
      valid_measures = {}

      end = time.time()
      elapsed = end - start
      train_error_string = Measures.get_error_string(train_measures, "train")
      valid_error_string = Measures.get_error_string(valid_measures, "valid")
      print (log.v1, "epoch", epoch + 1, "finished. elapsed:", "%.5f" % elapsed, "train_score:", "%.5f" % train_loss,\
          train_error_string, "valid_score:", valid_loss, valid_error_string,file=log.v1)
      if self.save:
        self.save_model(epoch + 1)

  def eval(self):
    start = time.time()
    # valid_loss, measures = self.run_epoch(self.trainer.validation_step, self.valid_data, 0)
    # valid_loss, measures = view_latent_space(self, self.test_network, self.valid_data, 0)
    valid_loss = 0
    measures = {}
    end = time.time()
    elapsed = end - start
    valid_error_string = Measures.get_error_string(measures, "valid")
    print ("eval finished. elapsed:", elapsed, "valid_score:", valid_loss, valid_error_string)

  def run(self):
    # if self.task == "segment":
    #   self.run_segment()
    if self.task == "train":
      self.train()
    elif self.task == "eval":
      self.eval()
    elif self.task in ("forward", "forward_train"):
      if self.task == "forward_train":
        network = self.train_network
        data = self.train_data
      else:
        network = self.test_network
        data = self.valid_data
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", True)
      forward(self, network, data, self.dataset, save_results=save_results, save_logits=save_logits)
    elif self.do_oneshot_or_online_or_offline:
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", False)
      if self.task == "oneshot":
        oneshot_forward(self, save_results=save_results, save_logits=save_logits)
      elif self.task == "online":
        online_forward(self, save_results=save_results, save_logits=save_logits)
      else:
        assert False, "Unknown task " + str(self.task)
    else:
      assert False, "Unknown task " + str(self.task)

  def save_model(self, epoch):
    tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, self.model_dir + self.model, epoch)

  # def run_segment(self):
  #   in_fn = "/home/dlrc/Documents/Segment/Jono/testdata/000000.jpg"
  #   image = scipy.misc.imread(in_fn)
  #   size = (int(image.shape[1]), int(image.shape[0]))
  #   mask, prob = self.segment(image,size)
  #
  #   out_folder = '/home/dlrc/Documents/Segment/Jono/outdata/'
  #   out_fn1 = out_folder + 'mask.png'
  #   out_fn2 = out_folder + 'prob.png'
  #   out_fn3 = out_folder + 'orig.png'
  #   scipy.misc.imsave(out_fn3, image)
  #   scipy.misc.imsave(out_fn1, mask)
  #   scipy.misc.imsave(out_fn2, prob)

  def segment(self,image,size,save_flag=False,threshold = 0.5):

      start = time.time()
      orig_size = (int(image.shape[1]), int(image.shape[0]))
      resized_image = scipy.misc.imresize(image, size)

      # hax_fn = "/home/"+username()+"/dlrc17-gdk/ext/OnAVOS/custom_dataset/JPEGImages/480p/live/000000.jpg"
      hax_fn = "/home/" + username() + "/Documents/Segment/OnAVOSold/custom_dataset/JPEGImages/480p/live/000000.jpg"
      scipy.misc.imsave(hax_fn, resized_image)

      tensor_out = self.test_network.y_softmax
      tensor_out_argmax = tf.argmax(tensor_out, 3)

      feed_dict = {}
      # feed_dict = {self.valid_data.img_placeholder: [image]}
      # feed_dict = {self.test_network.img: [image]}

      prob,y_argmax = self.session.run([tensor_out,tensor_out_argmax],feed_dict=feed_dict)
      prob = prob[0,:,:,1]
      # print(prob.max(),prob.min(),prob.mean())
      orig_size_prob = scipy.misc.imresize(prob, orig_size,interp='bicubic')
      # print(orig_size_prob.max(), orig_size_prob.min(), orig_size_prob.mean())
      mask = ((orig_size_prob>255*threshold)*255).astype("uint8")

      # mask = (y_argmax * 255).astype("uint8")
      # mask = numpy.squeeze(mask, axis=0)
      # prob = numpy.squeeze(prob[:, :, :, 1], axis=0)

      # mask = numpy.fliplr(mask)
      # prob = numpy.fliplr(prob)

      mask = scipy.misc.imresize(mask,size)
      prob = scipy.misc.imresize(prob,size)

      # if save_flag:
      #   saver_fol = "/home/dlrc/dlrc17-gdk/gdk/imagproc/video_output/"
      #   # fol_num = str(len(os.listdir(saver_fol))).zfill(4)
      #   # if not os.path.exists(fol_num):
      #   #     os.makedirs(fol_num)
      #   # dir = saver_fol+fol_num+"/"
      #   dir = saver_fol
      #   im_num = str(len(os.listdir(dir))).zfill(4)
      #   scipy.misc.imsave(dir+"/"+im_num+".jpg", image)
      #   scipy.misc.imsave(dir + "/" + im_num + ".png", mask)

      end = time.time()
      elapsed = end - start
      return mask,prob

  def get_latent(self, image):

    start = time.time()

    hax_fn = "/home/"+username()+"/dlrc17-gdk/ext/OnAVOS/custom_dataset/JPEGImages/480p/live_fol/00001_0001.png"
    # hax_fn = "/home/" + username() + "/Documents/class_data_final/live_fol/00001_0001.png"
    scipy.misc.imsave(hax_fn, image)

    out_layer_name = self.config.unicode("output_embedding_layer", "outputTriplet")
    out_layer = self.test_network.tower_layers[0][out_layer_name]
    assert len(out_layer.outputs) == 1
    out_feature = out_layer.outputs[0]
    out_feature_size = out_layer.n_features

    path_name = "live_fol/"
    path = self.valid_data.test_case
    idx_placeholder = self.valid_data.idx_placeholder
    idx_value = [0, 1, 0, 0]

    feature_val =self.session.run(out_feature,feed_dict={idx_placeholder: idx_value, path: path_name})

    end = time.time()
    elapsed = end - start
    # print(elapsed)
    return feature_val

  def get_latent_batch(self, images):

    start = time.time()

    for i,image in enumerate(images):
      hax_fn = "/home/"+username()+"/dlrc17-gdk/ext/OnAVOS/custom_dataset/JPEGImages/480p/live_fol/00001_"+str(i+1).zfill(4)+".png"
      # hax_fn = "/home/" + username() + "/Documents/class_data_final/live_fol/00001_" + str(i).zfill(4) + ".png"
      scipy.misc.imsave(hax_fn, image)

    out_layer_name = self.config.unicode("output_embedding_layer", "outputTriplet")
    out_layer = self.test_network.tower_layers[0][out_layer_name]
    assert len(out_layer.outputs) == 1
    out_feature = out_layer.outputs[0]
    out_feature_size = out_layer.n_features

    path_name = "live_fol/"
    path = self.valid_data.test_case
    idx_placeholder = self.valid_data.idx_placeholder
    num_images = len(images)
    idx_value = [0, num_images, 0, 0]

    feature_val =self.session.run(out_feature,feed_dict={idx_placeholder: idx_value, path: path_name})

    end = time.time()
    elapsed = end - start
    # print(elapsed)
    return feature_val
