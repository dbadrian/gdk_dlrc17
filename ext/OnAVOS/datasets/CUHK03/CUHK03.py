import tensorflow as tf
import numpy
import os

from datasets.Dataset import Dataset
from datasets.Util.Reader import load_image_tensorflow, load_normalized_image_tensorflow
from datasets.Util.Util import smart_shape, username
from datasets.Augmentors import parse_augmentors, apply_augmentors
from datasets.Util.Normalization import normalize

# CUHK03_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/CUHK03/"
CUHK03_DEFAULT_PATH = "/home/" + username() + "/dlrc17-gdk/ext/OnAVOS/custom_dataset/JPEGImages/480p/"
DEFAULT_INPUT_SIZE = [270, 90]
CUHK03_VOID_LABEL = 255

class CUHK03Dataset(Dataset):
  def __init__(self, config, subset, coord):
    super(CUHK03Dataset, self).__init__(subset)
    assert subset in ("train", "valid"), subset
    self.ignore_classes = None
    self.config = config
    self.subset = subset
    self.coord = coord
    self.data_dir = config.unicode("data_dir", CUHK03_DEFAULT_PATH)
    self.model = config.unicode("model","")
    self.train_folder = config.unicode("train_folder","train/")
    self.epoch_length = config.int("epoch_length", 1000)
    self.n_classes = config.int("num_classes",None)
    self.input_size = config.int_list("input_size", DEFAULT_INPUT_SIZE)
    self.input_size = tuple(self.input_size)
    self.batching_mode = config.unicode("batching_mode", "single")
    assert self.batching_mode in ("single","pair","group"), self.batching_mode
    self.validation_mode = config.unicode("validation_mode","embedding")
    assert self.validation_mode in ("embedding", "similarity"), self.validation_mode
    self.group_size = config.int("group_size",4)
    self.pair_ratio = config.float("pair_ratio", 1.0)
    augmentor_strings = self.config.unicode_list("augmentors_train", [])
    self.augmentors = parse_augmentors(augmentor_strings, self.void_label())

    self.train_names = sorted(os.listdir(self.data_dir + self.train_folder))
    train_val = numpy.array([(int(r.split('_')[0]), int(r.split('_')[1].split('.')[0])) for r in self.train_names])
    train_id_list,train_counts = numpy.unique(train_val[:,0],return_counts=True)
    self.train_counts = tf.constant(train_counts.astype(numpy.int32))
    self.num_train_id = train_id_list.shape[0]

    self.num_test_id = 6

    self.idx_placeholder = tf.placeholder(tf.int32, (4,), "idx")
    self.test_case = tf.placeholder(tf.string)
    self.use_end_network = tf.placeholder(tf.bool)

  def num_examples_per_epoch(self):
    return self.epoch_length

  def num_classes(self):
    return self.n_classes

  def create_input_tensors_dict(self, batch_size):

    ########################
    ####### TRAINING #######
    ########################
    if self.subset in "train":

      ####### Paired Batch-Mode #######
      if self.batching_mode in "pair":

        assert batch_size % 2 == 0
        batch_size /= 2

        rand = tf.random_uniform([5], maxval=tf.int32.max, dtype=tf.int32)
        sample_same_person = rand[0] % 2
        pers_id_1 = ((rand[1] - 1) % self.num_train_id) + 1
        pers_1_n_imgs = self.train_counts[pers_id_1-1]
        img_id_1 = ((rand[2] - 1) % pers_1_n_imgs) + 1

        def if_same_person():
          pers_id_2 = pers_id_1
          img_id_2 = ((rand[4] - 1) % (pers_1_n_imgs-1)) + 1
          img_id_2 = tf.cond(img_id_2 >= img_id_1, lambda: img_id_2 + 1, lambda: img_id_2)
          return pers_id_2, img_id_2

        def if_not_same_person():
          pers_id_2 = ((rand[3] - 1) % (self.num_train_id - 1)) + 1
          pers_id_2 = tf.cond(pers_id_2 >= pers_id_1, lambda: pers_id_2 + 1, lambda: pers_id_2)
          pers_2_n_imgs = self.train_counts[pers_id_2-1]
          img_id_2 = ((rand[4] - 1) % pers_2_n_imgs) + 1
          return pers_id_2, img_id_2

        pers_id_2, img_id_2 = tf.cond(tf.cast(sample_same_person, tf.bool), if_same_person, if_not_same_person)

        img1 = tf.as_string(pers_id_1, width=5, fill="0") + "_" + tf.as_string(img_id_1, width=4, fill="0") + ".png"
        img2 = tf.as_string(pers_id_2, width=5, fill="0") + "_" + tf.as_string(img_id_2, width=4, fill="0") + ".png"

        tag = img1 + " " + img2 + " " + tf.as_string(sample_same_person)

        img1 = self.data_dir + self.train_folder +'/'+ img1
        img2 = self.data_dir + self.train_folder +'/'+ img2

        img_val1 = load_image_tensorflow(img1, jpg=False)
        img_val1.set_shape(self.input_size + (3,))
        tensors = {"unnormalized_img": img_val1}
        tensors = apply_augmentors(tensors, self.augmentors)
        img_val1 = tensors["unnormalized_img"]
        img_val1 = normalize(img_val1)

        img_val2 = load_image_tensorflow(img2, jpg=False)
        img_val2.set_shape(self.input_size + (3,))
        tensors = {"unnormalized_img": img_val2}
        tensors = apply_augmentors(tensors, self.augmentors)
        img_val2 = tensors["unnormalized_img"]
        img_val2 = normalize(img_val2)

        pair = tf.stack([img_val1, img_val2])
        label = sample_same_person

        imgs, labels, tags = tf.train.batch([pair, label, tag], batch_size=batch_size)

        shape = smart_shape(imgs)
        shape2 = shape[1:]
        shape2[0] *= batch_size
        imgs = tf.reshape(imgs, shape2)

      ####### Group Batch-Mode #######
      elif self.batching_mode in "group":
        assert batch_size % self.group_size == 0
        batch_size /= self.group_size
        batch_size = int(batch_size)

        pers_ids = tf.random_shuffle(tf.range(1, self.num_train_id))[0:batch_size]

        def for_each_identity(p_idx):
          pers_id = pers_ids[p_idx]
          img_ids = tf.tile(tf.random_shuffle(tf.range(1, self.train_counts[pers_id - 1])), [4])[0:self.group_size]

          def for_each_img(i_idx):
            img_id = img_ids[i_idx]
            tag = tf.as_string(pers_id, width=5, fill="0") + "_" + tf.as_string(img_id, width=4, fill="0") + ".png"

            img = load_image_tensorflow(self.data_dir + self.train_folder +'/'+ tag, jpg=False)
            img.set_shape(self.input_size + (3,))
            tensors = {"unnormalized_img": img}
            tensors = apply_augmentors(tensors, self.augmentors)
            img = tensors["unnormalized_img"]
            img = normalize(img)

            label = p_idx
            img.set_shape(self.input_size + (3,))
            return img, label, tag

          imgs, labels, tags = tf.map_fn(for_each_img, tf.range(0, self.group_size),
                                         dtype=(tf.float32, tf.int32, tf.string))
          return imgs, labels, tags

        imgs, labels, tags = tf.map_fn(for_each_identity, tf.range(0, batch_size),
                                       dtype=(tf.float32, tf.int32, tf.string))

        def reshape(x):
          shape = smart_shape(x)
          shape2 = shape[1:]
          shape2[0] = self.group_size * batch_size
          x = tf.reshape(x, shape2)
          return x

        imgs = reshape(imgs)
        labels = reshape(labels)
        tags = reshape(tags)

      ####### Single Batch-Mode #######
      else: # self.batching_mode in "single":

        rand = tf.random_uniform([2], maxval=tf.int32.max, dtype=tf.int32)
        pers_id_1 = ((rand[0] - 1) % self.num_train_id) + 1
        pers_1_n_imgs = self.train_counts[pers_id_1 - 1]
        img_id_1 = ((rand[1] - 1) % pers_1_n_imgs) + 1

        img1 = tf.as_string(pers_id_1, width=5, fill="0") + "_" + tf.as_string(img_id_1, width=4, fill="0") + ".png"

        tag = img1

        img1 = self.data_dir + self.train_folder + '/' + img1

        img_val1 = load_image_tensorflow(img1, jpg=False)
        img_val1.set_shape(self.input_size + (3,))
        tensors = {"unnormalized_img": img_val1}
        tensors = apply_augmentors(tensors, self.augmentors)
        img_val1 = tensors["unnormalized_img"]
        img_val1 = normalize(img_val1)

        label = pers_id_1

        imgs, labels, tags = tf.train.batch([img_val1, label, tag], batch_size=batch_size)

    ##########################
    ####### Validation #######
    ##########################
    else: # self.subset in "valid":

      ####### Similarity Validation-Mode #######
      if self.validation_mode in "similarity":
        path = self.test_case + '/'
        start_idx = self.idx_placeholder[0]
        end_idx = self.idx_placeholder[1]
        end_net = self.use_end_network

        def if_end_net():
          pdx = self.idx_placeholder[2]

          def _load_imgs(idx):
            img1_idx = pdx + 1
            img2_idx = idx + 1
            label = tf.cond(abs(img1_idx - img2_idx) <= 0, lambda: img1_idx * 0 + 1, lambda: img1_idx * 0)

            img1 = path + tf.as_string(img1_idx, width=4, fill="0") + "_1.png"
            img2 = path + tf.as_string(img2_idx, width=4, fill="0") + "_2.png"
            tag = img1 + " " + img2 + " " + tf.as_string(label)

            img_val1 = tf.zeros(self.input_size + (3,))
            img_val1.set_shape(self.input_size + (3,))
            img_val2 = tf.zeros(self.input_size + (3,))
            img_val2.set_shape(self.input_size + (3,))
            pair = tf.stack([img_val1, img_val2])

            return pair, label, tag

          imgs, labels, tags = tf.map_fn(_load_imgs, tf.range(start_idx, end_idx),
                                         dtype=(tf.float32, tf.int32, tf.string))

          shape = smart_shape(imgs)
          shape2 = shape[1:]
          shape2[0] *= end_idx - start_idx
          imgs = tf.reshape(imgs, shape2)

          return imgs, labels, tags

        def if_not_end_net():
          test_size = self.idx_placeholder[2]
          test_num = self.idx_placeholder[3]

          def _load_imgs(idx):
            label = 0
            img = path + tf.as_string(idx + 1, width=4, fill="0") + "_" + tf.as_string(test_size + test_num) + ".png"

            tag = img
            img = self.data_dir + img

            img_val = load_normalized_image_tensorflow(img, jpg=False)
            img_val.set_shape(self.input_size + (3,))

            return img_val, label, tag

          imgs, labels, tags = tf.map_fn(_load_imgs, tf.range(start_idx, end_idx),
                                         dtype=(tf.float32, tf.int32, tf.string))
          shape = smart_shape(imgs)
          imgs = tf.reshape(imgs, shape)
          return imgs, labels, tags

        imgs, labels, tags = tf.cond(end_net, if_end_net, if_not_end_net)

      ####### Embedding Validation-Mode #######
      else: # self.validation_mode in "embedding":
        path = self.test_case + '/'
        start_idx = self.idx_placeholder[0]
        end_idx = self.idx_placeholder[1]

        test_size = self.idx_placeholder[2]
        test_num = self.idx_placeholder[3]

        def _load_imgs(idx):

          label = 0
          img = path + tf.as_string(test_size+1,width=5,fill="0") + "_" + tf.as_string(idx + 1, width=4, fill="0") + ".png"

          tag = img
          img = self.data_dir + img

          img_val = load_normalized_image_tensorflow(img, jpg=False)
          img_val.set_shape(self.input_size + (3,))

          return img_val, label, tag

        imgs, labels, tags = tf.map_fn(_load_imgs, tf.range(start_idx, end_idx),
                                       dtype=(tf.float32, tf.int32, tf.string))
        shape = smart_shape(imgs)
        imgs = tf.reshape(imgs, shape)

    tensors = {"inputs": imgs, "labels": labels, "tags": tags}

    self.images = imgs
    return tensors

  def void_label(self):
    return CUHK03_VOID_LABEL
