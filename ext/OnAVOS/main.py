#!/usr/bin/env python
import sys
import os
from datasets.Util.Util import username
# base_file=os.getcwd()
base_file = '/home/'+username()+'/dlrc17-gdk/ext/OnAVOS'
os.chdir(base_file)
sys.path.insert(0,'./')
sys.path.insert(0,'datasets')


from Engine import Engine
from Config import Config
from Log import log
import tensorflow as tf


def init_log(config):
  log_dir = config.dir("log_dir", "logs")
  model = config.unicode("model")
  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])


def main(_):
  assert len(sys.argv) == 2, "usage: main.py <config>"
  config_path = sys.argv[1]
  assert os.path.exists(config_path), config_path
  try:
    config = Config(config_path)
  except ValueError as e:
    print ("Malformed config file:", e)
    return -1
  init_log(config)
  config.initialize()
  #dump the config into the log
  #print >> log.v4, open(config_path).read()
  print (open(config_path).read())
  engine = Engine(config)
  engine.run()

if __name__ == '__main__':
  tf.app.run(main)
