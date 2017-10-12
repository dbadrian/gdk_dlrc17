#!/usr/bin/env python
import sys
import os
base_file =  os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_file)
sys.path.append(base_file)
sys.path.append(os.path.join(base_file, 'datasets'))

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


def create_latent_engine():
  # assert len(sys.argv) == 2, "usage: main.py <config>"
  # config_path = sys.argv[1]
  config_path = "configs/lego_32_run"
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
  engine = Engine(config,latent=True)
  return engine

def create_small_latent_engine():
  # assert len(sys.argv) == 2, "usage: main.py <config>"
  # config_path = sys.argv[1]
  config_path = "configs/lego_32_small_run"
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
  engine = Engine(config,latent=True)
  return engine

def create_small_latent_engine2():
  # assert len(sys.argv) == 2, "usage: main.py <config>"
  # config_path = sys.argv[1]
  config_path = "configs/lego_32_small_run2"
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
  engine = Engine(config,latent=True)
  return engine