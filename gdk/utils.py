__author__ = "David Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian"]
__license__ = "MIT"
__maintainer__ = "David Adrian"

import json
import base64
import time
import logging
import select

try:
    import numpy as np
except:
    pass

from socket import socket, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST, SO_REUSEADDR
import netifaces

import gdk.config as config

logger = logging.getLogger(__name__)


# Network
def find_interface_ip(interface):
    return netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']


def master_announcement(expected_slaves, interface, message_delay=5, repeats=12):
    # Detect IP of specificed Internet Device
    server_ip = find_interface_ip(interface)

    s = socket(AF_INET, SOCK_DGRAM)  # create UDP socket for broadcast
    s.bind((server_ip, 0))
    s.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)  # this is a broadcast socket

    r = socket(AF_INET, SOCK_DGRAM)  # create UDP socket
    r.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    r.bind((server_ip, config.SERVICE_ACK_PORT))

    # Add slaves who give a call back to this set and compare later against expected list of slaves
    ack_slaves = set()

    for repeat in range(repeats):
        data = config.SERVICE_BROADCAST_MAGIC + server_ip + ":" + json.dumps(list(ack_slaves))
        s.sendto(data.encode('utf-8'),
                 ('<broadcast>', config.SERVICE_BROADCAST_PORT))
        logger.debug("Broadcasting own IP+slaves: %s (try=%d)", data, repeat)

        # Check here, to ensure that we send at least once the complete list of slaves
        if ack_slaves == set(expected_slaves):
            logger.debug("Found all expected slaves")
            return True
        else:
            logger.debug("Not all slaves present yet: %s (expected: %s)", ack_slaves, expected_slaves)

        # Wait for messages and timeout eventually
        ready = select.select([r], [], [], message_delay)
        if ready[0]:
            data = r.recv(4096)
            if data.startswith(config.SERVICE_BROADCAST_MAGIC.encode('utf-8')):
                slave = data[len(config.SERVICE_BROADCAST_MAGIC):].decode('utf-8')
                if slave not in ack_slaves:
                    ack_slaves.add(slave) # add potential new slave to set
                    logger.debug("Found new slave: %s", slave)
        else:
            logger.debug("time out for receiving slave acknowledgements")

    logger.debug("Not all slaves registered")
    return False # not all slaves connected in time


def find_master(client_id, ack_repeats=5):
    s = socket(AF_INET, SOCK_DGRAM)  # create UDP socket
    s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    s.bind(('', config.SERVICE_BROADCAST_PORT))

    logger.debug("Waiting for broadcast from master...")
    while True:
        data, addr = s.recvfrom(4096)  # wait for a packet with correct magic 
        if data.startswith(config.SERVICE_BROADCAST_MAGIC.encode('utf-8')):
            master_ip, ack_slaves = data[len(config.SERVICE_BROADCAST_MAGIC):].decode('utf-8').split(":")
            logger.debug("Found master at IP: %s", master_ip)

            if client_id in json.loads(ack_slaves):
                logger.debug("%s: already registered", client_id)
                break

            logger.debug("Sending acknowledgement")
            ack_msg = config.SERVICE_BROADCAST_MAGIC + client_id
            s.sendto(ack_msg.encode('utf-8'), (master_ip, config.SERVICE_ACK_PORT))

    return master_ip


def announce_slave(client_id, master_ip):
    try:
        logger.debug("Sending acknowledgement")
        announce_slave.s.sendto(announce_slave.ack_msg.encode('utf-8'), (master_ip, config.SERVICE_ACK_PORT))
    except AttributeError:
        announce_slave.ack_msg = config.SERVICE_BROADCAST_MAGIC + client_id
        announce_slave.s = socket(AF_INET, SOCK_DGRAM)  # create UDP socket
        announce_slave.s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        announce_slave.s.bind(('', config.SERVICE_BROADCAST_PORT))

        
# Messaging
def encode_cmd(cmd_dict):
    return json.dumps(cmd_dict)

def encode_numpy_array(arr):
    return json.dumps([str(arr.dtype), base64.b64encode(arr).decode('utf-8'), arr.shape])

def decode_numpy_array(arr_string):
    # all credit to https://stackoverflow.com/a/30698135
    # get the encoded json dump
    enc = json.loads(arr_string)
    logger.debug(enc[0])
    logger.debug(enc[2])

    # build the numpy data type
    dataType = np.dtype(enc[0])

    # decode the base64 encoded numpy array data and create a new numpy array with this data & type
    dataArray = np.frombuffer(base64.decodestring(enc[1].encode('utf-8')), dataType)

    # if the array had more than one data set it has to be reshaped
    if len(enc) == 3:
        logger.debug("Reshaping array after decoding")
        return dataArray.reshape(enc[2])   # return the reshaped numpy array containing several data sets

    return dataArray

def decode_list_of_numpy_arrays(array_list):
    return [decode_numpy_array(arr) for arr in array_list]

# Math
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def scale_to_range(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
