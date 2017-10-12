__author__ = "David Adrian"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["David Adrian, Richard Kurle"]
__license__ = "MIT"
__maintainer__ = "David Adrian"

import logging
import logging.config
import os

def setup_logging(
    path='logger.json',
    level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=level)