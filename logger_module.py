import logging
import logging.config

logging.config.fileConfig("conf/logger.conf")

DEFAULT_LOGGER = 'logger01'


def _get_logger(name):
    return logging.getLogger(name)


def get_default_logger():
    return _get_logger(DEFAULT_LOGGER)
