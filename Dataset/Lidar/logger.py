# -*- coding: utf-8 -*-
"""global logger config
"""
import logging
import datetime

class SpecificLogFilter(logging.Filter):
    def filter(self, record):
        # Only log messages containing 'specific' in the message
        return 'timestamp' in record.getMessage()

def logger_config(log_savepath,logging_name):
    '''logger config
    '''
    # get logger name
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)

    # get file handler and set level
    file_handler = logging.FileHandler(log_savepath, encoding='UTF-8')
    file_handler.setLevel(logging.INFO)

    # Add the filter to the handler
    # log_filter = SpecificLogFilter()
    # file_handler.addFilter(log_filter)

    # format the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # console sream handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # add handler for logger objecter
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


# now = datetime.now()
# log_filename = now.strftime("%Y-%m-%d_%H-%M-%S.log")

log_filename = "lidar.log"
log_savepath = f'logs/{log_filename}'
logger = logger_config(log_savepath=log_savepath, logging_name='lidar')
