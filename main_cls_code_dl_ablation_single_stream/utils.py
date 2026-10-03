from torch import distributed as dist
import logging


def get_logger(logger_name, logging_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logging_file_path, mode='a')
    fmt = '[%(asctime)s]: %(message)s'
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def logging_info(logger, message):
    try:
        if dist.get_rank() == 0:
            logger.info(message)
    except Exception:
        logger.info(message)