import logging
import os


def get_my_logger(model_name):
    logger = logging.getLogger(model_name+"_logger")
    if not os.path.exists("log"):
        os.mkdir("log")
    file_handler = logging.FileHandler(os.path.join("log", model_name+".log"))
    stdout_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    return logger
