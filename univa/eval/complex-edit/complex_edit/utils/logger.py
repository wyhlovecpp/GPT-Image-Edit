import os
import os.path as osp
from loguru import logger


def setup_logger(output=None):
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = osp.join(output, "log.txt")
        os.makedirs(osp.dirname(filename), exist_ok=True)
        logger.add(filename, level="INFO")