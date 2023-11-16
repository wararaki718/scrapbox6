import logging
import sys
import time

import numpy as np
from scipy import sparse as sps

logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(name)s-%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    b = sps.csr_matrix(np.random.random((10000, 5000)))
    logger.info(f"b={sys.getrefcount(b)}")
    time.sleep(1)
    c = b
    logger.info(f"b={sys.getrefcount(b)}")
    logger.info(f"c={sys.getrefcount(c)}")
    time.sleep(1)
    d = b.transpose(copy=False)
    logger.info(f"b={sys.getrefcount(b)}")
    logger.info(f"c={sys.getrefcount(c)}")
    logger.info(f"d={sys.getrefcount(d)}")
    time.sleep(1)
    del c
    logger.info(f"b={sys.getrefcount(b)}")
    del b
    logger.info(f"d={sys.getrefcount(d)}")
    time.sleep(1)
    a = sps.csc_matrix(np.random.random((10000, 5000)))
    logger.info(f"d={sys.getrefcount(d)}")
    time.sleep(1)
    e = a.transpose(copy=True)
    logger.info(e.shape)
    time.sleep(1)
    logger.info(a.shape)

    print("DONE")


if __name__ == "__main__":
    main()
