import logging
import sys

from wfm.constants import *
from wfm import preprocessing
from wfm.eda import eda, profile
from wfm.model import model

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
