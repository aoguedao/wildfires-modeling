import logging
import sys

from .constants import *
from . import preprocessing
from .eda import eda, profile
from . import analysis
from . import train

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
