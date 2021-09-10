import logging
import sys

from wfm.constants import *
from wfm import preprocessing
from wfm.eda import eda, profile
from wfm.model import model_and_explanation, group_model_and_explanation

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
