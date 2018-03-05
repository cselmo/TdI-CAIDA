import os
import sys

##Replace the standard out
sys.stdout = sys.stderr

##Add this file path to sys.path in order to import settings
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

##Add this file path to sys.path in order to import app
sys.path.append('/home/cselmo/miniconda3/envs/as_detect/')

##Create appilcation for our app
from run import app as application

