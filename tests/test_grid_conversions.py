import sys, importlib
import pyspedas

from sweapai.src import functions as fn
from sweapai.src import misc_functions as misc_fn

if __name__=='__main__':
    # importing the config file provided at command line
    config_file = sys.argv[1]

    # loading the config file
    config = importlib.import_module(config_file).config

    # loading the credentials for pyspedas
    creds  = misc_fn.credential_reader(config['global']['CREDS_PATH'])

    # loading the PSP data for the given TRANGE with optional clipping
    span_L2 = fn.load_span_L2(config['global']['TRANGE'], 
                              CREDENTIALS=creds, 
                              CLIP=config['global']['CLIP'])

    span_L3 = fn.load_span_L3(trange,
                              CREDENTIALS=creds,
                              CLIP=config['global']['CLIP'])