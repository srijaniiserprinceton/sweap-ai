import sys, importlib
import numpy as np

from sweapai.src import functions as fn
from sweapai.src import misc_functions as misc_fn
from sweapai.src import span_functions

if __name__=='__main__':
    # importing the config file provided at command line
    config_file = sys.argv[1]

    # loading the config file
    config = importlib.import_module(config_file).config

    # loading the credentials for pyspedas
    creds  = misc_fn.credential_reader(config['global']['CREDS_PATH'])

    # loading the PSP data for the given TRANGE with optional clipping
    span_L2 = fn.load_span_L2(config['global']['TRANGE'], 
                              CREDENTIALS=None, 
                              CLIP=config['global']['CLIP'])

    span_L3 = fn.load_span_L3(config['global']['TRANGE'],
                              CREDENTIALS=None,
                              CLIP=config['global']['CLIP'])

    # getting the magnetic field unit vector from L3 data
    bvec = span_L3.MAGF_INST.data / np.linalg.norm(span_L3.MAGF_INST.data, axis=1)[:,np.newaxis]

    # making a dummy biMax dictionary for testing
    biMax = {}
    biMax['v_core'] = np.array([400, 0, 0])

    # obtaining the span grids in Cartesian coordiantes
    spangrids = span_functions.SPANpolar_to_SPANcartesian(span_L2)

    # choosing a time index for testing
    tidx = 0

    # testing SPAN to FA grid conversion (rotation)
    vpara_r, vperp_r = span_functions.rotate_SPANgrids_to_FAgrids(biMax, spangrids[tidx], bvec[tidx])

    # testing SPAN to FA grid conversion (projection)
    vpara_p, vperp_p = span_functions.project_SPANgrids_to_FAgrids(biMax, spangrids[tidx], bvec[tidx])