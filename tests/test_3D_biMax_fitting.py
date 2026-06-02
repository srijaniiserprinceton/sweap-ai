import sys, importlib
import numpy as np

from sweapai.src import functions as fn
from sweapai.src import misc_functions as misc_fn
from sweapai.src import span_functions
from sweapai.bimax_fitter_3D import fit_bimax

from sweapai.plotter import compare_fac_to_fit as plotfuncs
from sweapai.src import functions as funcs

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
    bhat = span_L3.MAGF_INST.data / np.linalg.norm(span_L3.MAGF_INST.data, axis=1)[:,np.newaxis]

    # making a dummy biMax dictionary for testing
    biMax = {}

    # obtaining the span grids in Cartesian coordiantes
    spangrids = span_functions.SPANpolar_to_SPANcartesian(span_L2)

    # choosing a time index for testing
    tidx = 18999

    # slicing out the velocity grid for that time
    vgrid = spangrids[tidx]

    # slicing out the bhat for the time index
    bhat = bhat[tidx]

    # fitting a drifting bi-Maxwellian
    vdfdata = span_L2.vdf.data[tidx]

    # filtering out below the count threshold
    count_threshold = config['span']['COUNT_THRESHOLD']
    countmask = span_L2.counts.data[tidx] > count_threshold
    vdfdata = vdfdata[countmask]
    vgrid = vgrid[:, countmask]

    # performing the core fitting only
    fit_coreonly, vdfdata_max = fit_bimax.core_fitting_only(tidx, vgrid, vdfdata, bhat, span_L3)

    # plotting the fit comparison
    plotfuncs.plot_comparison(vdfdata, vdfdata_max, vgrid, bhat, fit_coreonly, component='coreonly')

    # computing the FAC coordinates for the 
    logA_fit, Ux_fit, Uy_fit, Uz_fit, log_vth_para_fit, log_vth_perp_fit = fit_coreonly.x
    vpara, vperp = span_functions.project_SPANgrids_to_FAgrids(
                    biMax={'v_core': np.array([Ux_fit, Uy_fit, Uz_fit])},
                    spangrids=vgrid,
                    bvec=bhat
                    )

    # performing the core fitting only
    fit_coreandbeam, vdfdata_max = fit_bimax.core_and_beam_fitting(tidx, vgrid, vpara,
                                                                   vdfdata, bhat, fit_coreonly)

    # plotting the fit comparison
    plotfuncs.plot_comparison(vdfdata, vdfdata_max, vgrid, bhat, fit_coreandbeam, component='coreandbeam')
    funcs.print_coreandbeam_fit(fit_coreandbeam)

