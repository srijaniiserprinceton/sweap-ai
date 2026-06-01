import sys, importlib
import numpy as np

from sweapai.src import functions as fn
from sweapai.src import misc_functions as misc_fn
from sweapai.src import span_functions
from sweapai.bimax_fitter import bimax_emcee_2, bimax_emcee_5

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

    # testing SPAN to FA grid conversion (projection)
    vpara, vperp = span_functions.project_SPANgrids_to_FAgrids(biMax, spangrids[tidx], bvec[tidx])

    # fitting a drifting bi-Maxwellian
    vdf_rec = span_L2.vdf.data[tidx] / np.nanmax(span_L2.vdf.data[tidx])

    # applying count mask
    countmask = span_L2.counts.data[tidx] > 2
    vdf_rec = vdf_rec[countmask]
    vpara = vpara[countmask]
    vperp = vperp[countmask]

    # flattening arrays
    vdf_rec = vdf_rec.flatten()
    vpara = vpara.flatten()
    vperp = vperp.flatten()

    # making the symmetric side
    vdf_rec = np.concatenate([vdf_rec, vdf_rec[::-1]])
    vpara = -np.concatenate([vpara, vpara[::-1]])
    vperp = np.concatenate([vperp, -vperp[::-1]])

    # supres_vdf = bimax_emcee.supres(vdf_rec, vpara, vperp)

    '''
    result = bimax_emcee_2.fit_bimaxwellian(
        vdf_rec,
        vpara,
        vperp,
        nwalkers=64,
        nsteps=5000,
        burn=2000,
        thin=10,
        run_mcmc=True,
    )

    bimax_emcee_2.print_fit_result(result["best"])

    bimax_emcee_2.plot_bimax_comparison(
        vdf_rec,
        vpara,
        vperp,
        result["best"],
    )
    '''

    result = bimax_emcee_5.fit_two_bimax_generic(
        vdf_rec,
        vpara,
        vperp,
        floor=1e-12,
        shoulder_side="higher",
        plot=True,
    )

    bimax_emcee_5.print_two_bimax_generic(result)